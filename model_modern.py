"""같은 Transformer를 2026년의 방식으로 — Llama·Qwen·Gemma 계열 오픈 모델들이 공통으로 쓰는 부품만 골랐다.

model_2017.py와 한 줄씩 대응시켜 보면 바뀐 것은 다음과 같다(README의 대조표에 이유와 출처).

    LayerNorm(손수 구현, bias 있음)  → nn.RMSNorm (평균을 빼지 않고, bias 없음)
    사인파 절대 위치 인코딩을 입력에 더함 → RoPE: 어텐션 안에서 Q·K를 위치만큼 회전
    ReLU FFN (d → 4d → d)            → SwiGLU (d → ⅔·4d 두 갈래 → d, 파라미터 수는 같게)
    멀티헤드 어텐션 (K·V 헤드 8개)     → GQA (Q 헤드 8개가 K·V 헤드 2개를 나눠 씀) + QK-norm
    attention() 손수 구현 + -1e9 마스크 → F.scaled_dot_product_attention (FlashAttention 커널)
    Linear에 bias                   → bias 없음
    임베딩 3개 따로                   → 입력 임베딩 하나를 출력층과 공유(weight tying)
    매 스텝 전체 재계산 greedy        → KV 캐시

그리고 모델 구조 자체로 두 가지를 준비했다.
- EncoderDecoder: 위 부품으로 원본과 같은 인코더 6층 + 디코더 6층을 만든 것. "부품만" 바꾼 효과를 본다.
- DecoderOnly: 인코더 없이 디코더 한 줄기에 "<bos> 원문 <sep> 번역문"을 이어 넣는 GPT식. 지금 LLM의 모양.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import PAD


@dataclass
class Config:
    vocab: int
    d_model: int = 512
    n_head: int = 8
    n_kv_head: int = 2
    # SwiGLU는 가중치 행렬이 3개라 은닉을 4d의 ⅔로 줄여 ReLU FFN(2개 × 4d)과 파라미터 수를 맞춘다.
    # 4·512·⅔ = 1365를 64의 배수로 올려 1408 (Llama도 같은 규칙).
    ffn_hidden: int = 1408
    n_enc_layer: int = 6
    n_dec_layer: int = 6
    dropout: float = 0.1
    rope_theta: float = 10000.0
    max_len: int = 1024


class RotaryEmbedding(nn.Module):
    """RoPE (Su 외, RoFormer 2021). 위치 p에서 Q·K의 차원 쌍을 각도 p·θ_i만큼 회전한다.

    회전한 q_m과 k_n의 내적은 (m − n)에만 의존한다 — 절대 위치를 입력에 더하는 대신
    어텐션 점수에 상대 거리를 직접 새긴다. 값(V)과 잔차 줄기에는 위치 신호가 섞이지 않는다.
    """

    def __init__(self, head_dim, max_len, theta):
        super().__init__()
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x, pos0=0):
        # x: (B, h, T, head_dim). 캐시로 한 토큰씩 생성할 때는 pos0이 지금까지의 길이다.
        T = x.size(-2)
        cos = self.cos[pos0 : pos0 + T].to(x.dtype)
        sin = self.sin[pos0 : pos0 + T].to(x.dtype)
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_head
        self.h_kv = cfg.n_kv_head
        self.hd = cfg.d_model // cfg.n_head
        self.q_proj = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        # GQA: K·V는 헤드 2개분만 만든다. KV 캐시가 원본(8헤드)의 ¼이 된다.
        self.k_proj = nn.Linear(cfg.d_model, self.h_kv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.h_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)
        # QK-norm: 헤드별로 q·k를 RMSNorm해서 로짓 크기가 폭주하지 않게 한다(OLMo 2, Qwen3, Gemma 3).
        self.q_norm = nn.RMSNorm(self.hd)
        self.k_norm = nn.RMSNorm(self.hd)

    def forward(self, x, kv=None, rope=None, pos0=0, mask=None, is_causal=False, cache=None):
        """kv가 None이면 셀프 어텐션, 아니면 kv(인코더 출력)를 보는 교차 어텐션.

        cache는 층마다 하나씩 있는 dict. 셀프 어텐션은 지난 스텝의 K·V 뒤에 이번 것을 붙이고,
        교차 어텐션은 인코더 출력이 고정이라 첫 스텝에 한 번만 계산해 둔다.
        """
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(B, T, self.h, self.hd)).transpose(1, 2)
        if kv is not None and cache is not None and "k" in cache:
            k, v = cache["k"], cache["v"]
        else:
            src = x if kv is None else kv
            S = src.size(1)
            k = self.k_norm(self.k_proj(src).view(B, S, self.h_kv, self.hd)).transpose(1, 2)
            v = self.v_proj(src).view(B, S, self.h_kv, self.hd).transpose(1, 2)
            if rope is not None:
                k = rope(k, pos0)
            if kv is None and cache is not None and "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            if cache is not None:
                cache["k"], cache["v"] = k, v
        if rope is not None:
            q = rope(q, pos0)
        # 점수 행렬을 메모리에 만들지 않는 fused 커널. enable_gqa=True가 K·V 헤드를 Q 헤드 수만큼 나눠 쓴다.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """Shazeer, 「GLU Variants Improve Transformer」(2020). 한 갈래(gate)가 다른 갈래(up)를 곱해서 연다."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm 잔차 블록. 줄기(x)는 더해지기만 하고, 정규화는 가지로 들어가는 입력에만 한다."""

    def __init__(self, cfg: Config, cross=False):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.cross_norm = nn.RMSNorm(cfg.d_model) if cross else None
        self.cross = Attention(cfg) if cross else None
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope, pos0=0, mask=None, is_causal=False, memory=None, memory_mask=None, cache=None):
        x = x + self.drop(
            self.attn(self.attn_norm(x), rope=rope, pos0=pos0, mask=mask, is_causal=is_causal,
                      cache=None if cache is None else cache.setdefault("self", {}))
        )
        if self.cross is not None:
            # 교차 어텐션에는 RoPE를 걸지 않는다. 원문과 번역문의 위치 차이는 의미가 없다.
            x = x + self.drop(
                self.cross(self.cross_norm(x), kv=memory, mask=memory_mask,
                           cache=None if cache is None else cache.setdefault("cross", {}))
            )
        return x + self.drop(self.ffn(self.ffn_norm(x)))


def _init_weights(model, n_residual_layers):
    for name, p in model.named_parameters():
        if p.dim() < 2:
            continue
        std = 0.02
        # 잔차 줄기에 더해지는 마지막 투영은 층 수만큼 작게 (GPT-2). 쌓을수록 줄기가 커지는 것을 누른다.
        if name.endswith(("o_proj.weight", "down.weight")):
            std = 0.02 / math.sqrt(2 * n_residual_layers)
        nn.init.normal_(p, mean=0.0, std=std)


class DecoderOnly(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_head, cfg.max_len, cfg.rope_theta)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_dec_layer))
        self.norm = nn.RMSNorm(cfg.d_model)
        _init_weights(self, cfg.n_dec_layer)

    def forward(self, idx, caches=None, pos0=0):
        x = self.drop(self.embed(idx))
        # 오른쪽 패딩이면 패딩 마스크가 필요 없다. 진짜 토큰은 언제나 패딩보다 앞에 있어서
        # causal 마스크가 이미 패딩을 가린다. 마스크가 없으니 FlashAttention 커널을 그대로 탄다.
        # 캐시로 한 토큰씩 넣을 때는 그 토큰이 앞을 전부 봐야 하므로 causal을 끈다.
        is_causal = idx.size(1) > 1
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, pos0=pos0, is_causal=is_causal,
                      cache=None if caches is None else caches[i])
        # 출력층 = 입력 임베딩의 전치(weight tying). 별도 Generator 행렬이 없다.
        return F.linear(self.norm(x), self.embed.weight)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, eos):
        """prompt: (B, P) — 길이가 같은 원문끼리 묶어 넣는다(왼쪽 패딩과 위치 보정이 필요 없다)."""
        caches = [{} for _ in self.blocks]
        logits = self(prompt, caches)  # prefill: 프롬프트 전체를 한 번에
        out = []
        done = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)
        pos = prompt.size(1)
        for _ in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1)
            out.append(nxt)
            done |= nxt == eos
            if done.all():
                break
            logits = self(nxt[:, None], caches, pos0=pos)  # 새 토큰 하나만 계산
            pos += 1
        return torch.stack(out, dim=1)


class EncoderDecoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 원문·번역문이 BPE 사전을 공유하므로 임베딩도 하나(논문 3.4절의 공유를 이번엔 실제로 한다).
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_head, cfg.max_len, cfg.rope_theta)
        self.enc = nn.ModuleList(Block(cfg) for _ in range(cfg.n_enc_layer))
        self.enc_norm = nn.RMSNorm(cfg.d_model)
        self.dec = nn.ModuleList(Block(cfg, cross=True) for _ in range(cfg.n_dec_layer))
        self.dec_norm = nn.RMSNorm(cfg.d_model)
        _init_weights(self, cfg.n_enc_layer + cfg.n_dec_layer)

    def encode(self, src):
        # (B, 1, 1, S) — True인 키만 본다. 인코더는 양방향이라 여기서는 패딩 마스크가 필요하다.
        mask = (src != PAD)[:, None, None, :]
        x = self.drop(self.embed(src))
        for block in self.enc:
            x = block(x, self.rope, mask=mask)
        return self.enc_norm(x), mask

    def decode(self, tgt, memory, memory_mask, caches=None, pos0=0):
        x = self.drop(self.embed(tgt))
        is_causal = tgt.size(1) > 1
        for i, block in enumerate(self.dec):
            x = block(x, self.rope, pos0=pos0, is_causal=is_causal, memory=memory, memory_mask=memory_mask,
                      cache=None if caches is None else caches[i])
        return F.linear(self.dec_norm(x), self.embed.weight)

    def forward(self, src, tgt):
        memory, mask = self.encode(src)
        return self.decode(tgt, memory, mask)

    @torch.no_grad()
    def generate(self, src, max_new_tokens, bos, eos):
        memory, mask = self.encode(src)
        caches = [{} for _ in self.dec]
        ys = torch.full((src.size(0), 1), bos, dtype=src.dtype, device=src.device)
        out = []
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for pos in range(max_new_tokens):
            logits = self.decode(ys, memory, mask, caches, pos0=pos)
            ys = logits[:, -1].argmax(-1, keepdim=True)
            out.append(ys[:, 0])
            done |= ys[:, 0] == eos
            if done.all():
                break
        return torch.stack(out, dim=1)
