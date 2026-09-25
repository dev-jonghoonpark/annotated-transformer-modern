"""The Annotated Transformer의 모델 부분 — 원본 구조를 그대로 둔다.

harvardnlp/annotated-transformer (2018 Rush, 2022 Huang 외 개정)의 Part 1을 옮겼다.
클래스 이름과 구조는 원본과 같게 두어 원문과 나란히 읽을 수 있게 했고, 바꾼 것은 세 가지뿐이다.
- `super(Cls, self).__init__()` → `super().__init__()`
- `SublayerConnection`에 `norm_first` 인자를 달아 논문의 Post-LN도 켤 수 있게 했다(기본값은 원본대로 Pre-LN).
- `greedy_decode`를 문장 한 개가 아니라 배치로 돌게 했다. 매 스텝 앞부분 전체를 다시 계산하는 방식은 그대로다.

무엇이 낡았는지는 각 부분의 주석과 README의 대조표에 적었다. 현대판은 model_modern.py.
"""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    # 모델이 log_softmax까지 해서 내보내고, 손실은 KLDivLoss가 받는다(train.py의 LabelSmoothing).
    # 지금은 로짓을 그대로 내보내고 F.cross_entropy가 log_softmax와 NLL을 한 커널로 처리한다.
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # Pre-LN이라 마지막에 LayerNorm이 하나 더 필요하다(논문 Post-LN에는 없는 층).
        return self.norm(x)


class LayerNorm(nn.Module):
    # 원본의 손수 만든 LayerNorm. nn.LayerNorm과 두 군데가 다르다.
    # - x.std()는 불편분산(n-1로 나눔)이다. nn.LayerNorm은 n으로 나눈다.
    # - eps를 √ 밖에서 std에 더한다. nn.LayerNorm은 √(var + eps).
    # 결과 차이는 작지만 PyTorch의 fused 커널을 못 쓰고 연산 5개로 풀린다.
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    # 원본 docstring: "Note for code simplicity the norm is first as opposed to last."
    # 즉 원본 코드는 논문 본문(Post-LN: LayerNorm(x + Sublayer(x)))이 아니라 Pre-LN이다.
    # 튜토리얼이 설명하는 식과 실제로 도는 코드가 다른 셈이고, 결과적으로는 현대 쪽을 먼저 택했다.
    def __init__(self, size, dropout, norm_first=True):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, norm_first=True):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm_first), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, norm_first=True):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout, norm_first), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


def attention(query, key, value, mask=None, dropout=None):
    # 점수 행렬 (B, h, n, n)을 통째로 메모리에 만든다 — O(n²) 메모리.
    # 지금은 F.scaled_dot_product_attention 한 줄이 FlashAttention/메모리 효율 커널로
    # 이 행렬을 만들지 않고 같은 값을 낸다.
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # -1e9는 fp16 범위(±65504)를 넘어 half에서는 -inf가 되고, 한 행이 전부 가려지면 NaN이 난다.
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # Q·K·V·O 네 개 모두 bias가 있고 크기가 같다. 헤드마다 K·V를 따로 가진다(MHA).
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    # 논문은 "두 임베딩과 출력층이 같은 가중치를 공유한다"고 했지만 원본 코드는 공유하지 않는다.
    # (spacy로 언어별 사전을 따로 만들었으니 공유할 수가 없었다.)
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # 절대 위치를 입력에 더한다. 학습 때 본 길이(여기선 max_len=5000까지 표는 있지만,
    # 실제로 학습한 길이)를 넘어가면 성능이 무너진다. 현대 모델은 RoPE로 어텐션 안에서 상대 위치를 준다.
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, norm_first=True):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout, norm_first), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout, norm_first), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


@torch.no_grad()
def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol=None):
    """원본과 같이 매 스텝 지금까지 만든 번역문 전체를 디코더에 다시 넣는다.

    t번째 토큰을 만들 때 앞의 t-1개에 대한 K·V를 처음부터 또 계산하므로 한 문장에 O(n²)번의
    토큰 계산이 든다. 현대판은 KV 캐시로 새 토큰 하나만 계산한다(model_modern.generate).
    """
    memory = model.encode(src, src_mask)
    ys = torch.full((src.size(0), 1), start_symbol, dtype=src.dtype, device=src.device)
    done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
    for _ in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=-1)
        ys = torch.cat([ys, next_word[:, None]], dim=1)
        if end_symbol is not None:
            done |= next_word == end_symbol
            if done.all():
                break
    return ys
