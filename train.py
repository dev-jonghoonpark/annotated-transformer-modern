"""Multi30k de→en으로 세 모델을 같은 조건에서 학습하고 test2016 BLEU까지 잰다.

    uv run train.py --model 2017             # 원본 구조 + 원본 학습 레시피
    uv run train.py --model modern-encdec    # 현대 부품 + 현대 레시피, 인코더-디코더
    uv run train.py --model modern-deconly   # 현대 부품 + 현대 레시피, 디코더 전용

모델과 레시피는 따로 고를 수 있다(예: --model 2017 --recipe modern). 데이터·배치·스텝 수는 전부 같다.
결과는 results/<이름>.json에, 가중치는 checkpoints/<이름>.pt에 남는다.
"""

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F

import model_2017
import model_modern
from data import BOS, EOS, PAD, SEP, encode_pairs, lm_tensors, load_multi30k, load_tokenizer, seq2seq_tensors, token_batches

ROOT = Path(__file__).parent


# ── 2017 레시피: 원본 Part 2 그대로 ─────────────────────────────────────────────


def rate(step, model_size, factor, warmup):
    """Noam 스케줄. 최고점은 step = warmup에서 d^−0.5 · warmup^−0.5."""
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    """원본 구현: 스무딩한 목표 분포를 (N, V) 크기로 실제로 만들어 KLDivLoss에 넣는다.

    지금은 F.cross_entropy(label_smoothing=0.1)가 이 분포를 만들지 않고 같은 일을 한다.
    (분배 방식은 조금 다르다 — 원본은 ε을 정답·패딩을 뺀 V−2개에, PyTorch는 정답 포함 V개에 나눈다.)
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return self.criterion(x, true_dist.clone().detach())


# ── 현대 레시피 ────────────────────────────────────────────────────────────────


def warmup_cosine(step, warmup, total, floor=0.1):
    """선형 워밍업 뒤 코사인으로 최고점의 floor배까지. LambdaLR에 넣는 배율이다."""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def adamw_param_groups(model, weight_decay):
    # 행렬(임베딩·Linear)에만 weight decay. 정규화 층의 스케일(1차원)은 0으로 끌어당기면 안 된다.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# ── 모델별 어댑터: 배치 → (점수, 정답) ────────────────────────────────────────────


def build_model(args, vocab):
    if args.model == "2017":
        return model_2017.make_model(vocab, vocab, N=6, dropout=args.dropout, norm_first=not args.post_ln)
    cfg = model_modern.Config(vocab=vocab, dropout=args.dropout)
    if args.model == "modern-encdec":
        return model_modern.EncoderDecoder(cfg)
    cfg.n_dec_layer = args.dec_layers
    return model_modern.DecoderOnly(cfg)


def forward_batch(model, kind, batch, device):
    """점수(2017은 log-확률, 현대판은 로짓)와 맞힐 토큰을 돌려준다. 무시할 자리는 PAD."""
    if kind == "modern-deconly":
        seq, labels = (t.to(device, non_blocking=True) for t in lm_tensors(batch))
        labels = labels.masked_fill(labels == -100, PAD)
        return model(seq), labels
    src, tgt = (t.to(device, non_blocking=True) for t in seq2seq_tensors(batch))
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    if kind == "2017":
        # 원본 Batch 클래스와 같은 마스크: 원문은 패딩만, 번역문은 패딩 & 미래.
        src_mask = (src != PAD).unsqueeze(-2)
        tgt_mask = (tgt_in != PAD).unsqueeze(-2) & model_2017.subsequent_mask(tgt_in.size(-1)).to(device)
        out = model(src, tgt_in, src_mask, tgt_mask)
        return model.generator(out), tgt_out
    return model(src, tgt_in), tgt_out


@torch.no_grad()
def translate(model, kind, examples, device, amp, max_batch=128):
    """greedy 번역. 원문 길이가 같은 문장끼리 묶어서 패딩 없이 돌린다."""
    by_len = defaultdict(list)
    for i, (s, _) in enumerate(examples):
        by_len[len(s)].append(i)
    hyps = [None] * len(examples)
    gen_tokens = 0
    for n, idx in sorted(by_len.items()):
        max_new = min(int(1.5 * n) + 10, 128)
        for start in range(0, len(idx), max_batch):
            chunk = idx[start : start + max_batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                if kind == "2017":
                    src = torch.tensor([[BOS, *examples[i][0], EOS] for i in chunk], device=device)
                    ys = model_2017.greedy_decode(model, src, torch.ones_like(src[:, None, :], dtype=torch.bool),
                                                  max_new + 1, BOS, EOS)[:, 1:]
                elif kind == "modern-encdec":
                    src = torch.tensor([[BOS, *examples[i][0], EOS] for i in chunk], device=device)
                    ys = model.generate(src, max_new, BOS, EOS)
                else:
                    prompt = torch.tensor([[BOS, *examples[i][0], SEP] for i in chunk], device=device)
                    ys = model.generate(prompt, max_new, EOS)
            for i, row in zip(chunk, ys.tolist()):
                row = row[: row.index(EOS)] if EOS in row else row
                hyps[i] = row
                gen_tokens += len(row) + 1
    return hyps, gen_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["2017", "modern-encdec", "modern-deconly"], required=True)
    p.add_argument("--recipe", choices=["2017", "modern"], default=None, help="기본값: 모델에 맞는 쪽")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--max-tokens", type=int, default=6000, help="배치당 (src+tgt) 패딩 포함 토큰 수 상한")
    p.add_argument("--warmup", type=int, default=None, help="기본값: 2017=2000(Noam), modern=300")
    p.add_argument("--lr", type=float, default=1e-3, help="현대 레시피의 최고 학습률")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--dec-layers", type=int, default=14, help="modern-deconly의 층 수")
    p.add_argument("--post-ln", action="store_true", help="2017 모델을 논문대로 Post-LN으로")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    recipe = args.recipe or ("2017" if args.model == "2017" else "modern")
    warmup = args.warmup or (2000 if recipe == "2017" else 300)
    name = args.name or (args.model if recipe == ("2017" if args.model == "2017" else "modern") else f"{args.model}+{recipe}")

    torch.manual_seed(args.seed)
    device = "cuda"
    amp = recipe == "modern"
    # 2017 레시피는 순수 fp32(TF32도 끔). 현대 레시피는 bf16 autocast + TF32 matmul.
    torch.backends.cuda.matmul.allow_tf32 = amp
    torch.backends.cudnn.allow_tf32 = amp

    data = load_multi30k()
    tok = load_tokenizer(data["train"])
    train = encode_pairs(tok, data["train"])
    valid = encode_pairs(tok, data["validation"])
    test = encode_pairs(tok, data["test"])
    vocab = tok.get_vocab_size()

    model = build_model(args, vocab).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] params {n_params / 1e6:.2f}M  vocab {vocab}  recipe {recipe}  warmup {warmup}")

    if recipe == "2017":
        criterion = LabelSmoothing(size=vocab, padding_idx=PAD, smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: rate(s + 1, 512, 1.0, warmup))

        def loss_fn(scores, target):
            logp = scores if args.model == "2017" else F.log_softmax(scores.float(), dim=-1)
            ntok = (target != PAD).sum()
            return criterion(logp.reshape(-1, vocab), target.reshape(-1)) / ntok
    else:
        optimizer = torch.optim.AdamW(adamw_param_groups(model, 0.1), lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
                                      fused=True)
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine(s, warmup, args.steps))

        def loss_fn(scores, target):
            return F.cross_entropy(scores.float().reshape(-1, vocab), target.reshape(-1), ignore_index=PAD,
                                   label_smoothing=0.1)

    train_model = torch.compile(model) if args.compile else model

    @torch.no_grad()
    def valid_nll():
        # 비교용 검증 손실은 레시피와 상관없이 스무딩 없는 토큰당 NLL(= log 퍼플렉시티).
        model.eval()
        tot, n = 0.0, 0
        for batch in token_batches(valid, args.max_tokens, shuffle=False):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                scores, target = forward_batch(model, args.model, batch, device)
            tot += F.cross_entropy(scores.float().reshape(-1, vocab), target.reshape(-1), ignore_index=PAD,
                                   reduction="sum").item()
            n += (target != PAD).sum().item()
        model.train()
        return tot / n

    history = []
    # 데이터가 작아서(29,000문장) 끝까지 돌면 과적합한다. 검증 손실이 가장 낮았던 가중치도 따로 둔다.
    best = {"nll": float("inf"), "step": 0, "state": None}
    step, epoch = 0, 0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    window_loss, window_tok, t_win = 0.0, 0, time.time()
    model.train()
    while step < args.steps:
        for batch in token_batches(train, args.max_tokens, seed=args.seed * 1000 + epoch):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                scores, target = forward_batch(train_model, args.model, batch, device)
            loss = loss_fn(scores, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if recipe == "modern":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            step += 1
            window_loss += loss.item()
            window_tok += sum(len(s) + len(t) for s, t in batch)
            if step % 100 == 0:
                dt = time.time() - t_win
                print(f"step {step:5d} ep {epoch:2d} | loss {window_loss / 100:.3f} | "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} | {window_tok / dt:,.0f} tok/s")
                history.append({"step": step, "train_loss": window_loss / 100,
                                "lr": optimizer.param_groups[0]["lr"]})
                window_loss, window_tok, t_win = 0.0, 0, time.time()
            if step % args.eval_every == 0 or step == args.steps:
                v = valid_nll()
                history[-1]["valid_nll"] = v
                print(f"  valid nll {v:.3f}  ppl {math.exp(v):.2f}")
                if v < best["nll"]:
                    best.update(nll=v, step=step,
                                state={k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()})
            if step >= args.steps:
                break
        epoch += 1
    torch.cuda.synchronize()
    train_sec = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 2**30

    model.eval()
    refs = [en for _, en in data["test"]]
    torch.cuda.synchronize()
    t1 = time.time()
    hyps, gen_tokens = translate(model, args.model, test, device, amp)
    torch.cuda.synchronize()
    decode_sec = time.time() - t1
    hyp_text = tok.decode_batch(hyps)
    bleu = sacrebleu.corpus_bleu(hyp_text, [refs])
    print(f"[{name}] test2016 BLEU {bleu.score:.2f} @ {step}  ({bleu})")
    print(f"  train {train_sec / 60:.1f}min  peak {peak_mem:.2f}GiB  decode {decode_sec:.1f}s "
          f"({gen_tokens / decode_sec:,.0f} tok/s)")

    final_state = {k: t.detach().to("cpu", copy=True) for k, t in model.state_dict().items()}
    model.load_state_dict(best["state"])
    best_hyps, _ = translate(model, args.model, test, device, amp)
    best_text = tok.decode_batch(best_hyps)
    best_bleu = sacrebleu.corpus_bleu(best_text, [refs])
    print(f"[{name}] test2016 BLEU {best_bleu.score:.2f} @ {best['step']} (검증 손실 최저 {best['nll']:.3f})")

    result = {
        "name": name, "model": args.model, "recipe": recipe, "params": n_params, "steps": args.steps,
        "epochs": epoch, "max_tokens": args.max_tokens, "warmup": warmup, "dropout": args.dropout,
        "post_ln": args.post_ln, "dec_layers": args.dec_layers if args.model == "modern-deconly" else None,
        "train_sec": train_sec, "peak_mem_gib": peak_mem, "decode_sec": decode_sec, "gen_tokens": gen_tokens,
        "bleu": bleu.score, "bleu_str": str(bleu), "final_valid_nll": history[-1].get("valid_nll"),
        "best_step": best["step"], "best_valid_nll": best["nll"], "best_bleu": best_bleu.score,
        "best_bleu_str": str(best_bleu),
        "history": history,
        "samples": [{"de": data["test"][i][0], "ref": refs[i], "hyp": hyp_text[i], "hyp_best": best_text[i]}
                    for i in range(10)],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save(final_state, ROOT / "checkpoints" / f"{name}.pt")
    torch.save(best["state"], ROOT / "checkpoints" / f"{name}-best.pt")

if __name__ == "__main__":
    main()
