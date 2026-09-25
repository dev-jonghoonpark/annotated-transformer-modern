"""KV 캐시가 없으면 생성이 얼마나 느려지는가 — 같은 모델로 두 방식만 바꿔 잰다.

    uv run bench_decode.py

Multi30k 문장은 번역문이 평균 15토큰이라 train.py의 디코딩 시간에서는 차이가 잘 안 보인다.
여기서는 학습하지 않은 DecoderOnly(train.py와 같은 크기)로 길이를 늘려 가며
- 원본 greedy_decode 방식: 매 스텝 지금까지의 시퀀스 전체를 다시 넣는다
- KV 캐시: 새 토큰 하나만 넣고, 지난 K·V는 캐시에서 꺼낸다
의 스텝당 시간을 비교하고, 캐시 크기를 MHA(K·V 헤드 8개)와 GQA(2개)로 나눠 계산한다.
"""

import time

import torch

from model_modern import Config, DecoderOnly


@torch.no_grad()
def time_generation(model, n_tokens, use_cache, batch=8):
    x = torch.randint(4, model.cfg.vocab, (batch, 1), device="cuda")
    torch.cuda.synchronize()
    t = time.time()
    if use_cache:
        caches = [{} for _ in model.blocks]
        logits = model(x, caches)
        for pos in range(1, n_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            logits = model(nxt, caches, pos0=pos)
    else:
        seq = x
        for _ in range(1, n_tokens):
            nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
    torch.cuda.synchronize()
    return time.time() - t


def main():
    torch.manual_seed(0)
    model = DecoderOnly(Config(vocab=8000, n_dec_layer=14, dropout=0.0)).cuda().eval()
    print("배치 8, bf16. 토큰 n개를 생성하는 데 걸린 시간")
    print(f"{'n':>5} | {'재계산':>8} | {'KV 캐시':>8} | 배수")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        time_generation(model, 16, True)
        time_generation(model, 16, False)
        for n in [32, 64, 128, 256, 512, 1024]:
            a = time_generation(model, n, False)
            b = time_generation(model, n, True)
            print(f"{n:>5} | {a:7.2f}s | {b:7.2f}s | {a / b:5.1f}×")

    cfg = model.cfg
    hd = cfg.d_model // cfg.n_head
    for label, h_kv in [("MHA (원본, K·V 헤드 8)", cfg.n_head), ("GQA (현대판, K·V 헤드 2)", cfg.n_kv_head)]:
        per_token = 2 * cfg.n_dec_layer * h_kv * hd * 2  # K와 V, 층마다, bf16 2바이트
        print(f"{label}: 토큰당 {per_token / 1024:.0f}KB, 4096토큰 × 배치 8 = {per_token * 4096 * 8 / 2**30:.2f}GiB")


if __name__ == "__main__":
    main()
