"""results/*.json을 README용 표로 모은다.   uv run summarize.py"""

import json
from pathlib import Path

ORDER = ["2017", "2017-postln", "2017+modern", "modern-encdec+2017", "modern-encdec", "modern-deconly",
         "modern-deconly-nodrop"]


def main():
    rows = {}
    for f in sorted((Path(__file__).parent / "results").glob("*.json")):
        r = json.loads(f.read_text())
        rows[r["name"]] = r
    print("| 실행 | 모델 | 레시피 | 파라미터 | BLEU (6000스텝, seed 0 / 1) | BLEU (검증 손실 최저) | 검증 NLL 최저 | 학습 시간 | 최대 메모리 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in ORDER + sorted(n for n in set(rows) - set(ORDER) if not n.endswith("-s1")):
        if name not in rows:
            continue
        r = rows[name]
        seed = rows.get(f"{name}-s1")
        bleu = f"{r['bleu']:.2f}" + (f" / {seed['bleu']:.2f}" if seed else "")
        best = f"{r['best_bleu']:.2f} @{r['best_step']}" if "best_bleu" in r else "—"
        nll = f"{r['best_valid_nll']:.3f}" if "best_valid_nll" in r else "—"
        print(f"| `{name}` | {r['model']} | {r['recipe']} | {r['params'] / 1e6:.1f}M | {bleu} | {best} | {nll} | "
              f"{r['train_sec'] / 60:.1f}분 | {r['peak_mem_gib']:.2f}GiB |")


if __name__ == "__main__":
    main()
