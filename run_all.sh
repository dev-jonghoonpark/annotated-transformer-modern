#!/usr/bin/env bash
# README의 표를 만든 실행 전부. RTX 3070 8GB 한 장에서 합계 약 2.5시간.
set -euo pipefail
cd "$(dirname "$0")"
run() { uv run train.py "$@" 2>&1 | grep -v "split\|Warning" | tee -a results/log.txt; }
mkdir -p results
run --model 2017                                   # 원본 코드 그대로 (Pre-LN)
run --model 2017 --post-ln --name 2017-postln      # 논문 본문대로 Post-LN
run --model modern-encdec                          # 부품만 현대식
run --model modern-deconly                         # 디코더 전용 (GPT식)
run --model 2017 --recipe modern                   # 원본 구조 + 현대 레시피
run --model modern-encdec --recipe 2017            # 현대 구조 + 원본 레시피
run --model modern-deconly --dropout 0 --name modern-deconly-nodrop  # 요즘 LLM처럼 드롭아웃 0
# seed 편차 확인
run --model 2017 --seed 1 --name 2017-s1
run --model modern-encdec --seed 1 --name modern-encdec-s1
run --model modern-encdec --recipe 2017 --seed 1 --name modern-encdec+2017-s1
run --model modern-deconly --seed 1 --name modern-deconly-s1
uv run bench_decode.py | tee results/bench_decode.txt
