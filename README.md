# annotated-transformer-modern

**[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)를 2026년에 다시 읽기.** 원본 코드를 오늘의 PyTorch에서 돌아가게 고치고, 같은 모델을 지금 오픈 LLM(Llama·Qwen·Gemma 계열)이 쓰는 부품으로 다시 짰습니다. 둘을 같은 데이터·같은 스텝 수로 나란히 학습해 **무엇이 바뀌었고, 바뀐 것이 이 규모에서 실제로 무엇을 사 주는지** 숫자로 확인합니다.

- 📦 **GitHub 레포**: https://github.com/dev-jonghoonpark/annotated-transformer-modern
- 📚 **시리즈**: [how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works)의 Transformer 섹션. 앞 자료 [How Transformer Works](https://github.com/dev-jonghoonpark/how-transformer-works)가 논문을 브라우저에서 뜯어 본다면, 이쪽은 코드를 직접 돌려 봅니다.
- 원본: Sasha Rush(2018), Austin Huang·Suraj Subramanian·Jonathan Sum·Khalid Almubarak·Stella Biderman(2022 개정) — [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)

```
model_2017.py    원본 Part 1(모델)을 거의 그대로. 클래스 이름·구조가 원문과 같아 나란히 읽힌다
model_modern.py  같은 Transformer를 RMSNorm·RoPE·SwiGLU·GQA·QK-norm·SDPA·KV 캐시로
data.py          torchtext/spacy 대신 HF datasets + 공유 BPE, 토큰 수 기준 배칭
train.py         두 레시피(2017: Adam+Noam+fp32 / 현대: AdamW+cosine+bf16), 학습 → test2016 BLEU
bench_decode.py  KV 캐시 유무에 따른 생성 속도, MHA vs GQA 캐시 크기
run_all.sh       아래 표를 만든 실행 전부
results/         실행별 손실 곡선·검증 NLL·번역 예시(JSON)와 벤치마크 출력
```

## 결과

Multi30k 독→영(학습 29,000문장), 모든 실행이 같은 배치(패딩 포함 6,000토큰)로 6,000스텝(38 epoch), greedy 디코딩, test2016 1,000문장의 sacreBLEU. RTX 3070 8GB 한 장.

| 실행 | 모델 | 레시피 | 파라미터 | BLEU (6000스텝, seed 0 / 1) | BLEU (검증 손실 최저) | 검증 NLL 최저 | 학습 시간 | 최대 메모리 |
|---|---|---|---|---|---|---|---|---|
| `2017` | 2017 | 2017 | 56.4M | 39.03 / 39.01 | 37.14 @2500 | 1.753 | 20.5분 | 5.86GiB |
| `2017-postln` | 2017 | 2017 | 56.4M | 13.40 | 12.60 @3500 | 2.870 | 20.5분 | 5.86GiB |
| `2017+modern` | 2017 | modern | 56.4M | 37.41 | 37.26 @2000 | 1.777 | 11.9분 | 4.51GiB |
| `modern-encdec+2017` | modern-encdec | 2017 | 41.9M | 39.42 / 37.42 | 37.34 @2000 | 1.725 | 16.3분 | 5.22GiB |
| `modern-encdec` | modern-encdec | modern | 41.9M | 37.56 / 38.77 | 38.08 @1000 | 1.792 | 9.3분 | 3.50GiB |
| `modern-deconly` | modern-deconly | modern | 43.6M | 37.24 / 36.76 | 35.31 @1000 | 1.855 | 13.1분 | 4.27GiB |
| `modern-deconly-nodrop` | modern-deconly | modern | 43.6M | 36.19 | 34.05 @1000 | 1.959 | 12.9분 | 4.15GiB |

- `2017` = 원본 구조 + 원본 레시피, `modern-encdec` = 부품만 현대식(인코더 6 + 디코더 6), `modern-deconly` = 인코더 없이 GPT식 디코더 14층. `A+B`는 모델 A를 레시피 B로 학습한 것.
- 파라미터 수가 다른 것은 원본이 임베딩 행렬 3개(원문·번역문·출력층, 각 8000×512)를 따로 두고 모든 Linear에 bias를 달았기 때문입니다. 임베딩을 빼면 원본 44.1M, 현대 인코더-디코더 37.8M, 디코더 전용 39.5M입니다.

### 읽는 법

1. **BLEU는 전부 37~39로 비슷합니다.** 같은 설정도 seed만 바꾸면 1~2 BLEU가 움직입니다(`modern-encdec` 37.56 ↔ 38.77, `modern-encdec+2017` 39.42 ↔ 37.42). 두 seed 모두 39.0으로 가장 안정적으로 높았던 것은 원본 `2017`이었습니다. **29,000문장짜리 과제에서는 현대 부품이 번역 품질을 올려 주지 않습니다.**
2. **대신 속도와 메모리가 좋아집니다.** 학습 20.5분 → 9.3분(2.2배), 최대 메모리 5.86 → 3.50GiB. 나눠 보면 레시피만 바꿨을 때(`2017+modern`) 11.9분·4.51GiB로 대부분이 여기서 나옵니다. bf16·TF32와, (N×V) 분포를 만들지 않는 cross-entropy 덕입니다. 구조만 바꿨을 때(`modern-encdec+2017`)는 16.3분·5.22GiB이고, 둘을 합쳐야 9.3분이 됩니다.
3. **수렴이 빠릅니다.** 검증 손실 최저점이 원본은 2,500스텝, 현대판은 1,000~1,500스텝입니다. 그 시점 BLEU가 이미 38.08/37.90이니, 원본이 20분을 들여 가는 곳에 현대판은 2분 안팎으로 도착합니다.
4. **2017년 코드에서 이미 결정적이었던 것은 Pre-LN입니다.** 논문 본문대로 Post-LN으로 되돌리면(`2017-postln`) 나머지는 전부 같아도 BLEU 13.4에 머뭅니다. 원본 튜토리얼이 "코드 단순화"라며 슬쩍 바꾼 한 줄이 사실은 오늘날의 표준이었습니다.
5. **디코더 전용도 번역이 됩니다. 다만 약간 낮습니다.** 파라미터가 비슷한데 37.24/36.76으로, 인코더-디코더(37.56/38.77)보다 0.3~2 BLEU 아래입니다. 원문도 causal하게, 즉 앞에서 뒤로만 읽어 양방향 문맥을 잃는 대가입니다. 규모가 커지면 이 차이보다 "모든 과제를 한 줄기로 푼다"는 단순함이 이긴다는 것이 GPT 계열의 선택이었습니다.
6. **드롭아웃은 여기서는 아직 필요합니다.** 요즘 LLM 사전학습은 드롭아웃이 0입니다. 데이터를 거의 한 번씩만 보니 외울 틈이 없기 때문입니다. 같은 29,000문장을 38번 도는 여기서 0으로 끄면(`modern-deconly-nodrop`) 검증 손실 1.855 → 1.959, BLEU 37.24 → 36.19로 나빠집니다.
7. **검증 손실과 BLEU는 따로 움직입니다.** 모든 실행에서 검증 NLL은 1,000~2,500스텝에 최저를 찍고 다시 올라갑니다(과적합). 하지만 BLEU는 11번 중 9번이 끝까지 돈 쪽이 비슷하거나 더 높았습니다. NLL은 틀린 토큰에 준 확신에 큰 벌점을 주지만 greedy 디코딩은 argmax만 보기 때문입니다. NLL 최저점에서 멈췄다면 디코더 전용은 BLEU를 2~3 잃었을 것입니다(35.31/33.42).

### KV 캐시 (`bench_decode.py`)

test2016 번역은 번역문이 평균 15토큰이라 디코딩 시간 차이가 작습니다(원본 재계산 6.5초, 현대 인코더-디코더 4.8초). 그래서 같은 디코더 전용 모델(학습 전, 14층)로 길이를 늘려 가며 두 방식을 따로 쟀습니다.

```
배치 8, bf16. 토큰 n개를 생성하는 데 걸린 시간
    n |      재계산 |    KV 캐시 | 배수
   32 |    0.28s |    0.30s |   0.9×
   64 |    0.56s |    0.60s |   0.9×
  128 |    1.09s |    1.18s |   0.9×
  256 |    2.42s |    2.37s |   1.0×
  512 |    7.34s |    4.82s |   1.5×
 1024 |   26.85s |    9.57s |   2.8×
MHA (원본, K·V 헤드 8): 토큰당 28KB, 4096토큰 × 배치 8 = 0.88GiB
GQA (현대판, K·V 헤드 2): 토큰당 7KB, 4096토큰 × 배치 8 = 0.22GiB
```

- **짧을 때는 캐시가 이기지 못합니다.** 이 크기에서는 GPU 계산보다 토큰 하나마다 14층 × 수십 개 커널을 띄우는 고정비(약 9ms)가 더 큽니다. 실서비스 엔진이 CUDA graph나 `torch.compile`로 이 고정비부터 없애는 이유입니다.
- **길어지면 차이가 벌어집니다.** 길이를 2배로 늘리면(512 → 1,024) 재계산은 3.7배, 캐시는 2.0배 느려집니다. 재계산은 O(n²), 캐시는 O(n)입니다.
- 대신 캐시는 메모리를 먹습니다. **GQA가 그 캐시를 ¼로 줄입니다.** 원본의 MHA는 토큰당 28KB, GQA(K·V 헤드 2개)는 7KB입니다.

## 원본이 지금 안 도는 이유

원본은 `torch==1.11.0+cu113`, `torchtext==0.12`, `torchdata==0.3.0`, `spacy==3.2`에 고정돼 있습니다.

- **torch 1.11**은 Python 3.10까지만 휠이 있습니다. 3.12가 기본인 지금 환경에서는 설치부터 막힙니다.
- **torchtext**는 0.18(2024년 4월)을 마지막으로 개발이 중단됐고, **torchdata**는 0.10에서 DataPipe를 삭제했습니다. 원본의 `datasets.Multi30k`, `build_vocab_from_iterator`, `to_map_style_dataset`이 전부 이 둘에 걸려 있습니다.

그래서 데이터 부분(`data.py`)은 새로 썼습니다. 데이터는 HF Hub의 [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)(같은 29,000/1,014/1,000 분할)에서 받고, spacy 단어 사전 대신 독일어·영어를 함께 학습한 **바이트 수준 BPE 8,000개**를 씁니다. 논문도 37K 공유 BPE였습니다. 바이트 수준이라 `<unk>`가 없고, 디코딩하면 원문 문자열이 그대로 나와 sacreBLEU에 바로 넣을 수 있습니다. 두 모델이 같은 토크나이저를 쓰므로 비교는 공정하지만, 원본 블로그의 결과와 숫자를 직접 맞대 볼 수는 없습니다.

## 원본 코드에서 알고 읽어야 할 것

튜토리얼의 설명(논문 문장)과 실제로 도는 코드가 다른 곳이 있습니다. `model_2017.py`는 이것들을 **고치지 않고 그대로 두고** 주석을 달았습니다.

| 원본 코드 | 논문 / 기대 | 결과 |
|---|---|---|
| `SublayerConnection`이 `x + dropout(sublayer(norm(x)))` — **Pre-LN** | 본문은 `LayerNorm(x + Sublayer(x))` — Post-LN | docstring에 "for code simplicity the norm is first"라고만 적혀 있다. 위 표의 `2017-postln`이 논문대로 바꾼 것인데, 워밍업 2,000스텝으로는 **BLEU 13.4**에 그친다. 이 코드가 학습되는 이유의 상당 부분이 이 한 줄이다 |
| 임베딩 3개를 따로 둔다 | "두 임베딩과 출력층이 같은 가중치를 공유"(3.4절) | spacy로 언어별 사전을 따로 만들었으니 공유할 수 없었다. 파라미터 12.3M이 임베딩 |
| `LayerNorm`을 손수 구현: `x.std()`(n−1로 나눔), `std + eps` | `nn.LayerNorm`은 n으로 나누고 `√(var + eps)` | 값 차이는 작지만 fused 커널을 못 쓴다 |
| `run_epoch`에서 `scheduler.step()`이 **마이크로배치마다** 불린다(`accum_iter=10`) | 워밍업은 옵티마이저 스텝 기준 | 설정의 `warmup: 3000`은 실제로 옵티마이저 스텝 300번이다. 손실을 `accum_iter`로 나누는 줄도 주석 처리돼 있다 |
| 사전을 `train + val + test`로 만든다 | 사전은 학습 데이터로만 | 테스트 문장의 단어가 사전에 들어간다(누수) |
| 32문장씩 묶고 전부 길이 72로 패딩, 72보다 길면 `F.pad`에 음수가 들어가 조용히 잘린다 | 길이가 비슷한 문장끼리, 배치당 약 25,000토큰 | 여기서는 토큰 수 기준 배칭(`data.token_batches`)으로 바꿨다 |
| `greedy_decode`가 문장 하나씩, 매 스텝 전체를 다시 계산 | — | 배치로 돌게만 바꾸고 재계산 방식은 그대로 두었다(KV 캐시와의 비교용) |
| 마스크를 `-1e9`로 채운다 | — | fp16 범위(±65,504)를 넘는다. bf16/fp32에서는 괜찮다 |

## 무엇을 바꿨나 — 2017 → 2026

### 모델 (`model_2017.py` → `model_modern.py`)

| 부품 | 2017 (원본) | 2026 (현대판) | 왜 | 출처 · 쓰는 곳 |
|---|---|---|---|---|
| 정규화 | LayerNorm(평균·분산, bias) | **RMSNorm**(제곱평균만, bias 없음) | 평균을 빼는 단계가 없어도 성능이 같고 더 싸다 | Zhang & Sennrich 2019 · Llama, Qwen, Gemma |
| 정규화 위치 | Pre-LN(코드) / Post-LN(논문) | **Pre-norm** + 마지막 norm | 깊게 쌓아도 워밍업 없이 학습된다([용어 사전: Post-LN/Pre-LN](https://github.com/dev-jonghoonpark/how-ai-works/blob/main/dictionary.md)) | Xiong 외 2020 · 사실상 전부 |
| 위치 | 사인파 절대 위치를 **입력에 더함** | **RoPE**: 어텐션 안에서 Q·K를 위치만큼 회전 | 점수가 상대 거리(m−n)에만 의존. 값·잔차 줄기에 위치 신호가 섞이지 않고, 길이 확장(YaRN 등)의 바탕이 된다 | Su 외 2021 · Llama, Qwen, Gemma, Mistral |
| FFN | ReLU, d → 4d → d | **SwiGLU**, d → ⅔·4d(두 갈래) → d | 게이트가 붙은 FFN이 같은 파라미터에서 더 낫다. 행렬이 3개라 폭을 ⅔로 줄여 파라미터를 맞춘다(512 → 1408) | Shazeer 2020 · PaLM, Llama |
| 어텐션 헤드 | MHA: Q·K·V 헤드 8개씩 | **GQA**: Q 8개가 K·V 2개를 나눠 씀 | 생성 때 병목인 KV 캐시가 ¼ | Ainslie 외 2023 · Llama 2 70B 이후 대부분 |
| 어텐션 안정화 | 없음 | **QK-norm**(헤드별 RMSNorm) | 로짓 폭주를 막아 큰 학습률에서도 안정 | Dehghani 외 2023 · OLMo 2, Qwen3, Gemma 3 |
| 어텐션 계산 | 손으로 `QKᵀ`, softmax, `-1e9` 마스크 | **`F.scaled_dot_product_attention`** | (n×n) 점수 행렬을 메모리에 만들지 않는 FlashAttention 계열 커널 | Dao 외 2022 · PyTorch 2.0+ |
| bias | 모든 Linear에 | **없음** | 규모가 커지면 bias 없이도 같고 더 안정적 | PaLM · Llama |
| 임베딩 | 3개 따로, `× √d` | 입력 임베딩 하나를 출력층과 **공유** | 공유 BPE라 논문 3.4절대로 공유할 수 있다 | Press & Wolf 2017 · Gemma, 작은 Qwen |
| 초기화 | Xavier uniform | N(0, 0.02), 잔차에 더하는 투영은 `/√(2L)` | 층이 쌓일수록 줄기가 커지는 것을 누른다 | GPT-2 |
| 생성 | 매 스텝 전체 재계산 | **KV 캐시** | 새 토큰 하나만 계산 | — |
| 구조 | 인코더-디코더 | 인코더-디코더 **와** 디코더 전용 둘 다 | 지금 LLM은 `<bos> 원문 <sep> 번역문`을 한 줄로 이어 디코더 하나로 푼다 | GPT 계열 |

디코더 전용에서는 **패딩 마스크가 아예 없습니다.** 오른쪽 패딩이면 진짜 토큰이 언제나 패딩보다 앞에 있어서 causal 마스크가 이미 패딩을 가립니다. 마스크 없이 `is_causal=True`만 주면 되니 FlashAttention 커널을 그대로 탑니다. 인코더는 양방향이라 여기서도 패딩 마스크가 필요합니다.

### 학습 레시피 (`train.py`)

| | 2017 | 2026 |
|---|---|---|
| 옵티마이저 | Adam(β₂=0.98, ε=1e-9) | AdamW(β₂=0.95, weight decay 0.1 — 행렬에만) |
| 학습률 | Noam: `d^−0.5 · min(s^−0.5, s·w^−1.5)`, 최고 9.9e-4 @2000 | 선형 워밍업 300 → 코사인으로 최고점의 10%까지, 최고 1e-3 |
| 정밀도 | fp32 (TF32도 끔) | bf16 autocast + TF32 (fp16과 달리 loss scaling이 필요 없다) |
| 레이블 스무딩 | 스무딩된 목표 분포 (N×V)를 실제로 만들어 `KLDivLoss` | `F.cross_entropy(label_smoothing=0.1)` — 분포를 만들지 않는다 |
| 그래디언트 클리핑 | 없음 | 전체 norm 1.0 |
| 드롭아웃 | 0.1 | 0.1 유지(아래 참고) |

## 실행

```bash
uv sync
uv run train.py --model 2017             # 원본 구조 + 원본 레시피
uv run train.py --model modern-encdec    # 현대 부품, 인코더-디코더
uv run train.py --model modern-deconly   # 현대 부품, 디코더 전용
uv run train.py --model 2017 --recipe modern   # 섞어서
uv run train.py --model 2017 --post-ln         # 논문대로 Post-LN
uv run bench_decode.py                   # KV 캐시 벤치마크
./run_all.sh && uv run summarize.py      # 위 표 전체 (RTX 3070에서 약 2.5시간)
```

8GB GPU 기준입니다. 원본 레시피는 레이블 스무딩 분포(N×V fp32)를 여러 벌 만들어서 메모리를 가장 많이 씁니다. 모자라면 `--max-tokens`를 줄이세요.

## 한계

- 번역 한 과제, 29,000문장. 현대 부품 대부분은 수십억 파라미터·수조 토큰·긴 문맥에서 이득을 보려고 나온 것이라, 이 규모에서 BLEU로 드러나지 않는 것은 자연스럽습니다. 이 레포가 보여 주는 것은 "현대 부품이 더 좋다"가 아니라 "무엇이 바뀌었고, 작은 규모에서도 드러나는 이득은 무엇인가"입니다.
- greedy 디코딩만 합니다(원본의 빔 서치·체크포인트 평균은 없음). 하이퍼파라미터는 두 레시피 모두 조정하지 않았습니다.
