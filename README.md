# annotated-transformer-modern

[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)(harvardnlp)를 **2026년 라이브러리 버전에서 그대로 돌아가게** 고친 저장소다. 모델·학습·시각화 코드와 본문은 원본 그대로이고, 지금 버전에서 설치가 안 되거나 깨지는 곳만 손봤다. 원본 대비 코드 변경은 [4곳](#원본에서-고친-것)이다.

- 원본: [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer) — Sasha Rush(2018), Austin Huang·Suraj Subramanian·Jonathan Sum·Khalid Almubarak·Stella Biderman(2022 개정). MIT 라이선스(`LICENSE`)
- 📚 시리즈: [how-ai-works](https://github.com/dev-jonghoonpark/how-ai-works)의 Transformer 섹션. 논문 내용은 [How Transformer Works](https://github.com/dev-jonghoonpark/how-transformer-works)에서 브라우저로 뜯어 보고, 여기서는 코드를 직접 돌려 본다.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dev-jonghoonpark/annotated-transformer-modern/blob/main/AnnotatedTransformer.ipynb)

## 원본이 지금 안 도는 이유

원본 `requirements.txt`는 `torch==1.11.0+cu113`, `torchtext==0.12`, `torchdata==0.3.0`, `spacy==3.2`, `altair==4.1`에 고정돼 있다.

| 막히는 곳 | 증상 |
| --- | --- |
| torch 1.11 | Python 3.10까지만 휠이 있다. 3.11 이상에서는 설치 자체가 안 된다 |
| torchtext | 0.18(2024년 4월)을 끝으로 개발이 중단됐다. 최신 0.18을 깔아도 torch 2.3용으로 빌드된 `libtorchtext.so`를 로드하지 못해 `import torchtext`에서 `OSError: Could not load this library`로 죽는다 |
| torchdata | 0.10에서 DataPipe가 삭제됐다(torchtext의 `datasets.Multi30k`가 여기에 의존) |
| altair 5+ | 학습률 스케줄 차트에서 `ValueError: "warmup" is not one of the valid encoding data types` |
| torch 2.6+ | 두 번째 실행부터 `torch.load("vocab.pt")`가 `UnpicklingError`(`weights_only` 기본값이 `True`로 바뀜) |

## 원본에서 고친 것

`the_annotated_transformer.py`의 원본 대비 diff 전체다. 이 밖의 코드와 본문은 한 글자도 바꾸지 않았다.

1. **torchtext import 3줄 → `torchtext_compat`** (L122–125)
   ```diff
   -from torchtext.data.functional import to_map_style_dataset
   +from torchtext_compat import to_map_style_dataset
    from torch.utils.data import DataLoader
   -from torchtext.vocab import build_vocab_from_iterator
   -import torchtext.datasets as datasets
   +from torchtext_compat import build_vocab_from_iterator
   +from torchtext_compat import datasets
   ```
   원본이 torchtext에서 쓰는 것은 이 세 가지뿐이라, 같은 이름·같은 동작의 대체품을 [`torchtext_compat.py`](torchtext_compat.py)(약 100줄)에 만들었다.
   - `datasets.Multi30k(language_pair=("de", "en"))`: 같은 29,000 / 1,014 / 1,000 분할을 HF Hub의 [bentrevett/multi30k](https://huggingface.co/datasets/bentrevett/multi30k)에서 받는다. 원본 코드가 `train + val + test`로 이어 붙이므로 리스트로 돌려준다.
   - `build_vocab_from_iterator`: torchtext 0.12와 같은 순서로 사전을 만든다(specials 먼저, 나머지는 빈도 내림차순, 동률이면 사전순, `min_freq` 적용). `vocab(tokens)`, `vocab["<blank>"]`, `get_stoi()`, `get_itos()`, `set_default_index()`를 지원한다.
   - `to_map_style_dataset`: `list()`. `DistributedSampler`가 `len()`을 요구해서 있던 함수다.
2. **altair 차트의 콜론 이스케이프** (L1123)
   ```diff
   -        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
   +        .encode(x="step", y="Learning Rate", color="model_size\\:warmup:N")
   ```
   altair 5부터 필드 이름 안의 `:`를 타입 구분자로 읽는다. 컬럼 이름이 `model_size:warmup`이라 이스케이프가 필요하다.
3. **`torch.load(..., weights_only=False)`** (L1464)
   ```diff
   -        vocab_src, vocab_tgt = torch.load("vocab.pt")
   +        vocab_src, vocab_tgt = torch.load("vocab.pt", weights_only=False)
   ```
   `vocab.pt`에는 텐서가 아니라 사전 객체가 들어 있다. torch 2.6부터 기본값인 `weights_only=True`로는 읽을 수 없다. 첫 실행은 사전을 새로 만들어 저장만 하므로 멀쩡하고, **두 번째 실행부터** 깨진다. 모델 가중치(`multi30k_model_final.pt`)는 state_dict라 그대로 둔다.
4. **Colab 설치 주석** (L104–105): 주석 처리된 `!pip install` 줄의 고정 버전을 걷어내고 `torchtext_compat.py`를 받는 줄을 넣었다.

코드 밖에서 바꾼 것은 하나다. 원본은 spaCy 모델이 없으면 노트북 안에서 `python -m spacy download`로 받는데, uv가 만든 가상환경에는 pip이 없어서 이 방식이 실패한다. 그래서 `de_core_news_sm`·`en_core_web_sm` 휠을 `pyproject.toml` 의존성에 직접 걸었다.

## 실행

[uv](https://docs.astral.sh/uv/)로 관리한다.

```bash
uv sync
uv run jupyter lab AnnotatedTransformer.ipynb     # 노트북으로 읽으며 실행
# 또는 원본 Makefile의 execute 타깃과 같은 일:
uv run jupytext --execute --to ipynb the_annotated_transformer.py
```

원본처럼 `the_annotated_transformer.py`가 소스이고, `.ipynb`는 jupytext로 만든 결과물이다. 이 저장소의 `AnnotatedTransformer.ipynb`는 아래 환경에서 처음부터 끝까지 실행한 출력을 담고 있다.

- `multi30k_model_final.pt`가 없으면 노트북이 Multi30k 독→영 모델을 8 epoch 학습한다(원본 설정 그대로: 배치 32, `accum_iter=10`, warmup 3000). 두 번째 실행부터는 저장된 사전과 가중치를 불러와 바로 시각화까지 간다.
- `example_simple_model`(복사 과제)은 원본에서도 `execute_example(...)` 호출이 주석 처리돼 있다. 따로 돌려 보려면 `uv run python -c "import the_annotated_transformer as t; t.example_simple_model()"`.

### 검증한 환경과 결과

Python 3.12 · torch 2.14.0+cu126 · spacy 3.8.16 · altair 6.3.0 · pandas 3.0.6 · datasets 5.0.1 · GPUtil 1.4.0 · jupytext 1.19.5. 정확한 조합은 `uv.lock`에 고정돼 있다. RTX 3070 8GB 한 장.

| 확인한 것 | 결과 |
| --- | --- |
| 노트북 전체 실행 (학습 포함) | 오류 없음. 8 epoch 학습에 약 13분, 약 3,900 tokens/s |
| 사전 크기 (min_freq=2) | 독일어 8,316 · 영어 6,384 |
| 검증 손실 (epoch 0 → 7) | 3.90 → 1.44 |
| 번역 예시 (greedy) | `Eine Frau in <unk> Sportkleidung fährt Rollschuh .` → `A woman in athletic clothes is rollerskating .` |
| 두 번째 실행 (사전·가중치 재사용) | 오류 없음, 약 40초 |
| `example_simple_model` (CPU) | 약 2분 30초. 출력 `[0, 2, 3, 2, 4, 5, 6, 7, 8, 9]`(정답은 0~9, 시드를 고정하지 않는 예제라 실행마다 다르다) |
| 분산 학습 (`distributed: True`) | GPU가 한 장이라 확인하지 못했다 |

## 원본 README의 나머지

jupytext 워크플로(`.py`가 소스, 노트북은 산출물)와 black/flake8 규칙은 [원본 README](https://github.com/harvardnlp/annotated-transformer#notebook-setup)를 따른다. 원본 Makefile의 `setup`·`move-dataset` 타깃(IWSLT 수동 다운로드)은 이 노트북에서 쓰지 않아 가져오지 않았다.
