"""Multi30k(de→en) 불러오기, 공유 BPE 토크나이저, 토큰 수 기준 배칭.

원본(2022판)은 이 부분을 torchtext + torchdata + spacy로 했다.
- torchtext는 0.18(2024)을 끝으로 개발이 중단됐고, torchdata의 DataPipe는 0.10에서 삭제됐다.
  원본이 고정한 torch 1.11은 Python 3.10까지만 휠이 있어 지금 환경에서는 설치부터 막힌다.
- spacy로 단어를 자르고 언어별 사전(min_freq=2)을 따로 만들었는데, 사전에 없는 단어는 전부 <unk>가 된다.

여기서는 데이터를 HF Hub(bentrevett/multi30k, 같은 29,000/1,014/1,000 분할)에서 받고,
독일어·영어를 한꺼번에 학습한 바이트 수준 BPE 하나를 쓴다. 논문도 37K 공유 BPE였다.
바이트 수준이라 <unk>가 없고, 디코딩하면 원문 문자열이 그대로 복원돼 sacreBLEU에 바로 넣을 수 있다.
"""

import random
from pathlib import Path

import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA_DIR = Path(__file__).parent / "data"
TOKENIZER_PATH = DATA_DIR / "bpe.json"

PAD, BOS, EOS, SEP = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>"]


def load_multi30k():
    ds = load_dataset("bentrevett/multi30k", cache_dir=str(DATA_DIR / "hf"))
    return {split: list(zip(ds[split]["de"], ds[split]["en"])) for split in ds}


def load_tokenizer(train_pairs, vocab_size=8000):
    if TOKENIZER_PATH.exists():
        return Tokenizer.from_file(str(TOKENIZER_PATH))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    # 학습 분할만으로 학습한다. 원본은 train+val+test로 사전을 만들었다(테스트 누수).
    tok.train_from_iterator((s for pair in train_pairs for s in pair), trainer)
    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOKENIZER_PATH))
    return tok


def encode_pairs(tok, pairs, max_len=128):
    src = tok.encode_batch([de for de, _ in pairs])
    tgt = tok.encode_batch([en for _, en in pairs])
    # 원본은 max_padding보다 긴 문장에서 F.pad에 음수가 들어가 조용히 잘렸다. 여기서는 명시적으로 자른다.
    return [(s.ids[:max_len], t.ids[:max_len]) for s, t in zip(src, tgt)]


def token_batches(examples, max_tokens, shuffle=True, seed=0):
    """길이가 비슷한 문장끼리 묶어 (src+tgt 최대 길이 × 문장 수)가 max_tokens를 넘지 않게 한다.

    원본은 32문장씩 무작위로 묶은 뒤 전부 72로 패딩했다. 논문은 "길이가 비슷한 문장쌍끼리,
    배치당 원문·번역문 토큰 약 25,000개"였다 — 이쪽이 논문에 가깝다.
    """
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    # 완전히 정렬하면 매 epoch 같은 배치가 나오므로 100배치 분량씩 끊어서 정렬한다.
    chunk = 100 * max(1, max_tokens // 40)
    batches = []
    for start in range(0, len(idx), chunk):
        part = sorted(idx[start : start + chunk], key=lambda i: len(examples[i][0]) + len(examples[i][1]))
        batch, longest = [], 0
        for i in part:
            n = len(examples[i][0]) + len(examples[i][1]) + 3
            if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
                batches.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, n)
        if batch:
            batches.append(batch)
    if shuffle:
        rng.shuffle(batches)
    return [[examples[i] for i in b] for b in batches]


def pad_to(seqs, value=PAD, multiple=8):
    """오른쪽 패딩. 길이를 8의 배수로 맞추면 텐서 코어 효율이 좋고 torch.compile 재컴파일도 준다."""
    n = max(len(s) for s in seqs)
    n = (n + multiple - 1) // multiple * multiple
    return torch.tensor([s + [value] * (n - len(s)) for s in seqs])


def seq2seq_tensors(batch):
    """인코더-디코더용: src = <bos> 원문 <eos>, tgt = <bos> 번역문 <eos> (원본 collate_batch와 같은 모양)."""
    src = pad_to([[BOS, *s, EOS] for s, _ in batch])
    tgt = pad_to([[BOS, *t, EOS] for _, t in batch])
    return src, tgt


def lm_tensors(batch):
    """디코더 전용용: <bos> 원문 <sep> 번역문 <eos>를 한 줄로 잇고, 번역문 쪽만 손실에 넣는다.

    labels[i]는 i번째 위치가 맞혀야 할 다음 토큰이다. 원문 구간은 -100(무시)이라
    모델이 원문을 외우는 데 용량을 쓰지 않고 "<sep> 뒤를 잇는 법"만 배운다.
    """
    seqs, labels = [], []
    for s, t in batch:
        prompt = [BOS, *s, SEP]
        seqs.append(prompt + t)
        labels.append([-100] * (len(prompt) - 1) + t + [EOS])
    return pad_to(seqs), pad_to(labels, value=-100)
