"""원본이 torchtext에서 가져다 쓰던 세 가지를 같은 이름·같은 동작으로 대신한다.

torchtext는 0.18(2024년 4월)을 끝으로 개발이 중단됐다. 0.18은 torch 2.3에 맞춰 빌드된 네이티브
라이브러리(libtorchtext.so)를 들고 있어서, 설치는 되어도 최신 torch에서는 import하는 순간
`OSError: Could not load this library`로 죽는다.

원본 노트북은 torchtext의 아래 API만 쓴다.
- torchtext.datasets.Multi30k(language_pair=("de", "en")) → (train, valid, test) 반복자
- torchtext.vocab.build_vocab_from_iterator(..., min_freq, specials) → Vocab
- torchtext.data.functional.to_map_style_dataset(iter) → len()이 있는 Dataset

그래서 노트북 쪽은 import 세 줄만 이 모듈로 바꾸고 나머지 코드는 한 글자도 건드리지 않는다.
"""

from collections import Counter, OrderedDict
from types import SimpleNamespace


# ── torchtext.datasets.Multi30k ──────────────────────────────────────────────


def _multi30k(root=".data", split=("train", "valid", "test"), language_pair=("de", "en")):
    """torchtext의 Multi30k와 같은 29,000 / 1,014 / 1,000 문장쌍(test는 test_2016_flickr).

    torchtext가 받아 오던 원본 tar 대신 같은 데이터를 올려 둔 HF Hub의 bentrevett/multi30k를 쓴다.
    원본 코드가 `train + val + test`로 이어 붙이므로 리스트로 돌려준다.
    """
    from datasets import disable_progress_bars, load_dataset

    disable_progress_bars()
    ds = load_dataset("bentrevett/multi30k", cache_dir=root)
    names = {"train": "train", "valid": "validation", "test": "test"}
    src, tgt = language_pair
    out = [list(zip(ds[names[s]][src], ds[names[s]][tgt])) for s in ((split,) if isinstance(split, str) else split)]
    return out[0] if isinstance(split, str) else tuple(out)


datasets = SimpleNamespace(Multi30k=_multi30k)


# ── torchtext.vocab ──────────────────────────────────────────────────────────


class Vocab:
    """torchtext.vocab.Vocab 중 원본이 쓰는 부분: vocab(tokens), vocab[token], len,
    get_stoi(), get_itos(), set_default_index()."""

    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.default_index = None

    def __len__(self):
        return len(self.itos)

    def __contains__(self, token):
        return token in self.stoi

    def __getitem__(self, token):
        if token in self.stoi:
            return self.stoi[token]
        if self.default_index is None:
            raise RuntimeError(f"Token {token} not found and default index is not set")
        return self.default_index

    def __call__(self, tokens):
        return [self[t] for t in tokens]

    def set_default_index(self, index):
        self.default_index = index

    def get_default_index(self):
        return self.default_index

    def get_stoi(self):
        return dict(self.stoi)

    def get_itos(self):
        return list(self.itos)

    def lookup_tokens(self, indices):
        return [self.itos[i] for i in indices]


def build_vocab_from_iterator(iterator, min_freq=1, specials=None, special_first=True, max_tokens=None):
    """torchtext 0.12와 같은 순서로 사전을 만든다: specials 먼저, 나머지는 빈도 내림차순(동률이면 사전순)."""
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)
    specials = specials or []
    for s in specials:
        counter.pop(s, None)
    ordered = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    if max_tokens is not None:
        ordered = ordered[: max_tokens - len(specials)]
    words = [w for w, c in OrderedDict(ordered).items() if c >= min_freq]
    return Vocab(specials + words if special_first else words + specials)


# ── torchtext.data.functional.to_map_style_dataset ───────────────────────────


def to_map_style_dataset(iter_data):
    """반복자를 len()과 인덱싱이 되는 Dataset으로. DistributedSampler가 len()을 요구한다."""
    return list(iter_data)
