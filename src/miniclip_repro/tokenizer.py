from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class VocabularySpec:
    min_freq: int = 1
    max_size: int | None = None


class CaptionVocabulary:
    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        for token in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
            if token not in self.token_to_id:
                raise ValueError(f"Missing required token: {token}")

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.eos_token]

    def __len__(self) -> int:
        return len(self.token_to_id)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    @classmethod
    def build(cls, captions: list[str], spec: VocabularySpec) -> "CaptionVocabulary":
        counter: Counter[str] = Counter()
        for caption in captions:
            counter.update(cls.tokenize(caption))

        specials = [cls.pad_token, cls.unk_token, cls.bos_token, cls.eos_token]
        token_to_id = {token: idx for idx, token in enumerate(specials)}

        candidates = [
            (token, freq)
            for token, freq in counter.items()
            if freq >= spec.min_freq and token not in token_to_id
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        limit = None if spec.max_size is None else max(spec.max_size - len(specials), 0)
        for token, _freq in candidates[:limit]:
            token_to_id[token] = len(token_to_id)

        return cls(token_to_id)

    def encode(self, text: str, max_length: int) -> list[int]:
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS tokens.")
        token_ids = [self.token_to_id.get(token, self.unk_id) for token in self.tokenize(text)]
        token_ids = token_ids[: max_length - 2]
        encoded = [self.bos_id, *token_ids, self.eos_id]
        encoded.extend([self.pad_id] * (max_length - len(encoded)))
        return encoded

    def to_dict(self) -> dict[str, object]:
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CaptionVocabulary":
        raw_mapping = payload["token_to_id"]
        if not isinstance(raw_mapping, dict):
            raise TypeError("Vocabulary payload must contain a token_to_id mapping.")
        return cls({str(token): int(idx) for token, idx in raw_mapping.items()})

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> "CaptionVocabulary":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

