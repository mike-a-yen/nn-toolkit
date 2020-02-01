from collections import Counter
import operator
from pathlib import Path
import pickle
from typing import List, Union

import pandas as pd

from .token_store import TokenStore


class Vocab:
    def __init__(self, token_store: TokenStore) -> None:
        self.token_store = token_store

    def map_to_ints(self, tokens: List[str], add_specials: bool = True) -> List[int]:
        encoded = [self.token_store[token] for token in tokens]
        if add_specials:
            encoded = [self.token_store.bos_idx] + encoded + [self.token_store.eos_idx]
        return encoded

    def map_to_tokens(self, tokens: List[int], strip_pad: bool = True) -> List[str]:
        if strip_pad:
            decoded = [self.token_store.int_to_token[i] for i in tokens if i != self.token_store.pad_idx]
        else:
            decoded = [self.token_store.int_to_token[i] for i in tokens]
        return decoded
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_store
    
    def __getitem__(self, token: str) -> int:
        return self.token_store[token]

    def __getattr__(self, k: str):
        return getattr(self.token_store, k)
    
    @classmethod
    def from_file(cls, path: Path):
        with open(path, 'rb') as fo:
            token_store = pickle.load(fo)
        return cls(token_store)
    
    def to_file(self, path: Path) -> None:
        self.token_store.save(path)


class VocabBuilder:
    def __init__(self, max_size: int = None, min_count: int = None, max_count: int = None) -> None:
        self.max_size = max_size
        self.min_count = min_count
        self.max_count = max_count
        self.token_counter = Counter()
    
    def update_counter(self, tokens: List[str]) -> None:
        self.token_counter.update(tokens)
    
    def get_vocab_tokens(self) -> List[str]:
        tokens = []
        for token, count in sorted(self.token_counter.items(), key=lambda x: -x[1]):
            if self.passes_count_threshold(len(tokens), self.max_size, operator.ge):
                break
            if self.passes_count_threshold(count, self.min_count, operator.ge):
                #if self.passes_count_threshold(count, self.max_count, operator.le):
                tokens.append(token)
        return tokens

    def from_df(self, df: pd.DataFrame, token_cols: Union[List[str], str]) -> Vocab:
        if isinstance(token_cols, str):
            token_cols = [token_cols]
        tokens = []
        for col in token_cols:
            _ = df[col].apply(self.update_counter)
        vocab_tokens = self.get_vocab_tokens()
        token_store = TokenStore(vocab_tokens)
        return Vocab(token_store)

    def passes_count_threshold(self, count: int, threshold: int, op: operator) -> bool:
        if threshold is None:
            return True
        return op(count, threshold)
