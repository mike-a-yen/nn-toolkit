from typing import List


_PAD = "<pad>"
_UNK = "<unk>"
_BOS = "<bos>"
_EOS = "<eos>"
_SPECIALS = [_PAD, _UNK, _BOS, _EOS]


class TokenStore:
    def __init__(self, tokens: List[str] = []) -> None:
        self.token_to_int = dict()
        self.int_to_token = list()
        self.add_tokens(_SPECIALS)
        self.add_tokens(tokens)

    def add_tokens(self, tokens: List[str]) -> None:
        """Add a list of tokens."""
        [self.add_token(token) for token in tokens]

    def add_token(self, token: str) -> None:
        """Add a token to the vocab."""
        if token in self.token_to_int:
            return
        self.int_to_token.append(token)
        idx = self.int_to_token.index(token)
        self.token_to_int[token] = idx
    
    def __getitem__(self, token: str) -> int:
        return self.token_to_int.get(token, self.unk_idx)

    @property
    def pad(self) -> str:
        return _PAD

    @property
    def unk(self) -> str:
        return _UNK

    @property
    def bos(self) -> str:
        return _BOS
    
    @property
    def eos(self) -> str:
        return _EOS

    @property
    def pad_idx(self) -> int:
        return self.token_to_int[self.pad]

    @property
    def unk_idx(self) -> int:
        return self.token_to_int[self.unk]

    @property
    def bos_idx(self) -> int:
        return self.token_to_int[self.bos]
    
    @property
    def eos_idx(self) -> int:
        return self.token_to_int[self.eos]

    @property
    def size(self):
        return len(self.int_to_token)
    