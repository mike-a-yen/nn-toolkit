import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    """Token embedding layer with access to a positional embedding."""

    def __init__(self, vocab_size: int, hidden_size: int, maxlen: int = None, padding_idx: int = None, batch_first: bool = True) -> None:
        super().__init__()
        self.token_layer = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        self.pos_layer = None
        self.maxlen = maxlen
        self.batch_first = batch_first
        if maxlen is not None:
            self.pos_layer = PositionalEmbedding(maxlen, hidden_size)
            self.scale = 1. / math.sqrt(self.embedding_dim)
  
    def forward(self, X: torch.Tensor) -> torch.FloatTensor:
        token_emb = self.token_layer(X)
        if self.pos_layer is not None:
            time_dim = int(self.batch_first)  # if batch first, time dim is 1
            pos_emb = self.scale * self.pos_layer(X, time_dim)
            token_emb += pos_emb
        return token_emb

    def __getattr__(self, k: str):
        try:
            return super().__getattr__(k)
        except AttributeError:
            return getattr(self.token_layer, k)


class PositionalEmbedding(nn.Module):
    def __init__(self, maxlen: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            self._init_weight(maxlen, hidden_size),
            requires_grad=False
        )

    @torch.no_grad()
    def forward(self, X: torch.Tensor, dim: int = 1):
        """Get positional embedding from an arbitrary tensor.

        Parameters
        ----------
        X : torch.Tensor
            Expects a rank 2 tensor
        dim : int
            the dimension of `X` to condsider as the time dimension, by default 1
        """
        assert X.dim() == 2
        shape = list(X.size())
        T = shape.pop(dim)
        B = shape[0]
        pos_idx = torch.arange(0, T).to(self.weight.device).unsqueeze(0)
        emb = F.embedding(pos_idx, self.weight)  # (1, T, e)
        emb = emb.repeat(B, 1, 1)  # (B, T, e)
        if dim == 0:
            return torch.transpose(emb, 0, 1)  # (T, B, e)
        return emb

    def _init_weight(self, maxlen: int, hidden_size: int) -> torch.FloatTensor:
        weight = torch.empty(maxlen, hidden_size)
        evens = torch.arange(0, hidden_size, 2)
        odds = torch.arange(1, hidden_size, 2)
        even_freq = 10000**(evens.float() / hidden_size)
        odd_freq = 10000**(odds.float() / hidden_size)
        pos = torch.arange(0, maxlen).unsqueeze(1)
        weight[:, evens] = torch.sin(pos.float() / even_freq)
        weight[:, odds] = torch.cos(pos.float() / odd_freq)
        return weight
