import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_toolkit.text.model.embedding import Embedding
from nn_toolkit.text.model.sequence import TextSequenceEncoder


class LanguageModel(nn.Module):
    def __init__(
            self, 
            max_vocab_size: int, 
            hidden_size: int = 256, 
            num_layers: int = 3, 
            dropout_rate: float = 0.2, 
            maxlen: int = None,
            padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.maxlen = maxlen
        self.padding_idx = padding_idx

        self.embedding_layer = Embedding(self.max_vocab_size, self.hidden_size, self.maxlen, self.padding_idx)
        self.encoder = TextSequenceEncoder(self.hidden_size, self.num_layers, self.dropout_rate)
        self.decoder = SimpleDecoder(self.max_vocab_size, self.hidden_size)
        self.decoder.tie_weights(self.embedding_layer.weight)
    
    def forward(self, X: torch.LongTensor) -> torch.FloatTensor:
        emb = self.embedding_layer(X)
        mask = self.get_mask(X)
        emb = self.encoder(emb, mask)
        logit = self.decoder(emb)
        log_prob = F.log_softmax(logit, dim=-1)
        return log_prob.view(-1, self.max_vocab_size)
    
    def get_mask(self, X: torch.LongTensor) -> torch.ByteTensor:
        pad_val = self.embedding_layer.padding_idx
        if pad_val is None:
            return None
        return X == pad_val


class SimpleDecoder(nn.Module):
    def __init__(self, max_vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.max_vocab_size = max_vocab_size
        self.hidden_size = hidden_size
        self.densor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.max_vocab_size)
        )

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        return self.densor(X)

    def tie_weights(self, embedding_weights: torch.FloatTensor) -> None:
        self.densor[-1].weight = embedding_weights
