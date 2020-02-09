import json
from pathlib import Path

import torch
import torch.nn as nn

from nn_toolkit.text.model.dense import ResidualBlock


class TextSequenceEncoder(nn.Module):
    def __init__(
            self,
            hidden_size: int = 256, 
            num_layers: int = 3, 
            dropout_rate: float = 0.2, 
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.rnns = self._init_rnns()
        self.skips = self._init_skips()
    
    def forward(self, emb: torch.FloatTensor, mask: torch.ByteTensor = None) -> torch.FloatTensor:
        for skip, rnn in zip(self.skips, self.rnns):
            emb = self.apply_mask(emb, mask)
            new_emb, _ = rnn(emb)
            emb = skip(emb, new_emb)
        return emb

    def apply_mask(self, emb: torch.FloatTensor, mask: torch.ByteTensor = None) -> torch.FloatTensor:
        if mask is None:
            return emb
        return emb * (1 - mask.float().unsqueeze(-1))

    def _init_rnns(self) -> None:
        rnn_kwargs = {
            'input_size': self.hidden_size,
            'hidden_size': self.hidden_size,
            'num_layers': 1,
            'batch_first': True,
            'bidirectional': False
        }
        rnns = [nn.LSTM(**rnn_kwargs) for _ in range(self.num_layers)]
        return nn.ModuleList(rnns)

    def _init_skips(self) -> None:
        skips = [
            ResidualBlock(self.hidden_size, self.dropout_rate) for _ in range(self.num_layers)
        ]
        return nn.ModuleList(skips)
    
    def get_config(self) -> dict:
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
        }
    
    def save(self, path: Path) -> None:
        config = self.get_config()
        with open(path.with_suffix('.config'), 'w') as fo:
            json.dump(config, fo)
        torch.save(self.state_dict(), path)

    @classmethod
    def from_file(cls, path: Path) -> None:
        config_file = path.with_suffix('.config')
        with open(config_file) as fo:
            config = json.load(fo)
        model = cls(**config)
        model.load_state_dict(
            torch.load(path)
        )
        return model
