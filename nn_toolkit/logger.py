from pathlib import Path
import os

import torch.nn as nn
import wandb
from wandb.fastai import WandbCallback


class Logger:
    def __init__(self, project_name: str, sync: bool = False, **kwargs) -> None:
        if not sync:
            os.environ['WANDB_MODE'] = 'dryrun'
        wandb.init(project=project_name, **kwargs)
        self.dirname = Path(wandb.run.dir)

    def log_model(self, model: nn.Module) -> None:
        wandb.watch(model)

    def log_config(self, config: dict) -> None:
        wandb.config.update(config, allow_val_change=True)

    def log(self, params: dict) -> None:
        wandb.log(params)

    def log_file(self, filename) -> None:
        filepath = self.dirname / f'{filename}'
        wandb.save(str(filepath))
    
    def log_plot(self, plot, name: str) -> None:
        wandb.log({f'{name}', plot})

    def callback_fn(self) -> WandbCallback:
        return WandbCallback
