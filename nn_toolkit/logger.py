import wandb
from wandb.fastai import WandbCallback


class Logger:
    def __init__(self, project_name: str, **kwargs) -> None:
        wandb.init(project=project_name, **kwargs)
        self.dirname = wandb.run.dir

    def log_model(self, model: nn.Module) -> None:
        wandb.watch(model)
    
    def log_config(self, config: dict) -> None:
        wandb.config.update(**config)
    
    def log(self, params: dict) -> None:
        wandb.log(params)
    
    def callback_fn(self) -> WandbCallback:
        return WandbCallback
