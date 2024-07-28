import uuid
import pandas as pd
from pathlib import Path
from loggers.base import Logger

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger(Logger):
    def __init__(self) -> None:
        if wandb is None:
            raise ImportError("WandbLogger requires wandb. Install using `pip install wandb`")
        super().__init__()
        self.run = None

    def init_experiment(self, exp_name_log, full_name=None, setup=True, **kwargs):
        self.run = wandb.init(project=exp_name_log, name=full_name, **kwargs) if not wandb.run else wandb.run
        return self.run

    def log_params(self, params, model_name=None):
        if model_name:
            params = {model_name: params}
        self.run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics, source=None):
        if source:
            prefixed_metrics = {}
            for metric in metrics:
                prefixed_metrics[source + "/" + metric] = metrics[metric]
            metrics = prefixed_metrics
        self.run.log(metrics)

    def log_artifact(self, file, type=None):
        file_name, extension = None, ""
        file_pathlib = Path(file)
        file_name = file_pathlib.stem.replace(" ", "_") + str(uuid.uuid1())[:8]
        extension = file_pathlib.suffix
        art = wandb.Artifact(name=file_name, type=type or "exp_data")
        art.add_file(file)
        self.run.log_artifact(art)

        if extension == "html":
            self.run.log({file_name: wandb.Html(file)})
        elif extension == "csv":
            self.run.log({file_name: pd.read_csv(file)})

    def finish_experiment(self):
        if self.run:
            self.run.finish()
