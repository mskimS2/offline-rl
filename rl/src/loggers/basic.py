import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from loggers.base import Logger

console = Console()


class BasicLogger(Logger):
    """
    Basic experiment logger with structured local logging and rich console output.

    Responsibilities:
        - Initialize and finalize experiment logging sessions.
        - Persist configuration and metrics locally (JSON + CSV).
        - Display structured logs in console with rich formatting.
        - Record saved artifacts for traceability.

    This logger is intentionally lightweight and does not integrate with any
    external services (e.g., W&B, MLflow). Ideal for standalone RL experiments.
    """

    def __init__(self, log_dir: Path) -> None:
        """
        Initialize logger instance and configure file-based logging.

        Args:
            log_dir (Path): Directory to store logs, configs, and metrics.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure Python's logging for file and console
        self.logger = logging.getLogger(f"basiclogger_{self.log_dir}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.addHandler(logging.StreamHandler())
        self.logger.addHandler(logging.FileHandler(self.log_dir / "train.log", encoding="utf-8"))

        # Metrics CSV writer
        self.metrics_csv_path = self.log_dir / "metrics.csv"
        self.csv_file = self.metrics_csv_path.open("a", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_header_written = False

        # Internal experiment metadata
        self.exp_name: Optional[str] = None
        self.start_time: Optional[str] = None

    def init_experiment(self, exp_name_log: str, full_name: Optional[str] = None, **kwargs) -> None:
        """
        Initialize a new experiment run.

        Displays structured metadata header and sets up logging session.
        """
        self.exp_name = full_name or exp_name_log
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Metadata panel for console
        meta_table = Table.grid(expand=True)
        meta_table.add_row(
            f"[bold cyan]Experiment[/]: {self.exp_name}",
            f"[bold cyan]Started[/]: {self.start_time}",
            f"[bold cyan]Log Dir[/]: {str(self.log_dir)}",
        )
        console.print(Panel(meta_table, title="[bold green]Experiment Initialized[/]", expand=False))
        self.logger.info(f"[INIT] Experiment '{self.exp_name}' started at {self.start_time}")

    def finish_experiment(self) -> None:
        """
        Finalize the experiment logging session.

        Closes file handles and prints structured footer.
        """
        self.csv_file.close()
        console.rule(f"[bold red]Experiment {self.exp_name} Completed", style="red")
        self.logger.info(f"[FINISH] Experiment '{self.exp_name}' completed")

    def log_params(self, params: Dict[str, Any], model_name: Optional[str] = None) -> None:
        """
        Log hyperparameters or configuration entries.

        Args:
            params (Dict[str, Any]): Parameter dictionary.
            model_name (Optional[str]): Optional section prefix (e.g., "model").
        """
        if model_name:
            params = {model_name: params}

        # Console table for readability
        table = Table(title=f"[bold yellow]Parameters ({model_name or 'default'})", show_lines=True)
        table.add_column("Key", style="bold cyan")
        table.add_column("Value", style="white")
        for k, v in params.items():
            table.add_row(str(k), json.dumps(v) if isinstance(v, (dict, list)) else str(v))
        console.print(table)

        # Append to config.json (merging if already exists)
        config_path = self.log_dir / "config.json"
        if config_path.exists():
            existing = json.loads(config_path.read_text())
            existing.update(params)
            config_path.write_text(json.dumps(existing, indent=2))
        else:
            config_path.write_text(json.dumps(params, indent=2))

        self.logger.info(f"[PARAMS] {json.dumps(params, indent=2)}")

    def log_metrics(self, metrics: Dict[str, float], source: Optional[str] = None) -> None:
        """
        Log scalar metrics.

        Args:
            metrics (Dict[str, float]): Metric name-value pairs.
            source (Optional[str]): Optional prefix (e.g., 'train', 'eval').
        """
        if source:
            metrics = {f"{source}/{k}": v for k, v in metrics.items()}

        # Console summary
        metrics_str = " | ".join(f"[cyan]{k}[/]=[bold green]{v:.4f}[/]" for k, v in metrics.items())
        console.print(f"[bold magenta]Metrics[/]: {metrics_str}")

        # File log
        self.logger.info(f"[METRICS] {metrics_str}")

        # CSV logging
        if not self.csv_header_written:
            self.csv_writer.writerow(["step"] + list(metrics.keys()))
            self.csv_header_written = True

        step = metrics.pop("step", None)
        self.csv_writer.writerow([step or datetime.now().timestamp()] + list(metrics.values()))
        self.csv_file.flush()

    def log_artifact(self, file: Path, type: Optional[str] = None) -> None:
        """
        Record information about a saved artifact.

        Args:
            file (Path): Path to the saved artifact.
            type (Optional[str]): Optional type/category (e.g., 'checkpoint', 'plot').
        """
        file = Path(file)
        self.logger.info(f"[ARTIFACT] {file.name} ({type or 'generic'})")
        console.print(f"[bold blue]Artifact[/]: {file.name} ({type or 'generic'})")
