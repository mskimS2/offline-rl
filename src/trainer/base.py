import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Any, Dict


class OfflineRLTrainer(ABC):
    REQUIRED_NETWORK_KEYS = {"q1", "q2", "policy"}
    REQUIRED_OPTIMIZER_KEYS = {"alpha", "policy", "q1", "q2"}

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")

    @abstractmethod
    def initialize_networks(self) -> torch.nn.Module:
        """Initialize the model for the algorithm."""
        pass

    @abstractmethod
    def initialize_optimizers(self) -> torch.optim.Optimizer:
        """Initialize the optimizer for the algorithm."""
        pass

    @abstractmethod
    def initialize_logger(self):
        """Initialize the logger for the algorithm."""
        pass

    @abstractmethod
    def initialize_schedulers(self) -> Any:
        """Initialize the scheduler for learning rate adjustment."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        pass

    @abstractmethod
    def train(self) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self) -> float:
        """Evaluate the model."""
        pass

    @abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Load the model from the specified path."""
        pass
