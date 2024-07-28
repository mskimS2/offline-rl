import torch
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from typing import Any, Dict


class OfflineRLTrainer(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")

    @abstractmethod
    def initialize_model(self) -> torch.nn.Module:
        """Initialize the model for the algorithm."""
        pass

    @abstractmethod
    def initialize_optimizer(self) -> torch.optim.Optimizer:
        """Initialize the optimizer for the algorithm."""
        pass

    @abstractmethod
    def initialize_scheduler(self) -> Any:
        """Initialize the scheduler for learning rate adjustment."""
        pass

    @abstractmethod
    def initialize_loss_fn(self) -> Any:
        """Initialize the loss function."""
        pass

    @abstractmethod
    def load_data(self) -> None:
        """Load the offline data into the replay buffer."""
        pass

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        pass

    def train(self) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        pass

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the model to the specified path."""
        pass

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load the model from the specified path."""
        pass
