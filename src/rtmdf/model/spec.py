from abc import ABC, abstractmethod

import polars as pl
import torch
from torch import nn

from rtmdf.cal.loss import r_square_loss


class BaseModelSpec(ABC):
    """Specifies everything a model should contain, from config to data transforms."""

    _cols_x: list[str]
    _cols_y: list[str]
    _cols_w: list[str]
    _index: list[str] = ["date_id", "time_id", "symbol_id"]
    _features: list[str] = [f"feature_{i:02d}" for i in range(79)]
    _responders: list[str] = [f"responder_{i:01d}" for i in range(9)]
    _model: nn.Module
    _mae_loss = nn.L1Loss()
    _mse_loss = nn.MSELoss()
    # _rsq_loss = r_square_loss

    def __init__(self):
        # Default columns.
        self._cols_x = ["time_id", "symbol_id", "weight"] + self.features
        self._cols_y = ["responder_6"]
        self._cols_w = ["weight"]

    @staticmethod
    def _rsq_loss(y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Compute modified R^2 loss as defined in competition."""
        return r_square_loss(y_pred, y_true, weight)

    @property
    def index(self) -> list[str]:
        """Data source's index: "date_id", "time_id", "symbol_id"."""
        return self._index

    @property
    def features(self) -> list[str]:
        """Data source's features."""
        return self._features

    @property
    def responders(self) -> list[str]:
        """Data source's responders."""
        return self._responders

    @property
    def cols_x(self):
        """Columns in data source to be used as training input."""
        return self._cols_x

    @property
    def cols_y(self):
        """Columns in data source to be used as training target."""
        return self._cols_y

    @property
    def cols_w(self):
        """Columns in data source to be used as symbol weight."""
        return self._cols_w

    @property
    def model(self) -> nn.Module:
        """PyTorch neural network for prediction."""
        return self._model

    @property
    def device(self) -> str:
        """Get PyTorch device."""
        return self._device

    @device.setter
    def device(self, device: str):
        """Set PyTorch device."""
        self._model = self._model.to(device)
        self._device = device

    def log_model_params(self, model: nn.Module | None = None) -> None:
        """Print model parameter count."""
        if model is None:
            model = self._model
        model_total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {model_total_params:,} parameters, amongst which {trainable_params:,} are trainable.")

    @abstractmethod
    def eval_loss_train(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for training."""

    @abstractmethod
    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""

    @abstractmethod
    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""

    # Optional.
    def predict_custom(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" and other user-defined time series given input."""
        raise NotImplementedError("Custom prediction not implemented.")
