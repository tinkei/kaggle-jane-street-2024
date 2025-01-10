from abc import ABC, abstractmethod

from torch import nn


class BaseModelSpec(ABC):
    """Specifies everything a model should contain, from config to data transforms."""

    _cols_x: list[str]
    _cols_y: list[str]
    _cols_w: list[str]
    _index: list[str] = ["date_id", "time_id", "symbol_id"]
    _features: list[str] = [f"feature_{i:02d}" for i in range(79)]
    _responders: list[str] = [f"responder_{i:01d}" for i in range(9)]
    _model: nn.Module

    def __init__(self):
        # Default columns.
        self._cols_x = ["time_id", "symbol_id", "weight"] + self.features
        self._cols_y = ["responder_6"]
        self._cols_w = ["weight"]

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

    # @cols_w.setter
    # def cols_w(self, value: list[str]):
    #     self._cols_w = value

    @property
    def model(self) -> nn.Module:
        """PyTorch neural network for prediction."""
        return self._model

    # @property
    # @abstractmethod
    # def test(self):
    #     pass
