import polars as pl
import torch

from rtmdf.model.mlp import NeuralNetworkV1
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV01(BaseModelSpec):

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x

        # Config targets, if they are different from default.
        self._cols_y = self._cols_y

        # PyTorch model.
        self._model = NeuralNetworkV1(in_size=82, out_size=1, dropout=0.25)

    def eval_loss_train(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for training."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        loss_rsq = self._rsq_loss(y_pred, y, w)
        return loss_rsq, {
            "loss_rsq": loss_rsq,
        }

    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        loss_rsq = self._rsq_loss(y_pred, y, w)
        return loss_rsq, {
            "loss_rsq": loss_rsq,
        }

    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)

        pred = pl.DataFrame(y_pred.detach().cpu().numpy(), schema=["predict"])
        return pred  # Always a single column.

    def transform_source(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Transform data source before splitting into inputs and targets."""
        return df
