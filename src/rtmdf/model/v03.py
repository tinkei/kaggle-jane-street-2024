import polars as pl
import torch

from rtmdf.model.mlp import NeuralNetworkV1
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV03(BaseModelSpec):

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x

        # Config targets, if they are different from default.
        self._cols_y = self._responders + self._features

        # PyTorch model.
        self._model = NeuralNetworkV1(in_size=82, out_size=9 + 79, dropout=0.25)

    def eval_loss_train(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for training."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        loss_rsq = self._rsq_loss(y_pred[:, :9], y[:, :9], w)
        loss_mse = self._mse_loss(y_pred, y)
        loss = loss_rsq + loss_mse * 0.0001
        return loss, {
            "loss_rsq": loss_rsq,
            "loss_mse": loss_mse,
        }

    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        loss_rsq = self._rsq_loss(y_pred[:, :9], y[:, :9], w)
        return loss_rsq, {
            "loss_rsq": loss_rsq,
        }

    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)

        pred = pl.DataFrame(
            y_pred.detach().cpu().numpy(),
            schema=[f"responder_{i:01d}" for i in range(9)] + [f"feature_{i:02d}" for i in range(79)],
        )
        pred = pred.select(pl.col("responder_6").alias("predict"))
        return pred  # Always a single column.
