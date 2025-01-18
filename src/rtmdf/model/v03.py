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

    def log_loss_train(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> str:
        """Return formatted training loss to be printed."""
        log_str = (
            f"R^2: {1 - cum_named_losses['loss_rsq']:>+5f} ({cum_named_losses['loss_rsq']:>5f}) "
            f"MSE Loss: {cum_named_losses['loss_mse']:>7f}"
        )
        return log_str

    def log_loss_test(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> None:
        """Print test loss."""
        cum_loss /= sum_batch_sizes
        print(f"Test R^2 score: {1 - cum_loss:>+5f}")

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

    def transform_source(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Transform data source before splitting into inputs and targets."""
        return df
