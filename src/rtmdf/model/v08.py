import polars as pl
import torch

from rtmdf.cal.transform import append_mean_features
from rtmdf.model.mlp import NeuralNetworkV4
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV08(BaseModelSpec):
    _mean_features: list[str] = [f"mean_feature_{i:02d}" for i in range(79)]

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x + self._mean_features

        # Config targets, if they are different from default.
        self._cols_y = self._responders

        # PyTorch model.
        self._model = NeuralNetworkV4(in_size=82 + 79, out_size=9, hidden=400, num_layers=20, dropout=0.5)

        # Scale final predictions (to avoid expensive mistakes).
        self._scale_pred = 1.0

        # Scale predictions in test.
        self.test_scales = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1, 1.25, 1.5]

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
        named_losses = {f"loss_rsq_{scale:.4f}": self._rsq_loss(y_pred * scale, y, w) for scale in self.test_scales}
        return loss_rsq, named_losses

    def log_loss_train(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> str:
        """Return formatted training loss to be printed."""
        log_str = f"R^2: {1 - cum_named_losses['loss_rsq']:>+5f} ({cum_named_losses['loss_rsq']:>5f})"
        return log_str

    def log_loss_test(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> None:
        """Print test loss."""
        cum_loss /= sum_batch_sizes
        print(f"Test R^2 score: {1 - cum_loss:>+5f}")
        for scale in self.test_scales:
            cum_named_losses[f"loss_rsq_{scale:.4f}"] /= sum_batch_sizes
            print(f'Test R^2 scaled by {scale:.4f}x: {1 - cum_named_losses[f"loss_rsq_{scale:.4f}"]:>+5f}')

    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)

        pred = pl.DataFrame(y_pred.detach().cpu().numpy(), schema=[f"responder_{i:01d}" for i in range(9)])
        pred = pred.select(pl.col("responder_6").alias("predict"))
        return pred  # Always a single column.

    def transform_source(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Transform data source before splitting into inputs and targets."""
        df = append_mean_features(df)
        return df
