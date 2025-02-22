import polars as pl
import torch
from torch import nn

from rtmdf.cal.transform import append_mean_features
from rtmdf.model.mlp import NeuralNetworkV5
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV09(BaseModelSpec):
    _mean_features: list[str] = [f"mean_feature_{i:02d}" for i in range(79)]

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x + self._mean_features

        # Config targets, if they are different from default.
        self._cols_y = self._responders

        # PyTorch model.
        self._model = NeuralNetworkV5(in_size=82 + 79, out_size=9, hidden=100, num_layers=40, dropout=0.5)

        # Cross-Entropy loss.
        # https://discuss.pytorch.org/t/passing-the-weights-to-crossentropyloss-correctly/14731/24
        self._xen_loss = None  # We will define it later because we don't know yet which `device` we are on.

    @property
    def version(self) -> int:
        """Model version."""
        return 9

    def eval_loss_train(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for training."""
        if self._xen_loss is None:
            self._xen_loss = nn.CrossEntropyLoss(
                weight=torch.Tensor([0.55, 0.22, 0.14, 0.09]).to(self.device), label_smoothing=0.05
            )
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        pred_regress, pred_prob = y_pred
        # print("pred_regress", pred_regress.size())  # (batch, num_responder)
        # print("pred_prob", pred_prob.size())        # (batch, num_class, num_responder)

        # We noticed that the R^2 loss will blow up when y_true is small (denominator).
        #  Therefore, we train a classification model to determine if y_true is likely to be small in magnitude.
        #  Smaller than 0.06, or 10% percentile, or R^2 loss < -1.10 => Set prediction to 0.
        #  Smaller than 0.15, or 25% percentile, or R^2 loss < -0.24 => Multiply prediction by a quarter.
        #  Smaller than 0.26, or 40% percentile, or R^2 loss <  0.00 => Multiply prediction by half.
        #  We then reduce the "bet size" roughly inspired by Kelly Criterion:
        #  p_win / loss - p_loss / profit = (p_win x profit - p_loss x loss) / (profit x loss)
        y_abs = y.abs()
        y_class = torch.zeros_like(y_abs, dtype=torch.long)
        y_class = torch.where(y_abs > 0.06, 1, y_class)
        y_class = torch.where(y_abs > 0.15, 2, y_class)
        y_class = torch.where(y_abs > 0.26, 3, y_class)
        loss_xen = self._xen_loss(pred_prob, y_class)
        loss_mse = self._mse_loss(pred_regress, y)
        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.10, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)
        loss_rsq = self._rsq_loss(pred_regress, y, w)
        loss = loss_rsq + loss_mse / 2 + loss_xen / 4
        return loss, {
            "loss_rsq": loss_rsq,
            "loss_mse": loss_mse,
            "loss_xen": loss_xen,
        }

    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        pred_regress, pred_prob = y_pred
        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        # print("pred_regress", pred_regress.size())  # (batch, num_responder)
        # print("pred_prob   ", pred_prob.size())     # (batch, num_class, num_responder)
        # print("pred_class  ", pred_class.size())    # (batch, num_responder)
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)
        loss_rsq = self._rsq_loss(pred_regress, y, w)
        return loss_rsq, {
            "loss_rsq": loss_rsq,
        }

    def log_loss_train(self, cum_loss: float, cum_named_losses: dict[str, float], sum_batch_sizes: int = 1) -> str:
        """Return formatted training loss to be printed."""
        log_str = (
            f"R^2: {1 - cum_named_losses['loss_rsq']:>+5f} ({cum_named_losses['loss_rsq']:>5f}) "
            f"MSE Loss: {cum_named_losses['loss_mse']:>7f} "
            f"Cross Entropy: {cum_named_losses['loss_xen']:>7f}"
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
        pred_regress, pred_prob = y_pred

        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)

        pred = pl.DataFrame(pred_regress.detach().cpu().numpy() / 2.0, schema=[f"responder_{i:01d}" for i in range(9)])
        pred = pred.select(pl.col("responder_6").alias("predict"))
        return pred

    def transform_source(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Transform data source before splitting into inputs and targets."""
        df = append_mean_features(df)
        return df
