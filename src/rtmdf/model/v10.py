import polars as pl
import torch
from torch import nn

from rtmdf.model.mlp import NeuralNetworkV6
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV10(BaseModelSpec):
    _mean_features: list[str] = [f"mean_feature_{i:02d}" for i in range(79)]

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x + self._mean_features

        # Config targets, if they are different from default.
        self._cols_y = self._responders

        # PyTorch model.
        self._model = NeuralNetworkV6(in_size=82 + 79, out_size=9, hidden=300, num_layers=20, dropout=0.5)

        # Cross-Entropy loss.
        self._xen_loss = None  # We'll define it later because we don't know which `device` we're on.

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

    def predict(self, X: torch.Tensor) -> pl.DataFrame:
        """Predict "responder_6" given input."""
