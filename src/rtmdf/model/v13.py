import numpy as np
import polars as pl
import torch
from torch import nn

from rtmdf.cal.transform import append_lagged_features, append_mean_features
from rtmdf.constant import SMA_RESPONDER_MAP
from rtmdf.model.mlp import NeuralNetworkV7
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV13(BaseModelSpec):
    _mean_features: list[str] = [f"mean_feature_{i:02d}" for i in range(79)]
    _lag_features: list[str] = [f"feature_{i:02d}_lag_1" for i in range(79)]

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x + self._mean_features + self._lag_features

        # Config targets, if they are different from default.
        responder_y_cols = []
        for col in SMA_RESPONDER_MAP[4]:
            responder_y_cols.extend([f"{col}_lead_{i}" for i in range(4, 20, 4)])  # 4 new targets.
        self._cols_y = self._responders + responder_y_cols

        # PyTorch model.
        self._model = NeuralNetworkV7(
            in_size=82 + 79 * 2,
            out_sma=2,
            out_regress=9,
            out_class=6,
            hidden=300,
            num_layers=12,
            dropout=0.5,
            steps_predict=5,
        )

        # Cross-Entropy loss.
        self._xen_loss = None  # We will define it later because we don't know yet which `device` we are on.

        # Scale predictions smaller for training, then scale back during evaluation.
        self._scale_y = 10

        # Scale final predictions (to avoid expensive mistakes).
        self._scale_pred = 0.1

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
        pred_sma, pred_regress, pred_prob = y_pred
        # self.cols_y == [
        #     'responder_0', 'responder_1', 'responder_2', 'responder_3', 'responder_4',
        #     'responder_5', 'responder_6', 'responder_7', 'responder_8',
        #     'responder_2_lead_4', 'responder_2_lead_8', 'responder_2_lead_12', 'responder_2_lead_16',
        #     'responder_5_lead_4', 'responder_5_lead_8', 'responder_5_lead_12', 'responder_5_lead_16',
        #     'responder_8_lead_4', 'responder_8_lead_8', 'responder_8_lead_12', 'responder_8_lead_16',
        # ]
        # len(self.cols_y) 21
        # y.size() torch.Size([37752, 21])

        y_abs = y[:, 3:9].abs()
        y = y / self._scale_y

        # Loss from directly regressing responders and classifying magnitudes of responders.
        y_class = torch.zeros_like(y_abs, dtype=torch.long)
        y_class = torch.where(y_abs > 0.06, 1, y_class)
        y_class = torch.where(y_abs > 0.15, 2, y_class)
        y_class = torch.where(y_abs > 0.26, 3, y_class)
        loss_xen = self._xen_loss(pred_prob, y_class)
        loss_mse = self._mse_loss(pred_regress, y[:, :9]) * self._scale_y
        loss_rsq = self._rsq_loss(pred_regress[:, 3:9], y[:, 3:9], w)
        loss = loss_rsq + loss_mse / 2 + loss_xen / 4

        # Loss from reconstructing SMA.
        # print("pred_sma.size()", pred_sma.size())  # (batch, 5, 2)
        loss_sma004_r5 = self._mse_loss(pred_sma[:, :, 0], y[:, [5, 13, 14, 15, 16]]) * self._scale_y
        loss_sma004_r8 = self._mse_loss(pred_sma[:, :, 1], y[:, [8, 17, 18, 19, 20]]) * self._scale_y
        loss_sma020_r3 = self._mse_loss(pred_sma[:, :, 0].mean(dim=1), y[:, 3]) * self._scale_y
        loss_sma020_r6 = self._mse_loss(pred_sma[:, :, 1].mean(dim=1), y[:, 6]) * self._scale_y
        loss_sma004_r2 = self._mse_loss(pred_sma[:, :, 0] - pred_sma[:, :, 1], y[:, [2, 9, 10, 11, 12]]) * self._scale_y
        loss_sma020_r0 = self._mse_loss((pred_sma[:, :, 0] - pred_sma[:, :, 1]).mean(dim=1), y[:, 0]) * self._scale_y
        loss_sma_rsq = self._rsq_loss(pred_sma.mean(dim=1), y[:, [3, 6]], w)
        loss_sma_rsq += self._rsq_loss(pred_sma[:, :, 0], y[:, [5, 13, 14, 15, 16]], w)
        loss_sma_rsq += self._rsq_loss(pred_sma[:, :, 1], y[:, [8, 17, 18, 19, 20]], w)
        loss_sma_rsq /= 3  # 2 + 5 + 5. Adjust here s.t. outliers don't impact solution too much.
        loss += (
            (
                loss_sma004_r5
                + loss_sma004_r8
                + loss_sma020_r3
                + loss_sma020_r6
                + loss_sma004_r2
                + loss_sma020_r0
                + loss_sma_rsq
            )
            / 7
            / 2
        )

        # Debug sanity check.
        debug_cols_y = np.array(self.cols_y)
        assert all(["responder_3" in col for col in debug_cols_y[[3]]])
        assert all(["responder_6" in col for col in debug_cols_y[[6]]])
        assert all(["responder_2" in col for col in debug_cols_y[[2, 9, 10, 11, 12]]])
        assert all(["responder_5" in col for col in debug_cols_y[[5, 13, 14, 15, 16]]])
        assert all(["responder_8" in col for col in debug_cols_y[[8, 17, 18, 19, 20]]])

        return loss, {
            "loss_rsq": loss_rsq,
            "loss_mse": loss_mse,
            "loss_xen": loss_xen,
            "loss_sma004_r5": loss_sma004_r5,
            "loss_sma004_r8": loss_sma004_r8,
            "loss_sma020_r3": loss_sma020_r3,
            "loss_sma020_r6": loss_sma020_r6,
            "loss_sma004_r2": loss_sma004_r2,
            "loss_sma020_r0": loss_sma020_r0,
            "loss_sma_rsq": loss_sma_rsq,
        }

    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        pred_sma, pred_regress, pred_prob = y_pred
        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        # print("pred_regress", pred_regress.size())  # (batch, num_responder)
        # print("pred_prob   ", pred_prob.size())     # (batch, num_class, num_responder)
        # print("pred_class  ", pred_class.size())    # (batch, num_responder)

        pred_regress = pred_regress[:, 3:9] * self._scale_y
        loss_raw_rsq = self._rsq_loss(pred_regress[:, [3]], y[:, [6]], w)
        loss_kelly_rsq = self._rsq_loss(pred_regress[:, [3]] * 0.1, y[:, [6]], w)
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)
        loss_zerox_rsq = self._rsq_loss(pred_regress[:, [3]], y[:, [6]], w)  # Consider only Responder 6 in test loss.

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        loss_sma_raw_rsq = self._rsq_loss(pred_sma[:, [1]], y[:, [6]], w)
        loss_sma_kelly_rsq = self._rsq_loss(pred_sma[:, [1]] * 0.1, y[:, [6]], w)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.00, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.25, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.50, pred_sma)
        loss_sma_zerox_rsq = self._rsq_loss(pred_sma[:, [1]], y[:, [6]], w)  # Consider only Responder 6 in test loss.

        loss = (
            loss_raw_rsq + loss_kelly_rsq + loss_zerox_rsq + loss_sma_raw_rsq + loss_sma_kelly_rsq + loss_sma_zerox_rsq
        ) / 6
        return loss, {
            "loss_raw_rsq": loss_raw_rsq,
            "loss_kelly_rsq": loss_kelly_rsq,
            "loss_zerox_rsq": loss_zerox_rsq,
            "loss_sma_raw_rsq": loss_sma_raw_rsq,
            "loss_sma_kelly_rsq": loss_sma_kelly_rsq,
            "loss_sma_zerox_rsq": loss_sma_zerox_rsq,
        }

    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)
        pred_sma, pred_regress, pred_prob = y_pred

        # pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        pred_regress = pred_regress[:, 3:9] * self._scale_y  # Select only responders 3 - 8 from here on out.
        # pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        # pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        # pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        # pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.00, pred_sma)
        # pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.25, pred_sma)
        # pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.50, pred_sma)

        weight_sma = 0.4
        pred = (
            pred_regress[:, [0, 3]].detach().cpu().numpy() * (1 - weight_sma)
            + pred_sma.detach().cpu().numpy() * weight_sma
        ) * self._scale_pred
        pred = pl.DataFrame(pred, schema=[f"responder_{i:01d}" for i in [3, 6]])
        pred = pred.select(pl.col("responder_6").alias("predict"))
        return pred  # Always a single column.

    def predict_custom(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" and other user-defined time series given input."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)
        pred_sma, pred_regress, pred_prob = y_pred

        # Return unfiltered predictions to user.
        pred_regress_raw = torch.clone(pred_regress) * self._scale_y
        pred_sma_raw = torch.clone(pred_sma.mean(dim=1)) * self._scale_y

        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        pred_regress = pred_regress[:, 3:9] * self._scale_y  # Select only responders 3 - 8 from here on out.
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.00, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.25, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.50, pred_sma)

        weight_sma = 0.75
        # Include raw predictions.
        pred = pred_regress_raw.detach().cpu().numpy()  # Contains predictions for all 9 responders.
        pred[:, [3, 6]] = pred[:, [3, 6]] * (1 - weight_sma) + pred_sma_raw.detach().cpu().numpy() * weight_sma
        pred = pl.DataFrame(pred, schema=[f"responder_{i:01d}" for i in range(9)])
        # Artificially create columns for responders 0 - 2.
        pred = pred.with_columns(
            pl.col(f"responder_{i + 3:01d}")
            .truediv(self._scale_y)
            .add(1.0)
            .log()
            .sub(pl.col(f"responder_{i + 6:01d}").truediv(self._scale_y).add(1.0).log())
            .exp()
            .sub(1.0)
            .mul(self._scale_y)
            .alias(f"responder_{i:01d}")
            for i in range(0, 3)
        )
        pred = pred[[f"responder_{i:01d}" for i in range(9)]]
        # Include filtered predictions,
        pred_censored = (
            pred_regress[:, [0, 3]].detach().cpu().numpy() * (1 - weight_sma)
            + pred_sma.detach().cpu().numpy() * weight_sma
        ) * self._scale_pred
        pred_censored = pl.DataFrame(pred_censored, schema=[f"responder_{i:01d}_censored" for i in [3, 6]])
        pred = pred.with_columns(
            responder_3_censored=pred_censored["responder_3_censored"],
            responder_6_censored=pred_censored["responder_6_censored"],
        )
        return pred

    def transform_source(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Transform data source before splitting into inputs and targets."""
        df = append_mean_features(df)
        df = append_lagged_features(df, 1)
        return df
