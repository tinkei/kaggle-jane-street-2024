import numpy as np
import polars as pl
import torch
from torch import nn

from rtmdf.cal.transform import append_lagged_features, append_mean_features
from rtmdf.constant import SMA_RESPONDER_MAP
from rtmdf.model.mlp import NeuralNetworkV8
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV12(BaseModelSpec):
    _mean_features: list[str] = [f"mean_feature_{i:02d}" for i in range(79)]
    _lag_features: list[str] = [f"feature_{i:02d}_lag_1" for i in range(79)]

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        responder_x_cols = []
        for col in SMA_RESPONDER_MAP[4]:
            responder_x_cols.extend([f"{col}_lag_{i}" for i in range(1, 4 + 1)])
        for col in SMA_RESPONDER_MAP[20]:
            responder_x_cols.extend([f"{col}_lag_{i}" for i in range(1, 20 + 1)])
        for col in SMA_RESPONDER_MAP[120]:
            responder_x_cols.extend([f"{col}_lag_{i}" for i in range(1, 120 + 1, 20)])
        self._cols_x = self._cols_x + self._mean_features + self._lag_features + responder_x_cols

        # Config targets, if they are different from default.
        responder_y_cols = []
        for col in SMA_RESPONDER_MAP[4]:
            responder_y_cols.extend([f"{col}_lead_{i}" for i in range(0, 20, 4)])  # 5 new targets.
        for col in SMA_RESPONDER_MAP[4]:
            responder_y_cols.extend([f"{col}_lead_v2_{i}" for i in range(-4, 25)])  # 29 new targets.
        for col in SMA_RESPONDER_MAP[20]:
            responder_y_cols.extend([f"{col}_lead_v2_{i}" for i in range(-10, 11)])  # 21 new targets.
        self._cols_y = self._responders + responder_y_cols

        # PyTorch model.
        self._model = NeuralNetworkV8(
            in_size=82 + 79 * 2 + 90,
            out_sma=2,
            out_regress=9,
            out_class=6,
            out_lead_lag04=3,
            out_lead_lag20=3,
            hidden=320,
            num_layers=12,
            dropout=0.5,
            num_class=4,
            sma_steps=5,
            num_lead_lag04=29,
            num_lead_lag20=21,
        )

        # Cross-Entropy loss.
        self._xen_loss = None  # We will define it later because we don't know yet which `device` we are on.

        # Scale predictions smaller for training, then scale back during evaluation.
        self._scale_y = 100

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
        pred_sma, pred_regress, pred_prob, pred_lead_lag04, pred_lead_lag20 = y_pred
        # print("pred_sma.size()       ", pred_sma.size())
        # print("pred_regress.size()   ", pred_regress.size())
        # print("pred_prob.size()      ", pred_prob.size())
        # print("pred_lead_lag04.size()", pred_lead_lag04.size())
        # print("pred_lead_lag20.size()", pred_lead_lag20.size())
        # pred_sma.size()        torch.Size([47956, 5, 2])
        # pred_regress.size()    torch.Size([47956, 9])
        # pred_prob.size()       torch.Size([47956, 4, 6])
        # pred_lead_lag04.size() torch.Size([47956, 29, 3])
        # pred_lead_lag20.size() torch.Size([47956, 21, 3])

        y_abs = y[:, 3:9].abs()
        y = y / self._scale_y

        # Loss from directly regressing responders and classifying magnitudes of responders.
        y_class = torch.zeros_like(y_abs, dtype=torch.long)
        y_class = torch.where(y_abs > 0.06, 1, y_class)
        y_class = torch.where(y_abs > 0.15, 2, y_class)
        y_class = torch.where(y_abs > 0.26, 3, y_class)
        loss_xen = self._xen_loss(pred_prob, y_class)
        loss_mse = self._mse_loss(pred_regress, y[:, :9]) * self._scale_y
        loss_rsq = self._rsq_loss(pred_regress[:, 3:9], y[:, 3:9], w) / 6
        loss = loss_rsq + loss_mse / 2 + loss_xen / 4

        # Loss from reconstructing SMA.
        loss_sma004_r5 = self._mse_loss(pred_sma[:, :, 0], y[:, 14:19]) * self._scale_y
        loss_sma004_r8 = self._mse_loss(pred_sma[:, :, 1], y[:, 19:24]) * self._scale_y
        loss_sma020_r3 = self._mse_loss(pred_sma[:, :, 0].mean(dim=1), y[:, 3]) * self._scale_y
        loss_sma020_r6 = self._mse_loss(pred_sma[:, :, 1].mean(dim=1), y[:, 6]) * self._scale_y
        loss_sma004_r2 = self._mse_loss(pred_sma[:, :, 0] - pred_sma[:, :, 1], y[:, 9:14]) * self._scale_y
        loss_sma020_r0 = self._mse_loss((pred_sma[:, :, 0] - pred_sma[:, :, 1]).mean(dim=1), y[:, 0]) * self._scale_y
        loss_sma_rsq = self._rsq_loss(pred_sma.mean(dim=1), y[:, [3, 6]], w) / 2
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

        # Loss from extra regression terms.
        loss_ll_mse004 = self._mse_loss(pred_lead_lag04[:, :, 0], y[:, 24:53])
        loss_ll_mse004 += self._mse_loss(pred_lead_lag04[:, :, 1], y[:, 53:82])
        loss_ll_mse004 += self._mse_loss(pred_lead_lag04[:, :, 2], y[:, 82:111])
        loss_ll_mse004 *= self._scale_y / 3
        loss_ll_mse020 = self._mse_loss(pred_lead_lag20[:, :, 0], y[:, 111:132])
        loss_ll_mse020 += self._mse_loss(pred_lead_lag20[:, :, 1], y[:, 132:153])
        loss_ll_mse020 += self._mse_loss(pred_lead_lag20[:, :, 2], y[:, 153:174])
        loss_ll_mse020 *= self._scale_y / 3
        loss_ll_rsq = self._rsq_loss(pred_lead_lag04[:, :, 1], y[:, 53:82], w)
        loss_ll_rsq += self._rsq_loss(pred_lead_lag04[:, :, 2], y[:, 82:111], w)
        loss_ll_rsq /= 2
        loss += (loss_ll_mse004 + loss_ll_mse020 + loss_ll_rsq) / 3

        # Debug sanity check.
        debug_cols_y = np.array(self.cols_y)
        assert all(["responder_3" in col for col in debug_cols_y[[3]]])
        assert all(["responder_6" in col for col in debug_cols_y[[6]]])
        assert all(["responder_2" in col for col in debug_cols_y[9:14]])
        assert all(["responder_5" in col for col in debug_cols_y[14:19]])
        assert all(["responder_8" in col for col in debug_cols_y[19:24]])
        assert all(["responder_2" in col for col in debug_cols_y[24:53]])
        assert all(["responder_5" in col for col in debug_cols_y[53:82]])
        assert all(["responder_8" in col for col in debug_cols_y[82:111]])
        assert all(["responder_0" in col for col in debug_cols_y[111:132]])
        assert all(["responder_3" in col for col in debug_cols_y[132:153]])
        assert all(["responder_6" in col for col in debug_cols_y[153:174]])

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
            "loss_ll_mse004": loss_ll_mse004,
            "loss_ll_mse020": loss_ll_mse020,
            "loss_ll_rsq": loss_ll_rsq,
        }

    def eval_loss_test(
        self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor, to_device: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Evaluate prediction loss for test set."""
        if to_device:
            X, y, w = X.to(self.device), y.to(self.device), w.to(self.device)
        y_pred = self._model(X)
        pred_sma, pred_regress, pred_prob, pred_lead_lag04, pred_lead_lag20 = y_pred
        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        # print("pred_regress", pred_regress.size())  # (batch, num_responder)
        # print("pred_prob   ", pred_prob.size())     # (batch, num_class, num_responder)
        # print("pred_class  ", pred_class.size())    # (batch, num_responder)

        pred_regress = pred_regress[:, 3:9] * self._scale_y
        loss_raw_rsq = self._rsq_loss(pred_regress[:, [3]], y[:, [6]], w)
        loss_kelly_rsq = self._rsq_loss(pred_regress[:, [3]] * 0.1, y[:, [6]], w)
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.01, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.10, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.25, pred_regress)
        loss_zerox_rsq = self._rsq_loss(pred_regress[:, [3]], y[:, [6]], w)  # Consider only Responder 6 in test loss.

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        loss_sma_raw_rsq = self._rsq_loss(pred_sma[:, [1]], y[:, [6]], w)
        loss_sma_kelly_rsq = self._rsq_loss(pred_sma[:, [1]] * 0.1, y[:, [6]], w)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.01, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.10, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.25, pred_sma)
        loss_sma_zerox_rsq = self._rsq_loss(pred_sma[:, [1]], y[:, [6]], w)  # Consider only Responder 6 in test loss.

        loss_ll_raw_rsq = self._rsq_loss(pred_lead_lag20[:, [1]], y[:, [6]], w)
        loss_ll_kelly_rsq = self._rsq_loss(pred_lead_lag20[:, [1]] * 0.1, y[:, [6]], w)
        pred_lead_lag20 = pred_lead_lag20[:, 10, [1, 2]] * self._scale_y
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 0, pred_lead_lag20 * 0.01, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 1, pred_lead_lag20 * 0.10, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 2, pred_lead_lag20 * 0.25, pred_lead_lag20)
        loss_ll_zerox_rsq = self._rsq_loss(pred_lead_lag20[:, [1]], y[:, [6]], w)

        loss = (
            loss_raw_rsq
            + loss_kelly_rsq
            + loss_zerox_rsq
            + loss_sma_raw_rsq
            + loss_sma_kelly_rsq
            + loss_sma_zerox_rsq
            + loss_ll_raw_rsq
            + loss_ll_kelly_rsq
            + loss_ll_zerox_rsq
        ) / 9
        return loss, {
            "loss_raw_rsq": loss_raw_rsq,
            "loss_kelly_rsq": loss_kelly_rsq,
            "loss_zerox_rsq": loss_zerox_rsq,
            "loss_sma_raw_rsq": loss_sma_raw_rsq,
            "loss_sma_kelly_rsq": loss_sma_kelly_rsq,
            "loss_sma_zerox_rsq": loss_sma_zerox_rsq,
            "loss_ll_raw_rsq": loss_ll_raw_rsq,
            "loss_ll_kelly_rsq": loss_ll_kelly_rsq,
            "loss_ll_zerox_rsq": loss_ll_zerox_rsq,
        }

    def predict(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" given input. Returns a single-columned DataFrame "predict"."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)
        pred_sma, pred_regress, pred_prob, pred_lead_lag04, pred_lead_lag20 = y_pred
        # pred_sma.size()        torch.Size([47956, 5, 2])
        # pred_regress.size()    torch.Size([47956, 9])
        # pred_prob.size()       torch.Size([47956, 4, 6])
        # pred_lead_lag04.size() torch.Size([47956, 29, 3])
        # pred_lead_lag20.size() torch.Size([47956, 21, 3])

        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        pred_regress = pred_regress[:, 3:9] * self._scale_y  # Select only responders 3 - 8 from here on out.
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.00, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.25, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.50, pred_sma)

        pred_lead_lag20 = pred_lead_lag20[:, 10, [1, 2]] * self._scale_y
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 0, pred_lead_lag20 * 0.00, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 1, pred_lead_lag20 * 0.25, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 2, pred_lead_lag20 * 0.50, pred_lead_lag20)

        weight_sma = 0.75
        weight_reg = 0.25
        weight_ll = 0.00
        pred = (
            pred_regress[:, [0, 3]].detach().cpu().numpy() * weight_reg
            + pred_sma.detach().cpu().numpy() * weight_sma
            + pred_lead_lag20.detach().cpu().numpy() * weight_ll
        ) * self._scale_pred
        pred = pl.DataFrame(pred, schema=[f"responder_{i:01d}" for i in [3, 6]])
        pred = pred.select(pl.col("responder_6").alias("predict"))
        return pred  # Always a single column.

    def predict_custom(self, X: torch.Tensor, to_device: bool = True) -> pl.DataFrame:
        """Predict "responder_6" and other user-defined time series given input."""
        if to_device:
            X = X.to(self.device)
        y_pred = self._model(X)
        pred_sma, pred_regress, pred_prob, pred_lead_lag04, pred_lead_lag20 = y_pred

        # Return unfiltered predictions to user.
        pred_regress_raw = torch.clone(pred_regress) * self._scale_y
        pred_sma_raw = torch.clone(pred_sma.mean(dim=1)) * self._scale_y
        pred_lead_lag20_raw = torch.clone(pred_lead_lag20[:, 10, [1, 2]]) * self._scale_y

        pred_class = nn.Softmax(dim=1)(pred_prob).argmax(1)
        pred_regress = pred_regress[:, 3:9] * self._scale_y  # Select only responders 3 - 8 from here on out.
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.00, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.25, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.50, pred_regress)

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.00, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.25, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.50, pred_sma)

        pred_lead_lag20 = pred_lead_lag20[:, 10, [1, 2]] * self._scale_y
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 0, pred_lead_lag20 * 0.00, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 1, pred_lead_lag20 * 0.25, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 2, pred_lead_lag20 * 0.50, pred_lead_lag20)

        weight_sma = 0.75
        weight_reg = 0.25
        weight_ll = 0.00
        # Include raw predictions.
        pred = pred_regress_raw.detach().cpu().numpy()  # Contains predictions for all 9 responders.
        pred[:, [3, 6]] = (
            pred[:, [3, 6]] * weight_reg
            + pred_sma_raw.detach().cpu().numpy() * weight_sma
            + pred_lead_lag20_raw.detach().cpu().numpy() * weight_ll
        )
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
            pred_regress[:, [0, 3]].detach().cpu().numpy() * weight_reg
            + pred_sma.detach().cpu().numpy() * weight_sma
            + pred_lead_lag20.detach().cpu().numpy() * weight_ll
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
