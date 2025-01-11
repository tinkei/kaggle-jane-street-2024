import polars as pl
import torch
from torch import nn

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

        # Scale predictions smaller for training, then scale back during evaluation.
        self._scale_y = 100

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
        loss_sma004_r3 = self._mse_loss(pred_sma[:, :, 0], y[:, 14:19]) * self._scale_y
        loss_sma004_r6 = self._mse_loss(pred_sma[:, :, 1], y[:, 19:24]) * self._scale_y
        loss_sma020_r3 = self._mse_loss(pred_sma[:, :, 0].mean(dim=1), y[:, 3]) * self._scale_y
        loss_sma020_r6 = self._mse_loss(pred_sma[:, :, 1].mean(dim=1), y[:, 6]) * self._scale_y
        loss_sma004_x = self._mse_loss(pred_sma[:, :, 0] - pred_sma[:, :, 1], y[:, 9:14]) * self._scale_y
        loss_sma020_x = self._mse_loss((pred_sma[:, :, 0] - pred_sma[:, :, 1]).mean(dim=1), y[:, 0]) * self._scale_y
        loss_sma_rsq = self._rsq_loss(pred_sma.mean(dim=1), y[:, [3, 6]], w) / 2
        loss += (
            loss_sma004_r3
            + loss_sma004_r6
            + loss_sma020_r3
            + loss_sma020_r6
            + loss_sma004_x
            + loss_sma020_x
            + loss_sma_rsq
        ) / 7

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
        loss_ll_rsq /= 58
        loss += (loss_ll_mse004 + loss_ll_mse020 + loss_ll_rsq) / 3

        # Debug sanity check.
        assert all(["responder_3" in col for col in self.cols_y[[3]]])
        assert all(["responder_6" in col for col in self.cols_y[[6]]])
        assert all(["responder_2" in col for col in self.cols_y[9:14]])
        assert all(["responder_5" in col for col in self.cols_y[14:19]])
        assert all(["responder_8" in col for col in self.cols_y[19:24]])
        assert all(["responder_2" in col for col in self.cols_y[24:53]])
        assert all(["responder_5" in col for col in self.cols_y[53:82]])
        assert all(["responder_8" in col for col in self.cols_y[82:111]])
        assert all(["responder_0" in col for col in self.cols_y[111:132]])
        assert all(["responder_3" in col for col in self.cols_y[132:153]])
        assert all(["responder_6" in col for col in self.cols_y[153:174]])

        return loss, {
            "loss_rsq": loss_rsq,
            "loss_mse": loss_mse,
            "loss_xen": loss_xen,
            "loss_sma004_r3": loss_sma004_r3,
            "loss_sma004_r6": loss_sma004_r6,
            "loss_sma020_r3": loss_sma020_r3,
            "loss_sma020_r6": loss_sma020_r6,
            "loss_sma004_x": loss_sma004_x,
            "loss_sma020_x": loss_sma020_x,
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
        pred_regress = torch.where(pred_class == 0, pred_regress * 0.01, pred_regress)
        pred_regress = torch.where(pred_class == 1, pred_regress * 0.10, pred_regress)
        pred_regress = torch.where(pred_class == 2, pred_regress * 0.25, pred_regress)
        loss_rsq = self._rsq_loss(pred_regress[:, [6]], y[:, [6]], w)  # Consider only Responder 6 in test loss.

        pred_sma = pred_sma.mean(dim=1) * self._scale_y
        pred_sma = torch.where(pred_class[:, [0, 3]] == 0, pred_sma * 0.01, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 1, pred_sma * 0.10, pred_sma)
        pred_sma = torch.where(pred_class[:, [0, 3]] == 2, pred_sma * 0.25, pred_sma)
        loss_sma_rsq = self._rsq_loss(pred_sma[:, [1]], y[:, [6]], w)  # Consider only Responder 6 in test loss.

        pred_lead_lag20 = pred_lead_lag20[:, 10, [1, 2]] * self._scale_y
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 0, pred_lead_lag20 * 0.01, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 1, pred_lead_lag20 * 0.10, pred_lead_lag20)
        pred_lead_lag20 = torch.where(pred_class[:, [0, 3]] == 2, pred_lead_lag20 * 0.25, pred_lead_lag20)
        loss_ll = self._rsq_loss(pred_lead_lag20[:, [1]], y[:, [6]], w)

        loss = loss_rsq + loss_sma_rsq + loss_ll
        return loss, {
            "loss_rsq": loss_rsq,
            "loss_sma_rsq": loss_sma_rsq,
            "loss_ll": loss_ll,
        }

    def predict(self, X: torch.Tensor) -> pl.DataFrame:
        """Predict "responder_6" given input."""
