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
