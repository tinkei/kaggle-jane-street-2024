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
