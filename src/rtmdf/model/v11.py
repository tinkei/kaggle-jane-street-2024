from rtmdf.constant import SMA_RESPONDER_MAP
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV11(BaseModelSpec):
    _mean_features: list[str] = [f"mean_feature_{i:02d}" for i in range(79)]

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
        self._cols_x = self._cols_x + self._mean_features + responder_x_cols

        # Config targets, if they are different from default.
        responder_y_cols = []
        for col in SMA_RESPONDER_MAP[4]:
            responder_y_cols.extend([f"{col}_lead_{i}" for i in range(4, 20, 4)])  # 4 new targets.
        self._cols_y = self._responders + responder_y_cols
