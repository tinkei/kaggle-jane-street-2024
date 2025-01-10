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
