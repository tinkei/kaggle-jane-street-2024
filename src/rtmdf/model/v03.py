from rtmdf.model.mlp import NeuralNetworkV1
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV03(BaseModelSpec):

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x

        # Config targets, if they are different from default.
        self._cols_y = self._responders + self._features

        # PyTorch model.
        self._model = NeuralNetworkV1(in_size=82, out_size=9 + 79, dropout=0.25)
