from rtmdf.model.mlp import NeuralNetworkV2
from rtmdf.model.spec import BaseModelSpec


class ModelSpecV04(BaseModelSpec):

    def __init__(self):
        super().__init__()

        # Config inputs, if they are different from default.
        self._cols_x = self._cols_x

        # Config targets, if they are different from default.
        self._cols_y = self._responders

        # PyTorch model.
        self._model = NeuralNetworkV2()
