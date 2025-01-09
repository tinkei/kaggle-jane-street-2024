from torch import nn


class NeuralNetworkV1(nn.Module):
    def __init__(self, in_size: int = 82, out_size: int = 1, dropout: float = 0.25):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(in_size),
            nn.Dropout(dropout),
            nn.Linear(in_size, 384),
            nn.GELU(),
            nn.BatchNorm1d(384),
            nn.Dropout(dropout),
            nn.Linear(384, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 384),
            nn.GELU(),
            nn.BatchNorm1d(384),
            nn.Linear(384, out_size),
        )

    def forward(self, x):
        responder_6 = self.linear_relu_stack(x)
        return responder_6


class NeuralNetworkV2(nn.Module):
    def __init__(self, in_size: int = 82, out_size: int = 9, dropout: float = 0.5):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.BatchNorm1d(in_size),
            nn.Dropout(dropout),
            nn.Linear(82, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, out_size),
        )

    def forward(self, x):
        all_responders = self.linear_relu_stack(x)
        return all_responders


class BlockV3(nn.Module):
    def __init__(self, in_neurons: int, out_neurons: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_neurons, out_neurons),
            nn.BatchNorm1d(out_neurons),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class NeuralNetworkV3(nn.Module):
    def __init__(
        self, in_size: int = 82, out_size: int = 9, hidden: int = 512, num_layers: int = 20, dropout: float = 0.5
    ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.BatchNorm1d(in_size),
            nn.Dropout(dropout),
            nn.Linear(in_size, hidden),
            nn.GELU(),
        )

        self.num_layers = num_layers
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(BlockV3(hidden, hidden, dropout))
        self.layers = nn.ModuleList(self.layers)

        self.output_block = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_size),
        )

    def forward(self, x):
        x = self.input_block(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = self.output_block(x)
        return x


class NeuralNetworkV4(nn.Module):
    def __init__(
        self, in_size: int = 82, out_size: int = 9, hidden: int = 512, num_layers: int = 20, dropout: float = 0.5
    ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
            nn.GELU(),
        )

        self.num_layers = num_layers
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(BlockV3(hidden, hidden, dropout))
        self.layers = nn.ModuleList(self.layers)

        self.output_block = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_size),
        )

    def forward(self, x):
        x = self.input_block(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        x = self.output_block(x)
        return x


class NeuralNetworkV5(nn.Module):
    def __init__(
        self,
        in_size: int = 82,
        out_size: int = 9,
        hidden: int = 512,
        num_layers: int = 20,
        dropout: float = 0.5,
        num_class: int = 4,
    ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.BatchNorm1d(hidden),
            # nn.Dropout(dropout),  # Disabled, to make sure the model can learn from the means.
            nn.GELU(),
        )

        self.num_layers = num_layers
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(BlockV3(hidden, hidden, dropout))
        self.layers = nn.ModuleList(self.layers)

        # Extra linear layer hoping that it'd help bypass weight decay to increase output prediction magnitude.
        self.output_block1 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden),
            nn.Linear(hidden, out_size),
        )

        self.out_size = out_size
        self.num_class = num_class
        self.output_block2 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, num_class * out_size),
            # nn.Softmax(dim=1),  # No need, because `nn.CrossEntropyLoss()` has LogSoftmax included.
        )

    def forward(self, x):
        x = self.input_block(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        y_regress = self.output_block1(x)
        y_prob = self.output_block2(x).view(-1, self.num_class, self.out_size)
        return y_regress, y_prob


class NeuralNetworkV6(NeuralNetworkV5):
    def __init__(
        self,
        in_size: int = 82,
        out_size: int = 9,
        hidden: int = 512,
        num_layers: int = 20,
        dropout: float = 0.5,
        num_class: int = 4,
    ):
        super().__init__(in_size, out_size, hidden, num_layers, dropout, num_class)

        # Removed extra linear layer.
        self.output_block1 = nn.Sequential(
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, out_size),
        )


class NeuralNetworkV7(nn.Module):
    def __init__(
        self,
        in_size: int = 82,
        out_sma: int = 2,
        out_regress: int = 6,
        out_class: int = 6,
        hidden: int = 512,
        num_layers: int = 10,
        dropout: float = 0.5,
        num_class: int = 4,
        steps_predict: int = 120,
    ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.BatchNorm1d(hidden),
            # nn.Dropout(dropout),  # Disabled, to make sure the model can learn from the means.
            nn.GELU(),
            BlockV3(hidden, hidden, dropout),
        )

        self.num_layers = num_layers
        self.layers = []
        for _ in range(self.num_layers):
            self.layers.append(BlockV3(hidden, hidden, dropout))
        self.layers = nn.ModuleList(self.layers)

        # Output group 1: Predict each time step and calculate the SMA later.
        self.out_sma = out_sma
        self.steps_predict = steps_predict
        self.output_block1 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            BlockV3(hidden // 2, hidden // 2, dropout),
            BlockV3(hidden // 2, hidden // 2, dropout),
            BlockV3(hidden // 2, hidden // 2, dropout),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, steps_predict * out_sma),
        )

        # Output group 2: Predict directly the SMA responders.
        self.out_regress = out_regress
        self.steps_predict = steps_predict
        self.output_block2 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            BlockV3(hidden // 2, hidden // 2, dropout),
            BlockV3(hidden // 2, hidden // 2, dropout),
            BlockV3(hidden // 2, hidden // 2, dropout),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, out_regress),
        )

        # Output group 3: Predict zero-crossings and avoid them.
        self.out_class = out_class
        self.num_class = num_class
        self.output_block3 = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            BlockV3(hidden // 2, hidden // 2, dropout),
            BlockV3(hidden // 2, hidden // 2, dropout),
            BlockV3(hidden // 2, hidden // 2, dropout),
            nn.BatchNorm1d(hidden // 2),
            nn.Linear(hidden // 2, num_class * out_class),
            # nn.Softmax(dim=1),  # No need, because `nn.CrossEntropyLoss()` has LogSoftmax included.
        )

    def forward(self, x):
        x = self.input_block(x)
        for i in range(self.num_layers):
            x = self.layers[i](x)
        y_step = self.output_block1(x).view(-1, self.steps_predict, self.out_sma)
        y_regress = self.output_block2(x)
        y_prob = self.output_block3(x).view(-1, self.num_class, self.out_class)
        return y_step, y_regress, y_prob
