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
            nn.BatchNorm1d(in_neurons),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class NeuralNetworkV3(nn.Module):
    def __init__(
        self, in_size: int = 82, out_size: int = 9, hidden: int = 1024, num_layers: int = 10, dropout: float = 0.5
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
