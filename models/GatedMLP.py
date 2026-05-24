import torch
import torch.nn as nn

from models.Extractable import Extractable
from models.utils import get_activation


class GatedLayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        act_cls = get_activation(activation)
        self.value = nn.Linear(input_size, output_size, bias=bias)
        self.gate = nn.Linear(input_size, output_size, bias=bias)
        self.act = act_cls()
        self.bn = nn.BatchNorm1d(output_size) if use_batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        v = self.value(x)
        if self.bn is not None:
            v = self.bn(v)
        v = self.act(v)
        g = torch.sigmoid(self.gate(x))
        out = v * g
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class GatedMLP(nn.Module, Extractable):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=(64, 64, 64, 64),
        activation="relu",
        dropout=0.0,
        use_batch_norm=False,
        bias=True,
        model_id=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if hidden_sizes is None:
            hidden_sizes = []
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.model_id = model_id
        layers = []
        prev = input_size
        for hidden_size in hidden_sizes:
            layers.append(
                GatedLayer(
                    prev,
                    hidden_size,
                    activation=activation,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    bias=bias,
                )
            )
            prev = hidden_size

        self.layers = nn.ModuleList(layers)
        self.output_proj = nn.Linear(prev, output_size, bias=bias)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.output_proj(out)
