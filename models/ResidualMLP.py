import torch.nn as nn
from models.Extractable import Extractable
from models.utils import get_activation


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, activation="relu", dropout=0.0, bias=True):
        super().__init__()
        act_cls = get_activation(activation)
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.act1 = act_cls()
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.act2 = act_cls()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        out = self.act1(self.fc1(x))
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.fc2(out)
        out = out + x
        out = self.act2(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class ResidualMLP(nn.Module, Extractable):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=64,
        num_blocks=2,
        activation="relu",
        dropout=0.0,
        bias=True,
        model_id=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        act_cls = get_activation(activation)
        self.model_id = model_id
        self.input_proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.input_act = act_cls()
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_size, activation=activation, dropout=dropout, bias=bias) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x):
        out = self.input_act(self.input_proj(x))
        for block in self.blocks:
            out = block(out)
        return self.output_proj(out)
