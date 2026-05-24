import torch.nn as nn
from models.Extractable import Extractable
from models.utils import get_activation


class MLPModel(nn.Module, Extractable):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=(64, 64),
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

        act_cls = get_activation(activation)
        self.model_id = model_id
        layers = []
        prev = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev, hidden_size, bias=bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_size

        layers.append(nn.Linear(prev, output_size, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
