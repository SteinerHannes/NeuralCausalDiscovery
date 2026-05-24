import torch
import torch.nn as nn
from models.Extractable import Extractable
from models.utils import get_activation


class ShallowMLP(nn.Module, Extractable):
    def __init__(
        self,
        input_size,
        hidden1_size,
        hidden2_size,
        output_size,
        activation="relu",
        model_id=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        act_cls = get_activation(activation)
        self.model_id = model_id
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)
        self.act1 = act_cls()
        self.act2 = act_cls()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x
