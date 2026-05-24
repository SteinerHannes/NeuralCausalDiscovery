from models.MLPModel import MLPModel


class BottleneckMLP(MLPModel):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_sizes=(64, 16, 64),
        activation="relu",
        dropout=0.0,
        use_batch_norm=False,
        bias=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
            bias=bias,
            *args,
            **kwargs,
        )
