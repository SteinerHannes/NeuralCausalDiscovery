import sys
import torch.nn as nn


class Tee(object):
    def __init__(self, *args, proc_cr=False, **kwargs):
        self.file = open(*args, **kwargs)
        self.stdout = sys.stdout
        self.proc_cr = proc_cr
        #self.dbg_print = lambda msg: print(msg, file=self.stdout)
        self.dbg_print = lambda msg: None
        self.nl_pos = 0
        sys.stdout = self

    def __del__(self):
        self.dbg_print(f'del {self.file.name}')
        self.free()

    def free(self):
        if self.file.closed:
            self.dbg_print(f'  !!no action since resources already freed')
            return
        sys.stdout = self.stdout
        self.file.close()

    def __enter__(self):
        self.dbg_print(f'enter {self.file.name}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dbg_print(f'exit {self.file.name}')
        self.free()

    def write(self, data):
        self.stdout.write(data)
        if self.proc_cr:
            # split lines such that last line never ends with a linesep
            lines = (data + 'X').splitlines(keepends=True)
            lines[-1] = lines[-1][:-1]
            # write full lines
            for line in lines[:-1]:
                self.file.write(line)
                # convert CR
                if line.endswith('\r'):
                    self.file.seek(self.nl_pos)
                else:
                    self.nl_pos = self.file.tell()
            # write incomplete (last) line
            self.file.write(lines[-1])
        else:
            self.file.write(data)

    def flush(self):
        # print('flush', file=self.stdout)
        self.file.flush()


_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
}


def get_activation(name):
    if isinstance(name, str):
        key = name.lower()
        if key not in _ACTIVATION_MAP:
            raise ValueError(f"Unknown activation: {name}")
        return _ACTIVATION_MAP[key]
    if isinstance(name, type) and issubclass(name, nn.Module):
        return name
    raise ValueError(f"Invalid activation spec: {name}")
