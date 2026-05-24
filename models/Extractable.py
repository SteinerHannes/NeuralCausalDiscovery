import sys
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class Extractable:
    _DEFAULT_ACTIVATION_TYPES = (
        nn.ReLU,
        nn.LeakyReLU,
        nn.Sigmoid,
        nn.Tanh,
        nn.GELU,
        nn.ELU,
        nn.SiLU,
        nn.Softplus,
    )
    _DEFAULT_PREACT_TYPES = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    )

    def extract(self, dataloader: torch.utils.data.DataLoader, method="activations", **kwargs) -> pd.DataFrame:
        if method == "activations":
            return self.extract_activations(dataloader, **kwargs)
        if method == "pre_activations":
            return self.extract_pre_activations(dataloader, **kwargs)
        if method == "inputs_outputs":
            return self.extract_inputs_outputs(dataloader, **kwargs)
        if method == "input_gradients":
            return self.extract_input_gradients(dataloader, **kwargs)
        if method == "weights":
            return self.extract_weights(dataloader=dataloader, **kwargs)
        raise ValueError(f"Unknown extraction method: {method}")

    def extract_inputs_outputs(
        self,
        dataloader: torch.utils.data.DataLoader,
        include_inputs: bool = True,
        include_outputs: bool = True,
        flatten: bool = True,
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        return self._extract_inputs_outputs(
            dataloader,
            include_inputs,
            include_outputs,
            flatten,
            progress=progress,
            progress_desc=progress_desc,
        )

    def extract_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_types=None,
        include_inputs: bool = True,
        include_outputs: bool = True,
        flatten: bool = True,
        prefix: str = "act",
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        layer_types = self._resolve_layer_types(layer_types, self._DEFAULT_ACTIVATION_TYPES)
        return self._extract_with_hooks(
            dataloader=dataloader,
            layer_types=layer_types,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
            flatten=flatten,
            prefix=prefix,
            progress=progress,
            progress_desc=progress_desc,
        )

    def extract_pre_activations(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_types=None,
        include_inputs: bool = True,
        include_outputs: bool = True,
        flatten: bool = True,
        prefix: str = "pre",
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        layer_types = self._resolve_layer_types(layer_types, self._DEFAULT_PREACT_TYPES)
        return self._extract_with_hooks(
            dataloader=dataloader,
            layer_types=layer_types,
            include_inputs=include_inputs,
            include_outputs=include_outputs,
            flatten=flatten,
            prefix=prefix,
            progress=progress,
            progress_desc=progress_desc,
        )

    def extract_input_gradients(
        self,
        dataloader: torch.utils.data.DataLoader,
        include_inputs: bool = True,
        include_outputs: bool = True,
        flatten: bool = True,
        grad_target: str = "sum",
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        batch_arrays = []
        columns = None
        device = next(self.parameters()).device
        target = self._parse_grad_target(grad_target)
        if target not in ("sum", "mean") and not isinstance(target, int):
            raise ValueError(f"Invalid grad_target: {grad_target}")

        loader = self._iter_with_progress(dataloader, progress, progress_desc)
        for x, _ in loader:
            x = x.float().to(device)
            x = x.detach().requires_grad_(True)
            self.zero_grad(set_to_none=True)

            out = self(x)
            out_flat = out if not flatten else out.reshape(out.shape[0], -1)
            if target == "sum":
                loss = out_flat.sum()
            elif target == "mean":
                loss = out_flat.mean()
            else:
                loss = out_flat[:, target].sum()

            grads = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

            x_flat = x.detach().cpu()
            if flatten:
                x_flat = x_flat.reshape(x_flat.shape[0], -1)
            out_cpu = out.detach().cpu()
            if flatten:
                out_cpu = out_cpu.reshape(out_cpu.shape[0], -1)
            grads_cpu = grads.detach().cpu()
            if flatten:
                grads_cpu = grads_cpu.reshape(grads_cpu.shape[0], -1)

            parts = []
            if include_inputs:
                parts.append(x_flat)
            parts.append(grads_cpu)
            if include_outputs:
                parts.append(out_cpu)
            if not parts:
                continue
            batch_arrays.append(torch.cat(parts, dim=1).numpy())

            if columns is None:
                local_columns = []
                if include_inputs:
                    local_columns.extend([f"input_{j}" for j in range(x_flat.shape[1])])
                local_columns.extend([f"input_grad_{j}" for j in range(grads_cpu.shape[1])])
                if include_outputs:
                    local_columns.extend([f"output_{j}" for j in range(out_cpu.shape[1])])
                columns = local_columns

        if not batch_arrays:
            return pd.DataFrame(columns=columns or [])
        return pd.DataFrame(np.concatenate(batch_arrays, axis=0), columns=columns)

    def extract_weights(
        self,
        dataloader: torch.utils.data.DataLoader = None,
        repeat_for_samples: bool = False,
        include_bias: bool = True,
        prefix: str = "param",
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        record = {}
        for name, param in self.named_parameters():
            if not include_bias and "bias" in name:
                continue
            flat = param.detach().cpu().reshape(-1)
            safe_name = self._sanitize_name(name)
            for idx in range(flat.shape[0]):
                record[f"{prefix}_{safe_name}_{idx}"] = float(flat[idx].item())

        if repeat_for_samples and dataloader is not None:
            num_rows = len(dataloader.dataset)
            return pd.DataFrame([record.copy() for _ in range(num_rows)])
        return pd.DataFrame([record])

    def _extract_inputs_outputs(
        self,
        dataloader,
        include_inputs,
        include_outputs,
        flatten,
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ):
        batch_arrays = []
        columns = None
        device = next(self.parameters()).device
        loader = self._iter_with_progress(dataloader, progress, progress_desc)
        with torch.inference_mode():
            for x, _ in loader:
                x = x.float().to(device)
                out = self(x)

                x_cpu = x.detach().cpu()
                out_cpu = out.detach().cpu()
                if flatten:
                    x_cpu = x_cpu.reshape(x_cpu.shape[0], -1)
                    out_cpu = out_cpu.reshape(out_cpu.shape[0], -1)

                parts = []
                if include_inputs:
                    parts.append(x_cpu)
                if include_outputs:
                    parts.append(out_cpu)
                if not parts:
                    batch_arrays.append(np.empty((x_cpu.shape[0], 0), dtype=np.float32))
                    if columns is None:
                        columns = []
                    continue
                batch_arrays.append(torch.cat(parts, dim=1).numpy())

                if columns is None:
                    local_columns = []
                    if include_inputs:
                        local_columns.extend([f"input_{j}" for j in range(x_cpu.shape[1])])
                    if include_outputs:
                        local_columns.extend([f"output_{j}" for j in range(out_cpu.shape[1])])
                    columns = local_columns
        if not batch_arrays:
            return pd.DataFrame(columns=columns or [])
        return pd.DataFrame(np.concatenate(batch_arrays, axis=0), columns=columns)

    def _extract_with_hooks(
        self,
        dataloader: torch.utils.data.DataLoader,
        layer_types,
        include_inputs: bool,
        include_outputs: bool,
        flatten: bool,
        prefix: str,
        progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> pd.DataFrame:
        layers = [
            (name, module)
            for name, module in self.named_modules()
            if name and isinstance(module, layer_types)
        ]
        activations = {}
        hooks = []

        def _make_hook(layer_name):
            def _hook(module, inputs, output):
                activations[layer_name] = output.detach()
            return _hook

        for name, module in layers:
            hooks.append(module.register_forward_hook(_make_hook(name)))

        batch_arrays = []
        columns = None
        device = next(self.parameters()).device

        loader = self._iter_with_progress(dataloader, progress, progress_desc)
        with torch.inference_mode():
            for x, _ in loader:
                x = x.float().to(device)
                activations.clear()
                out = self(x)

                x_cpu = x.detach().cpu()
                out_cpu = out.detach().cpu()
                if flatten:
                    x_cpu = x_cpu.reshape(x_cpu.shape[0], -1)
                    out_cpu = out_cpu.reshape(out_cpu.shape[0], -1)

                layer_values = []
                layer_column_blocks = []
                for layer_idx, (name, _) in enumerate(layers):
                    act = activations.get(name)
                    if act is None:
                        layer_values.append(None)
                        layer_column_blocks.append([])
                        continue
                    act_cpu = act.detach().cpu()
                    if flatten:
                        act_cpu = act_cpu.reshape(act_cpu.shape[0], -1)
                    layer_values.append(act_cpu)
                    safe_name = self._sanitize_name(name)
                    layer_column_blocks.append(
                        [f"{prefix}_{layer_idx + 1}_{safe_name}_{k}" for k in range(act_cpu.shape[1])]
                    )

                parts = []
                if include_inputs:
                    parts.append(x_cpu)
                parts.extend([act_cpu for act_cpu in layer_values if act_cpu is not None])
                if include_outputs:
                    parts.append(out_cpu)
                if not parts:
                    batch_arrays.append(np.empty((x_cpu.shape[0], 0), dtype=np.float32))
                    if columns is None:
                        columns = []
                    continue
                batch_arrays.append(torch.cat(parts, dim=1).numpy())

                if columns is None:
                    local_columns = []
                    if include_inputs:
                        local_columns.extend([f"input_{j}" for j in range(x_cpu.shape[1])])
                    for block in layer_column_blocks:
                        local_columns.extend(block)
                    if include_outputs:
                        local_columns.extend([f"output_{j}" for j in range(out_cpu.shape[1])])
                    columns = local_columns

        for hook in hooks:
            hook.remove()

        if not batch_arrays:
            return pd.DataFrame(columns=columns or [])
        return pd.DataFrame(np.concatenate(batch_arrays, axis=0), columns=columns)

    def _sanitize_name(self, name: str) -> str:
        return name.replace(".", "_")

    def _iter_with_progress(self, dataloader, progress: bool, progress_desc: Optional[str]):
        if not progress:
            return dataloader
        desc = progress_desc or "extract"
        return tqdm(dataloader, desc=desc, unit="bat", file=sys.stdout, ascii=True)

    def _resolve_layer_types(self, layer_types, default_types):
        if layer_types is None:
            return default_types
        if isinstance(layer_types, str):
            layer_types = [layer_types]
        resolved = []
        for entry in layer_types:
            if isinstance(entry, str):
                if not hasattr(nn, entry):
                    raise ValueError(f"Unknown layer type: {entry}")
                resolved.append(getattr(nn, entry))
            elif isinstance(entry, type):
                resolved.append(entry)
            else:
                raise ValueError(f"Invalid layer type entry: {entry}")
        return tuple(resolved)

    def _parse_grad_target(self, grad_target):
        if isinstance(grad_target, str):
            if grad_target.isdigit():
                return int(grad_target)
            return grad_target
        return grad_target
