# -*- coding:utf-8 -*-
#
# Copyright (C) 2021, Saarland University
# Copyright (C) 2021, Maximilian Köhl <koehl@cs.uni-saarland.de>

from __future__ import annotations

import typing as t

import json

try:
    import torch
except ImportError:
    raise ImportError(
        "Missing optional dependency `torch`.\n"
        "Using MombaGym's NN export function requires `torch`."
    )


class UnsupportedLayerError(Exception):
    pass


class _FakeTensor:
    parent: t.Optional[_FakeTensor]
    module: t.Optional[torch.nn.Module]

    def __init__(
        self,
        parent: t.Optional[_FakeTensor] = None,
        module: t.Optional[torch.nn.Module] = None,
    ):
        self.parent = parent
        self.module = module

    def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore
        if func is torch.functional.F.linear:
            (_, weight) = args
            bias = kwargs.get("bias", None)
            out_features, in_features = weight.shape
            module = torch.nn.Linear(in_features, out_features, bias=bias is not None)
            module.weight = weight
            if bias is not None:
                module.bias = bias
            return _FakeTensor(self, module)
        elif func is torch.functional.F.relu:
            return _FakeTensor(self, torch.nn.ReLU())
        elif func is torch.functional.F.celu:
            return _FakeTensor(self, torch.nn.CELU(alpha=kwargs.get("alpha", 1.0)))
        else:
            raise UnsupportedLayerError(
                f"unsupported function {func!r} applied to layer"
            )


def _dump_layer(name: str, layer: torch.nn.Module) -> t.Any:
    result: t.Dict[str, t.Any] = {"name": name, "kind": layer.__class__.__name__}
    if isinstance(layer, torch.nn.Linear):
        result["inputSize"] = layer.in_features
        result["outputSize"] = layer.out_features
        result["hasBiases"] = getattr(layer, "bias", None) is not None
        result["weights"] = layer.weight.tolist()
        result["biases"] = layer.bias.tolist()
    elif isinstance(layer, torch.nn.ReLU):
        pass
    elif isinstance(layer, torch.nn.CELU):
        result["alpha"] = layer.alpha
    else:
        raise UnsupportedLayerError(f"layer of type {type(layer)} not supported")
    return result


def _dump_layers(net: torch.nn.Sequential) -> t.Any:
    return [_dump_layer(name, layer) for name, layer in net.named_children()]


def dump_nn(net: torch.nn.Module) -> str:
    if not isinstance(net, torch.nn.Sequential):
        result = net.forward(_FakeTensor())
        assert isinstance(
            result, _FakeTensor
        ), "the result of forwarding should be a `_FakeTensor`"
        layers = []
        while result.module is not None:
            layers.append(result.module)
            result = result.parent
        net = torch.nn.Sequential(*reversed(layers))
    return json.dumps({"layers": _dump_layers(net)}, indent=2)
