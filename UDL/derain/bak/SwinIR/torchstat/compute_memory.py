import torch.nn as nn
import torch
import numpy as np


def compute_memory(module, inp, out):
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
        return compute_ReLU_memory(module, inp, out)
    elif isinstance(module, nn.PReLU):
        return compute_PReLU_memory(module, inp, out)
    elif isinstance(module, nn.Conv2d):
        return compute_Conv2d_memory(module, inp, out)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_BatchNorm2d_memory(module, inp, out)
    elif isinstance(module, nn.LayerNorm):
        return compute_LayerNorm_memory(module, inp, out)
    elif isinstance(module, nn.Linear):
        return compute_Linear_memory(module, inp, out)
    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
        return compute_Pool2d_memory(module, inp, out)
    else:
        print(f"[Memory]: {type(module).__name__} is not supported!")
        return (0, 0)
    pass


def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compute_ReLU_memory(module, inp, out):
    assert isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU))
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


def compute_PReLU_memory(module, inp, out):
    assert isinstance(module, (nn.PReLU))
    batch_size = inp.size()[0]
    mread = batch_size * (inp.size()[1:].numel() + num_params(module))
    mwrite = batch_size * inp.size()[1:].numel()

    return (mread, mwrite)


def compute_Conv2d_memory(module, inp, out):
    # Can have multiple inputs, getting the first one
    assert isinstance(module, nn.Conv2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())

    batch_size = inp.size()[0]
    in_c = inp.size()[1]
    out_c, out_h, out_w = out.size()[1:]

    # This includes weighs with bias if the module contains it.
    mread = batch_size * (inp.size()[1:].numel() + num_params(module))
    mwrite = batch_size * out_c * out_h * out_w
    return (mread, mwrite)


def compute_BatchNorm2d_memory(module, inp, out):
    assert isinstance(module, nn.BatchNorm2d)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size, in_c, in_h, in_w = inp.size()

    mread = batch_size * (inp.size()[1:].numel() + 2 * in_c)
    mwrite = inp.size().numel()
    return (mread, mwrite)

def compute_LayerNorm_memory(module, inp, out):
    assert isinstance(module, nn.LayerNorm)
    if len(inp.size()) == 3:
        inp = inp.unsqueeze(0)
    if len(out.size()) == 3:
        out = out.unsqueeze(0)
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size, in_c = inp.size()[:2]

    mread = batch_size * (inp.size()[2:].numel() + 2 * in_c)
    mwrite = inp.size().numel()
    return (mread, mwrite)

def compute_Linear_memory(module, inp, out):
    assert isinstance(module, nn.Linear)
    if len(inp.size()) > 3:
        inp = inp.reshape(inp.size(0), inp.size(1), -1)
    if len(out.size()) > 3:
        out = out.reshape(out.size(0), out.size(1), -1)

    batch_size = inp.size()[0]
    if len(inp.size()) == 3:# and inp.size(0) == 1:
        inp = inp[0, ...]#.squeeze(0)
    if len(out.size()) == 3:# and out.size(0) == 1:
        out = out[0, ...]#.squeeze(0)
    assert len(inp.size()) == 2 and len(out.size()) == 2

    mread = batch_size * (inp.size()[1:].numel() + num_params(module))
    mwrite = out.size().numel()

    return (mread, mwrite)


def compute_Pool2d_memory(module, inp, out):
    assert isinstance(module, (nn.MaxPool2d, nn.AvgPool2d))
    assert len(inp.size()) == 4 and len(inp.size()) == len(out.size())
    batch_size = inp.size()[0]
    mread = batch_size * inp.size()[1:].numel()
    mwrite = batch_size * out.size()[1:].numel()
    return (mread, mwrite)
