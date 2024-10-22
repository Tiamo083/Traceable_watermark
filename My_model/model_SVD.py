import collections
import os
import json
import torch
import random
import inspect
import accelerate
from pathlib import Path
from PIL import Image
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset

class model_SVD(nn.Module):
    def __init__(self, model, param_name, scale = 1.0):
        super().__init__()
        w = getattr(model, param_name)
        self.w_shape = w.shape
        U, S, Vh = torch.linalg.svd(w.detach().view(w.size(0), -1), full_matrices = False)
        self.register_buffer("U", U, persistent=False)
        self.register_buffer("S", S, persistent=False)
        self.register_buffer("Vh", Vh, persistent=False)
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.register_buffer("scale", torch.tensor(scale, device=w.device))
    
    def extra_repr(self):
        return f"Scale: {self.scale}"
    
    def forward(self, w):
        w_alt = self.U @ torch.diag(nn.functional.relu(self.S + self.scale * self.delta)) @ self.Vh
        return w_alt.view(*self.w_shape)

def apply_SVD(module, param_name = "weight", scale = 1.0):
    after_svd_model = model_SVD(module, param_name, scale)
    nn.utils.parametrize.register_parametrization(module, param_name, after_svd_model)
    getattr(module.parametrizations, param_name).original.data = torch.empty(0)
    return module

def convert_to_SVD(model):
    learnable_parameters = nn.ParameterList()
    learnable_parameters_1d = nn.ParameterList()
    for module in model.modules():
        if isinstance(module, nn.Conv2d): #  or isinstance(module, nn.Linear):
            apply_SVD(module, "weight")
            learnable_parameters.append(module.parametrizations.weight[0].delta)
        elif isinstance(module, nn.LayerNorm) or isinstance(module, nn.GroupNorm):
            apply_SVD(module, "weight")
            learnable_parameters_1d.append(module.parametrizations.weight[0].delta)
        elif isinstance(module, nn.MultiheadAttention):
            apply_SVD(module, "in_proj_weight")
            learnable_parameters.append(module.parametrizations.in_proj_weight[0].delta)
        elif isinstance(module, nn.Linear):
            apply_SVD(module, "bias")
            learnable_parameters.append(module.parametrizations.bias[0].delta)
            
    return learnable_parameters, learnable_parameters_1d

def load_model_for_svd(model):
    learnable_parameters, learnable_parameters_1d = convert_to_SVD(model)
    return {"params": learnable_parameters, "params_1d": learnable_parameters_1d}