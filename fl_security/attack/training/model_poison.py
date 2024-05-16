import numpy as np
import torch


def noise_gradient_attack(model, noise_level=0.1):
    for param in model.parameters():
        if param.grad is not None:
            param.grad += torch.randn_like(param.grad) * noise_level


def noise_weight_attack(model, noise_level=0.1):
    for param in model.parameters():
        param.data += torch.randn_like(param.data) * noise_level


def random_weight(model, attack_ratio=0.1):
    for param in model.parameters():
        if np.random.rand() < attack_ratio:
            param.data = torch.randn_like(param.data)
