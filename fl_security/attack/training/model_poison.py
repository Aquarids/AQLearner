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

# http://arxiv.org/abs/1802.07927
def finite_norm_grad_attack(benign_grads, gamma):
    benign_mean_grads = {}
    for key in benign_grads[0].keys():
        layer_grads = torch.stack([grad[key] for grad in benign_grads])
        benign_mean_grads[key] = layer_grads.mean(dim=0)

    byzantine_grads = {}
    for key, grad in benign_mean_grads.items():
        dim = grad.numel()
        e = torch.zeros(dim)
        attack_coord = torch.randint(0, dim, (1,)).item()
        e[attack_coord] = 1
        e = e.view(grad.shape)
        byzantine_grads[key] = grad + gamma * e

    return benign_mean_grads + gamma * e

def infinite_norm_grad_attack(grads):
    byzantine_grads = {key: torch.ones(dim) for key, dim in grads.items()}
    return byzantine_grads