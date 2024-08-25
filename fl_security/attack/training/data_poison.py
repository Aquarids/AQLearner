import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm
from scipy.stats import norm


def binary_label_flip(loader):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)),
                       desc='Poisoning data: Flip binary labels')
    for data, target in loader:
        target = 1 - target
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets,
                                                      dim=0)),
                                       batch_size=loader.batch_size,
                                       shuffle=False)


def replace_label(loader, target_label: int, replace_label: int):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(
        range(len(loader)),
        desc=f'Poisoning data: Replace label {target_label} to {replace_label}'
    )
    for data, target in loader:
        target = torch.where(target == target_label, replace_label, target)
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets,
                                                      dim=0)),
                                       batch_size=loader.batch_size,
                                       shuffle=False)


def label_flip(loader, flip_ratio: float, num_classes: int):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)), desc='Poisoning data: Flip labels')
    for data, target in loader:
        target = torch.tensor([
            np.random.choice(num_classes)
            if np.random.rand() < flip_ratio else t for t in target
        ])
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets,
                                                      dim=0)),
                                       batch_size=loader.batch_size,
                                       shuffle=False)


def sample_poison(loader, poison_ratio=0.1, noise_level=0.5):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)),
                       desc='Poisoning data: Sample poison')
    for data, target in loader:
        batch_size = data.size(0)
        n_poison = int(batch_size * poison_ratio)

        poison_data = torch.randn(
            n_poison, *data.shape[1:]) * noise_level + data[:n_poison]
        poison_target = torch.randint(0, target.max() + 1, (n_poison, ))

        combined_data = torch.cat([data[n_poison:], poison_data], dim=0)
        combined_target = torch.cat([target[n_poison:], poison_target], dim=0)

        indices = torch.randperm(batch_size)
        combined_data = combined_data[indices]
        combined_target = combined_target[indices]

        poisoned_samples.append(combined_data)
        poisoned_targets.append(combined_target)
        prgress_bar.update(1)
    prgress_bar.close()

    poisoned_samples = torch.cat(poisoned_samples, dim=0)
    poisoned_targets = torch.cat(poisoned_targets, dim=0)

    poisoned_dataset = torch.utils.data.TensorDataset(poisoned_samples,
                                                      poisoned_targets)
    poisoned_loader = torch.utils.data.DataLoader(poisoned_dataset,
                                                  batch_size=loader.batch_size,
                                                  shuffle=True)

    return poisoned_loader


def ood_data(loader, poison_ratio=0.1):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)),
                       desc='Poisoning data: Out of distribution data')
    for data, target in loader:
        poison_index = np.random.choice(len(target),
                                        int(len(target) * poison_ratio),
                                        replace=False)
        data[poison_index] = torch.randn_like(data[poison_index])
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets,
                                                      dim=0)),
                                       batch_size=loader.batch_size,
                                       shuffle=False)

# http://arxiv.org/abs/1902.06156
# against Trimmed Mean, Krum, and Bulyan
def mean_shift_convergence_attack(n_clients, malicious_clients_params):
    n_malicious = len(malicious_clients_params)
    n_required = (n_clients // 2 + 1) - n_malicious
    z_max = norm.ppf(1 - (n_clients - n_required) / n_required)

    # Number of params
    n_params = malicious_clients_params[0].shape[0]
    adjusted_params = np.zeros_like(malicious_clients_params)

    for j in range(n_params):
        mu_j = np.mean([p[j] for p in malicious_clients_params])
        sigma_j = np.std([p[j] for p in malicious_clients_params])

        adjusted_params[:, j] = mu_j + z_max * sigma_j

    for i in range(n_malicious):
        malicious_clients_params[i] = adjusted_params[i]

    return malicious_clients_params

def mean_shift_backdoor_attack(n_clients, malicious_clients_params, alpha, backdoor_train_fn):
    n_malicious = len(malicious_clients_params)
    n_required = (n_clients // 2 + 1) - n_malicious
    z_max = norm.ppf(1 - (n_clients - n_required) / n_required)

    n_params = malicious_clients_params[0].shape[0]
    mu_j = np.mean(malicious_clients_params, axis=0)
    sigma_j = np.std(malicious_clients_params, axis=0)

    V = backdoor_train_fn(mu_j, alpha)

    adjusted_params = np.zeros_like(malicious_clients_params)
    for j in range(n_params):
        for i in range(n_malicious):
            adjusted_params[i, j] = max(mu_j[j] - z_max * sigma_j[j], min(V[j], mu_j[j] + z_max * sigma_j[j]))

    for i in range(n_malicious):
        malicious_clients_params[i] = adjusted_params[i]

    return malicious_clients_params

# http://arxiv.org/abs/1807.00459
def _constrain_and_scale_backdoor_loss(y_pred, y_true, model, alpha, anomaly_penalty, p_norm):
    loss = torch.nn.functional.cross_entropy(y_pred, y_true)
    p_norm_loss = anomaly_penalty * torch.norm(torch.cat([p.view(-1) for p in model.parameters()]), p=p_norm)
    return alpha * loss + (1 - alpha) * p_norm_loss
        
def _estimate_upper_bound():
    raise NotImplementedError

def _scale_backdoor_model_weights(model, gamma):
    with torch.no_grad():
        for p in model.parameters():
            p.data *= gamma