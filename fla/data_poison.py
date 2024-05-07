import numpy as np
import torch
import torch.utils.data

from tqdm import tqdm

def binary_label_flip(loader):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)), desc='Poisoning data: Flip binary labels')
    for data, target in loader:
        target = 1 - target
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets, dim=0)), batch_size=loader.batch_size, shuffle=False)

def replace_label(loader, target_label: int, replace_label: int):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)), desc=f'Poisoning data: Replace label {target_label} to {replace_label}')
    for data, target in loader:
        target = torch.where(target == target_label, replace_label, target)
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets, dim=0)), batch_size=loader.batch_size, shuffle=False)

def label_flip(loader, flip_ratio: float, num_classes: int):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)), desc='Poisoning data: Flip labels')
    for data, target in loader:
        target = torch.tensor([np.random.choice(num_classes) if np.random.rand() < flip_ratio else t for t in target])
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets, dim=0)), batch_size=loader.batch_size, shuffle=False)

def sample_poison(loader, poison_ratio=0.1, noise_level=0.5):
    poisoned_samples = []
    poisoned_targets = []
    
    prgress_bar = tqdm(range(len(loader)), desc='Poisoning data: Sample poison')
    for data, target in loader:
        batch_size = data.size(0)
        n_poison = int(batch_size * poison_ratio)

        poison_data = torch.randn(n_poison, *data.shape[1:]) * noise_level + data[:n_poison]
        poison_target = torch.randint(0, target.max() + 1, (n_poison,))

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

    poisoned_dataset = torch.utils.data.TensorDataset(poisoned_samples, poisoned_targets)
    poisoned_loader = torch.utils.data.DataLoader(poisoned_dataset, batch_size=loader.batch_size, shuffle=True)

    return poisoned_loader

def ood_data(loader, poison_ratio=0.1):
    poisoned_samples = []
    poisoned_targets = []

    prgress_bar = tqdm(range(len(loader)), desc='Poisoning data: Out of distribution data')
    for data, target in loader:
        poison_index = np.random.choice(len(target), int(len(target) * poison_ratio), replace=False)
        data[poison_index] = torch.randn_like(data[poison_index])
        poisoned_samples.append(data)
        poisoned_targets.append(target)
        prgress_bar.update(1)
    prgress_bar.close()

    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.cat(poisoned_samples, dim=0), torch.cat(poisoned_targets, dim=0)), batch_size=loader.batch_size, shuffle=False)
