import torch
import torch.utils.data
import numpy as np
from tqdm import tqdm

from fl.controller import FLController


def model_inversion_attack(controller: FLController, target_class, n_samples,
                           input_shape, n_step, lr):
    synthetic_inputs = torch.randn(n_samples, *input_shape)
    synthetic_labels = torch.full((n_samples, ), target_class)

    datasets = torch.utils.data.TensorDataset(synthetic_inputs,
                                              synthetic_labels)
    loader = torch.utils.data.DataLoader(datasets,
                                         batch_size=n_samples,
                                         shuffle=False)

    progress_bar = tqdm(range(n_step), desc="Model inversion attack")
    for _ in range(n_step):
        _, y_probs = controller.predict(loader)
        y_probs = torch.tensor(y_probs)

        target_probs = y_probs[:, target_class]
        synthetic_grads = (1 - target_probs).view(
            -1, 1, 1, 1).expand_as(synthetic_inputs)
        synthetic_inputs.data += synthetic_grads * lr
        synthetic_inputs.data.clamp_(0, 1)

        progress_bar.update(1)
    progress_bar.close()

    return synthetic_inputs.detach()
