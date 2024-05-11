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


def label_inference_attack(controller: FLController,
                           target_class,
                           test_loader,
                           threshold=0.5):
    _, y_probs = controller.predict(test_loader)
    y_probs = torch.tensor(y_probs)
    target_probs = y_probs[:, target_class]
    average_probs = target_probs.mean()

    if average_probs > threshold:
        return True
    return False


def feature_inference_attack(controller: FLController,
                             input_shape,
                             n_samples,
                             feature_index,
                             delta=0.1,
                             threshold=0.05):
    synthetic_input = torch.randn(n_samples, *input_shape)
    synthetic_label = torch.full((n_samples, ), 0)
    normal_dataset = torch.utils.data.TensorDataset(synthetic_input,
                                                    synthetic_label)
    normal_loader = torch.utils.data.DataLoader(normal_dataset,
                                                batch_size=n_samples,
                                                shuffle=False)

    synthetic_input[:, feature_index] += delta
    attack_dataset = torch.utils.data.TensorDataset(synthetic_input,
                                                    synthetic_label)
    attack_loader = torch.utils.data.DataLoader(attack_dataset,
                                                batch_size=n_samples,
                                                shuffle=False)

    normal_y_pred, _ = controller.predict(normal_loader)
    attack_y_pred, _ = controller.predict(attack_loader)
    normal_y_pred = torch.tensor(normal_y_pred)
    attack_y_pred = torch.tensor(attack_y_pred)

    confidence_diff = torch.abs(attack_y_pred - normal_y_pred)
    confidence_diff_ratio = confidence_diff / normal_y_pred
    if confidence_diff_ratio.mean() > threshold:
        return False
    return True
