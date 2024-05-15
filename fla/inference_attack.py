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
    random_input = torch.randn(n_samples, *input_shape)
    random_label = torch.full((n_samples, ), 0)
    normal_dataset = torch.utils.data.TensorDataset(random_input, random_label)
    normal_loader = torch.utils.data.DataLoader(normal_dataset,
                                                batch_size=n_samples,
                                                shuffle=False)

    synthetic_input = random_input.clone()
    synthetic_input[:, feature_index] += delta

    attack_dataset = torch.utils.data.TensorDataset(synthetic_input,
                                                    random_label)
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
        return True
    return False


def sample_inference_attack(controller: FLController,
                            target_sample,
                            input_shape,
                            n_samples,
                            threshold=1):
    random_inputs = torch.randn(n_samples, *input_shape)
    random_labels = torch.full((n_samples, ), 0)
    target_input, target_label = target_sample

    random_inputs[0] = target_input
    random_labels[0] = target_label

    datasets = torch.utils.data.TensorDataset(random_inputs, random_labels)
    loader = torch.utils.data.DataLoader(datasets,
                                         batch_size=n_samples,
                                         shuffle=False)

    y_pred, _ = controller.predict(loader)
    y_pred = torch.tensor(y_pred)
    target_output = y_pred[0]
    random_outputs = y_pred[1:]

    random_mean = torch.mean(random_outputs, dim=0)
    random_std = torch.std(random_outputs, dim=0)
    target_z_score = abs((target_output - random_mean) / random_std)

    if target_z_score > threshold:
        return True
    return False
