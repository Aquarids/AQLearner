import numpy as np
import torch
import torch.utils.data

import torch.utils.data.dataloader
from tqdm import tqdm


def laplace_dp(data: torch.Tensor, epsilon, sensitivity):
    return data + torch.tensor(np.random.laplace(
        loc=0, scale=sensitivity / epsilon, size=data.size()),
                               dtype=data.dtype)


def gaussian_dp(data: torch.Tensor, epsilon, delta, sensitivity):
    return data + torch.tensor(np.random.normal(
        loc=0,
        scale=sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon,
        size=data.size()),
                               dtype=data.dtype)


def clip_and_discretize_labels(noisy_labels, num_classes):
    clipped_labels = torch.clamp(noisy_labels, min=0, max=num_classes - 1)
    labels = torch.round(clipped_labels)
    return labels


class DPSGD:

    def __init__(self, model: torch.nn.Module, sigma, clip_value, delta):
        self.model = model
        self.sigma = sigma
        self.clip_value = clip_value
        self.delta = delta

    def _clip_and_noise_grad(self):
        total_norm = 0
        for param in self.model.parameters():
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item()**2
        total_norm = total_norm**0.5

        clip_coef = self.clip_value / (total_norm + 1e-6)
        if clip_coef < 1:
            for param in self.model.parameters():
                param.grad.data.mul_(clip_coef)

        for param in self.model.parameters():
            noise = torch.randn_like(param.grad.data)
            param.grad.data.add_(noise, alpha=self.sigma)

    def train(self, loader, n_iters):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(range(n_iters * len(loader)),
                            desc="DPSGD Training")
        for _ in range(n_iters):
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                self._clip_and_noise_grad()
                optimizer.step()
                progress_bar.update(1)
        progress_bar.close()


class PATE:

    def __init__(self,
                 model: torch.nn.Module,
                 teachers: list[torch.nn.Module],
                 epsilon,
                 sensitivity,
                 n_classes=-1):
        self.model = model
        self.teachers = teachers
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.n_classes = n_classes

    def _split_loader(self, loader: torch.utils.data.DataLoader, n_parts):
        part_len = len(loader) // n_parts

        all_inputs = []
        all_labels = []
        for inputs, labels in loader:
            all_inputs.append(inputs)
            all_labels.append(labels)
        all_inputs = torch.cat(all_inputs)
        all_labels = torch.cat(all_labels)

        loaders = []
        for i in range(n_parts):
            start = i * part_len
            end = (i + 1) * part_len
            inputs = all_inputs[start:end]
            labels = all_labels[start:end]
            dataset = torch.utils.data.TensorDataset(inputs, labels)

            loaders.append(
                torch.utils.data.DataLoader(dataset,
                                            batch_size=loader.batch_size,
                                            shuffle=True))
        return loaders

    def _train_teachers(self, loaders, n_iters):

        progress_bar = tqdm(range(len(loaders) * len(loaders[0]) * n_iters),
                            desc="PATE Teachers Training")
        for teacher, loader in zip(self.teachers, loaders):
            teacher.train()

            optimizer = torch.optim.SGD(teacher.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()
            for _ in range(n_iters):
                for inputs, labels in loader:
                    optimizer.zero_grad()
                    outputs = teacher(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    progress_bar.update(1)
        progress_bar.close()

    def _aggr_teacher(self, loader):
        vote_preds = []
        for input, _ in loader:
            teacher_preds = []
            for teacher in self.teachers:
                teacher.eval()
                predicitons = []
                with torch.no_grad():
                    probs = teacher(input)
                    predicitons += torch.argmax(probs, dim=1).tolist()
                    teacher_preds.append(predicitons)
            vote_pred = torch.tensor(teacher_preds).mode(dim=0).values
            vote_preds.append(vote_pred)
        vote_preds = torch.cat(vote_preds)

        noisy_labels = laplace_dp(vote_preds, self.epsilon, self.sensitivity)
        noisy_labels = clip_and_discretize_labels(noisy_labels, self.n_classes)
        return noisy_labels

    def _train_student(self, noisy_labels, loader, n_iters):
        loader_inputs = []
        for input, _ in loader:
            loader_inputs.append(input)
        loader_inputs = torch.cat(loader_inputs)

        noisy_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(loader_inputs, noisy_labels),
            batch_size=loader.batch_size,
            shuffle=True)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(range(len(noisy_loader) * n_iters),
                            desc="PATE Model Training")
        for _ in range(n_iters):
            for inputs, labels in noisy_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        progress_bar.close()

    def train(self, loader, n_iters):
        n_part = len(self.teachers) + 1
        loaders = self._split_loader(loader, n_part)
        self._train_teachers(loaders[:-1], n_iters)
        noisy_labels = self._aggr_teacher(loaders[-1])
        self._train_student(noisy_labels, loaders[-1], n_iters)
