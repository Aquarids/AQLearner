import torch
from tqdm import tqdm

from fl.client import Client
from fl.model_factory import type_multi_classification, type_binary_classification, type_regression

import fl_security.data_poison as DataPoison
import fl_security.model_poison as ModelPoison

attack_type_none = "no_attack"
# data poison
attack_label_flip = "label_flip"
attack_sample_poison = "sample_poison"
attack_ood_data = "ood_data"

# model poison
attack_gradient_poison = "gradient_poison"
attack_weight_poison = "weight_poison"

attack_backdoor = "backdoor"


class MaliciousClient(Client):

    def __init__(self, model, criterion, optimizer, type, attack_type):
        super().__init__(model, criterion, optimizer, type)
        self.attack_type = attack_type

    def setArgs(self, **kwargs):
        self.kwargs = kwargs

    def setDataLoader(self, train_loader, n_iters=10):
        if self.attack_type == attack_label_flip:
            if self.type == type_multi_classification:
                poisoned_loader = DataPoison.label_flip(
                    train_loader,
                    flip_ratio=self.kwargs["flip_ratio"],
                    num_classes=self.kwargs["num_classes"])
            elif self.type == type_binary_classification:
                poisoned_loader = DataPoison.binary_label_flip(train_loader)
            elif self.type == type_regression:
                raise ValueError(
                    "Regression does not support label flip attack")
        elif self.attack_type == attack_sample_poison:
            poisoned_loader = DataPoison.sample_poison(
                train_loader,
                poison_ratio=self.kwargs["poison_ratio"],
                noise_level=self.kwargs["noise_level"])
        elif self.attack_type == attack_ood_data:
            poisoned_loader = DataPoison.ood_data(
                train_loader, poison_ratio=self.kwargs["poison_ratio"])
        elif self.attack_type == attack_backdoor:
            if self.type == type_multi_classification:
                # trigger the backdoor only when the label is 1
                poisoned_loader = DataPoison.replace_label(train_loader,
                                                           target_label=1,
                                                           replace_label=0)
        else:
            # No attack
            poisoned_loader = train_loader
        return super().setDataLoader(poisoned_loader, n_iters)

    def train(self, round_idx=-1):
        if self.attack_type == attack_gradient_poison:
            progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                                desc="Client training progress")
            for _ in range(self.n_iters):
                for X_batch, y_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()

                    # Add gradient noise
                    ModelPoison.noise_gradient_attack(
                        self.model, noise_level=self.kwargs["noise_level"])

                    self.optimizer.step()
                    progress_bar.update(1)
            progress_bar.close()
        elif self.attack_type == attack_weight_poison:
            progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                                desc="Client training progress")
            for _ in range(self.n_iters):
                for X_batch, y_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    # Add weight noise
                    ModelPoison.noise_weight_attack(
                        self.model, noise_level=self.kwargs["noise_level"])

                    progress_bar.update(1)
            progress_bar.close()
        elif self.attack_type == attack_backdoor:
            progress_bar = tqdm(range(self.n_iters * len(self.train_loader)),
                                desc="Client training progress")
            for _ in range(self.n_iters):
                for X_batch, y_batch in self.train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = self.criterion(output, y_batch)
                    loss.backward()
                    self.optimizer.step()

                    # trigger the backdoor attack every 2 rounds
                    if round_idx % 2 == 1:
                        ModelPoison.random_weight(self.model, attack_ratio=0.5)

                    progress_bar.update(1)
            progress_bar.close()
        else:
            super().train()
