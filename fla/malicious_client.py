from fl.client import Client
from fl.model_factory import type_multi_classification, type_binary_classification, type_regression
from fla.data_poison import DataPoison

attack_label_flip = "label_flip"
attack_sample_poison = "sample_poison"
attack_ood_data = "ood_data"

class MaliciousClient(Client):
    def __init__(self, model, criterion, optimizer, type, attack_type):
        super().__init__(model, criterion, optimizer, type)
        self.attack_type = attack_type

    def setDataLoader(self, train_loader, n_iters=10):
        data_poison = DataPoison()
        if self.attack_type == attack_label_flip:
            if self.type == type_multi_classification:
                poisoned_loader = data_poison.label_flip(train_loader, flip_ratio=0.7, num_classes=10)
            elif self.type == type_binary_classification:
                poisoned_loader = data_poison.binary_label_flip(train_loader, flip_ratio=0.7)
            elif self.type == type_regression:
                raise ValueError("Regression does not support label flip attack")
        elif self.attack_type == attack_sample_poison:
            poisoned_loader = data_poison.sample_poison(train_loader, poison_ratio=0.7, noise_level=10)
        elif self.attack_type == attack_ood_data:
            poisoned_loader = data_poison.ood_data(train_loader, poison_ratio=0.7)
        else:
            # No attack
            poisoned_loader = train_loader
        return super().setDataLoader(poisoned_loader, n_iters)