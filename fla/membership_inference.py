import numpy as np
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

label_non_member = 0
label_member = 1


class ShadowModel:

    class ShadowCNN(torch.nn.Module):

        def __init__(self):
            super(ShadowModel.ShadowCNN, self).__init__()
            self.fc1 = torch.nn.Linear(10, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 32)
            self.fc4 = torch.nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x

    def __init__(self):
        self.model = ShadowModel.ShadowCNN()

    def train(self, loader, n_iters):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(range(n_iters * len(loader)),
                            desc="Shadow Model training")
        for _ in range(n_iters):
            for inputs, labels in loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
        progress_bar.close()

    def predict(self, loader):
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for inputs, target in loader:
                output = self.model(inputs)
                outputs.append((output, target))
        return outputs


class MembershipInferenceAttack:

    class AttackDataset(torch.utils.data.Dataset):

        def __init__(self, data, labels):
            self.data = data
            self.labels = labels

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx],
                                dtype=torch.float32), torch.tensor(
                                    self.labels[idx], dtype=torch.long)

    class AttackModel(torch.nn.Module):

        def __init__(self, input_size):
            super(MembershipInferenceAttack.AttackModel, self).__init__()
            self.fc1 = torch.nn.Linear(input_size, 64)
            self.fc2 = torch.nn.Linear(64, 32)
            self.fc3 = torch.nn.Linear(32, 2)  # 2 classes (member, non-member)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        def fit(self, loader, n_iters):
            self.train()
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            progress_bar = tqdm(range(n_iters * len(loader)),
                                desc="Attack Model training")
            for _ in range(n_iters):
                for inputs, labels in loader:
                    optimizer.zero_grad()
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    progress_bar.update(1)
            progress_bar.close()

        def evaluate(self, loader):
            self.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs.data, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            return correct / total

    def __init__(self, target_model: torch.nn.Module, input_shape, n_classes):
        self.target_model = target_model
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.membership_shadow_model = ShadowModel()
        self.non_membership_shadow_model = ShadowModel()
        self.attack_model = None

    def create_shadow_dataset(self, n_samples):
        random_inputs = torch.randn(n_samples, *self.input_shape)
        random_labels = torch.randint(0,
                                      self.n_classes, (n_samples, ),
                                      dtype=torch.long)
        member_datasets = torch.utils.data.TensorDataset(
            random_inputs, random_labels)
        member_loader = torch.utils.data.DataLoader(member_datasets,
                                                    batch_size=n_samples,
                                                    shuffle=False)

        _, member_probs = self.target_model.predict(member_loader)
        member_probs = np.array(member_probs)
        member_dataset = torch.utils.data.TensorDataset(
            torch.tensor(member_probs, dtype=torch.float32), random_labels)

        non_member_probs = torch.distributions.Dirichlet(
            torch.ones(self.n_classes)).sample((n_samples, ))
        non_member_dataset = torch.utils.data.TensorDataset(
            non_member_probs, random_labels)
        return member_dataset, non_member_dataset

    def train_shadow_model(self, member_loader, non_member_loader, n_iters):
        self.membership_shadow_model.train(member_loader, n_iters)
        self.non_membership_shadow_model.train(non_member_loader, n_iters)

    def attack(self, member_loader, non_member_loader, n_iters, train_ratio):
        member_preds = self._get_predictions(self.membership_shadow_model,
                                             member_loader)
        non_member_preds = self._get_predictions(
            self.non_membership_shadow_model, non_member_loader)
        attack_dataset = self._create_attack_dataset(member_preds,
                                                     non_member_preds)
        train_dataset, test_dataset = self._splite_attack_dataset(
            attack_dataset, train_ratio)

        attack_model = MembershipInferenceAttack.AttackModel(
            input_size=len(member_preds[0][0]))
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=100,
                                                   shuffle=True)
        attack_model.fit(train_loader, n_iters)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=100,
                                                  shuffle=False)
        accuracy = attack_model.evaluate(test_loader)
        return accuracy

    def _get_predictions(self, shadow_model, loader):
        preds = []
        shadow_outputs = shadow_model.predict(loader)
        for output, target in shadow_outputs:
            softmax_output = torch.nn.functional.softmax(output, dim=1)
            for i in range(len(target)):
                preds.append((softmax_output[i].numpy(), target[i].item()))
        return preds

    def _create_attack_dataset(self, member_probs, non_member_probs):
        attack_data = []
        attack_labels = []
        for pred, _ in member_probs:
            attack_data.append(pred)
            attack_labels.append(label_member)
        for pred, _ in non_member_probs:
            attack_data.append(pred)
            attack_labels.append(label_non_member)
        return MembershipInferenceAttack.AttackDataset(attack_data,
                                                       attack_labels)

    def _splite_attack_dataset(self, attack_dataset, train_ratio):
        n_train = int(len(attack_dataset) * train_ratio)
        n_test = len(attack_dataset) - n_train
        train_dataset, test_dataset = torch.utils.data.random_split(
            attack_dataset, [n_train, n_test])
        return train_dataset, test_dataset
