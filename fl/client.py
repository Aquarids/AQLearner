import torch
import torch.utils.data

class Client:
    def __init__(self, model: torch.nn.Module, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loaders = []
        self.n_rounds = 0
        self.gradient_noise = 0
        self.summed_gradients = None

    def setGradientNoise(self, noise):
        self.gradient_noise = noise

    def setDataLoader(self, train_loader, n_rounds, n_batch_size=100, n_iters=10):
        self.n_rounds = n_rounds
        self.n_iters = n_iters
        self.train_loaders = self._slipt_data(train_loader, batch_size=n_batch_size, n_rounds=n_rounds)

    # assume client decide to splite the data into n_rounds
    def _slipt_data(self, train_loader, batch_size, n_rounds):
        X, y = train_loader.dataset.tensors
        samples_per_client = len(X) // n_rounds
        total_samples_used = samples_per_client * n_rounds
        
        X_adjusted = X[:total_samples_used]
        y_adjusted = y[:total_samples_used]
        
        X_train_rounds = torch.split(X_adjusted, samples_per_client)
        y_train_rounds = torch.split(y_adjusted, samples_per_client)

        train_loaders = []
        for round_idx in range(n_rounds):
            train_loaders.append(torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train_rounds[round_idx], y_train_rounds[round_idx]), batch_size=batch_size, shuffle=True))
        
        return train_loaders

    def train(self, round_idx=0):
        self.model.train()
        train_loader = self.train_loaders[round_idx]
        for _ in range(self.n_iters):
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

    def _encrypt_gradient(self, gradient):
        return gradient + self.gradient_noise
    
    # sum other clients' gradients with noise
    def sum_gradients(self, previous_gradients):
        cur_gradients = [self._encrypt_gradient(param.grad) for param in self.model.parameters()]
        if previous_gradients is None:
            self.summed_gradients = cur_gradients
        else:
            self.summed_gradients = [cur_grad + prev_grad for cur_grad, prev_grad in zip(cur_gradients, previous_gradients)]

        return self.summed_gradients
    
    # just for verifying the sum gradients, should not be used
    def get_gradients(self):
        return [param.grad for param in self.model.parameters()]

    def get_summed_gradients(self):
        return self.summed_gradients
    
    def update_model(self, new_model):
        self.model.load_state_dict(new_model.state_dict())