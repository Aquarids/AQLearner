import torch
import random

from tqdm import tqdm
from fl.client import Client
from fl.model_metric import ModelMetric
from fl.model_factory import type_regresion, type_binary_classification, type_multi_classification

class Server:
    def __init__(self, model: torch.nn.Module, optimizer, criterion, type=type_regresion, clients=[], encrypt=False):
        if len(clients) == 0:
            raise ValueError("Clients can not be empty")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.type = type
        self.model_metric = ModelMetric(type)
        self.clients = clients
        self.n_clients = len(clients)
        self.encrypt = encrypt
        self.noises = []
        if encrypt:
            self._setNoise()
    
    # for evaluating model after each round
    def setTestLoader(self, test_loader):
        self.test_loader = test_loader

    def _setNoise(self):
        self.noises = []
        for client in self.clients:
            noise = random.random()
            client.setGradientNoise(noise)
            self.noises.append(noise)

    def decrypt_sum_gradients(self, gradients):
        decrypted_gradients = []
        summed_noises = sum(self.noises)
        for grad in gradients:
            grad -= summed_noises
            decrypted_gradients.append(grad)
        return decrypted_gradients

    def train(self, n_rounds):
        self.model_metric.reset()
        progress_bar = tqdm(range(n_rounds * self.n_clients), desc="Training progress")
        for round_idx in range(n_rounds):
            gradients = []
            previous_client = None
            for client_id in range(self.n_clients):
                client: Client = self.clients[client_id]
                client.train(round_idx)

                # sum gradients from previous client, this action should not be bone in server class, here just for showing the idea
                previous_gradients = None
                if previous_client is not None:
                    previous_gradients = previous_client.get_summed_gradients()
                client.sum_gradients(previous_gradients)

                # get sum of gradients from last client, thus server do not know each client's gradient
                if client_id == self.n_clients - 1:
                    gradients = client.get_summed_gradients()
                previous_client = client
                progress_bar.update(1)
            self.aggretate_gradients(gradients)

            y_pred_value, y_prob_value = self.predict(self.test_loader)
            y_test_value = []
            for _, y in self.test_loader:
                y_test_value += y.tolist()
            self.model_metric.update(y_test_value, y_pred_value, y_prob_value, round_idx)

        progress_bar.close()
        
    def aggretate_gradients(self, gradients):
        if gradients is None:
            return
        
        if self.encrypt:
            decrypted_sum_gradients = self.decrypt_sum_gradients(gradients)

        avg_grad = []
        for grad in decrypted_sum_gradients:
            avg_grad.append(grad / self.n_clients)
        
        for param, grad in zip(self.model.parameters(), avg_grad):
            param.grad = grad

        self.model.train()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_model(self):
        return self.model
    
    def predict(self, loader):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            possiblities = []
            for X, _ in loader:
                if type_multi_classification == self.type:
                    possiblity = self.model(X)
                    possiblities += possiblity.tolist()
                    predictions += torch.argmax(possiblity, dim=1).tolist()
                elif type_binary_classification == self.type:
                    possiblity = self.model(X)
                    possiblities += possiblity.tolist()
                    predictions += torch.where(possiblity >= 0.5, 1, 0).tolist()
                elif type_regresion == self.type:
                    possiblity = None
                    predictions += self.model(X).tolist()
            return predictions, possiblities
        
    def summary(self):
        self.model_metric.summary()
        print("Global model: ", self.model)  
        print("Client Gradients Noise: ", self.noises)      
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
