import torch
import random

from tqdm import tqdm
from fla.defend.mpc.client import Client
from fla.defend.mpc.decryptor import Decryptor
from fl.model_metric import ModelMetric
from fl.model_factory import type_regression, type_binary_classification, type_multi_classification

class Server:
    def __init__(self, model: torch.nn.Module, optimizer, criterion, type=type_regression, clients=[]):
        if len(clients) == 0:
            raise ValueError("Clients can not be empty")
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.type = type
        self.model_metric = ModelMetric(type)
        self.clients = clients
        self.n_clients = len(clients)

        self.decryptor = Decryptor()
    
    # for evaluating model after each round
    def setTestLoader(self, test_loader):
        self.test_loader = test_loader

    def train(self, n_rounds):
        self.model_metric.reset()
        progress_bar = tqdm(range(n_rounds * self.n_clients), desc="Training progress")
        for round_idx in range(n_rounds):
            self.decryptor.reset()
            gradients = []
            previous_client = None
            for client_id in range(self.n_clients):
                client: Client = self.clients[client_id]
                client.train(round_idx)
                self.decryptor.add_noise(client.get_noise())

                # sum gradients from previous client, this action should not be bone in server class, here just for showing the idea
                if previous_client is not None:
                    client.sum_gradients(previous_client.get_summed_gradients())
                else:
                    client.sum_gradients(None)
                previous_client = client

                # get sum of gradients from last client, thus server do not know each client's gradient
                if client_id == self.n_clients - 1:
                    gradients = client.get_summed_gradients()
                
                progress_bar.update(1)
            self.aggretate_gradients(gradients)

            y_pred_value, y_prob_value = self.predict(self.test_loader)
            y_test_value = []
            for _, y in self.test_loader:
                y_test_value += y.tolist()
            self.model_metric.update(y_test_value, y_pred_value, y_prob_value, round_idx)

        progress_bar.close()

    def verify_grads(self, decrypted_sum_grads):
        # verify the sum gradients, should not be used in real case
        original_summed_grads = None
        prev_grads = None
        cur_grads = None
        for client in self.clients:
            cur_grads = client.get_original_gradients()
            if prev_grads is None:
                original_summed_grads = cur_grads
            else:
                original_summed_grads = [prev_grad + cur_grad for prev_grad, cur_grad in zip(prev_grads, cur_grads)]
            prev_grads = original_summed_grads

        self.decryptor.verfiy_sum_gradients(original_summed_grads, decrypted_sum_grads)

        
    def aggretate_gradients(self, encrypted_summed_grads):
        if encrypted_summed_grads is None:
            return
        
        decrypted_sum_grads = self.decryptor.decrypt_sum_gradients(encrypted_summed_grads)
        self.verify_grads(decrypted_sum_grads)

        avg_grad = []
        for grad in decrypted_sum_grads:
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
            possibilities = []
            for X, _ in loader:
                if type_multi_classification == self.type:
                    possiblity = self.model(X)
                    possibilities += possiblity.tolist()
                    predictions += torch.argmax(possiblity, dim=1).tolist()
                elif type_binary_classification == self.type:
                    possiblity = self.model(X)
                    possibilities += possiblity.tolist()
                    predictions += torch.where(possiblity >= 0.5, 1, 0).tolist()
                elif type_regression == self.type:
                    possiblity = None
                    predictions += self.model(X).tolist()
            return predictions, possibilities
        
    def summary(self):
        print(f"Max diff of gradients: {self.decryptor.max_diff}")
        self.model_metric.summary()
        print("Global model: ", self.model)   
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
