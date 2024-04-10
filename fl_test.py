import torch
import torch.utils.data
import unittest
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing

from fl.client import Client
from fl.server import Server
from fl.model_factory import ModelFactory

class TestModelFactory(unittest.TestCase):
    def test_model_factory(self):
        model_factory = ModelFactory()
        model_params = {
            "model_type": "regression",
            "learning_rate": 0.001,
            "optimizer": "sgd",
            "criterion": "mse",
            "layers": [
                {
                    "type": "linear",
                    "in_features": 10,
                    "out_features": 5
                },
                {
                    "type": "linear",
                    "in_features": 5,
                    "out_features": 1
                }
            ]
        }
        model, model_type, optimizer, criterion = model_factory.create_model(model_params)
        
        print("Model Type: ", model_type)
        print("Model Detail: ", model)        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params}")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name}, Size: {param.size()}, Values: {param.data}")
        print("Optimizer: ", optimizer)
        print("criterion: ", criterion)

class TestFL(unittest.TestCase):        
    class SimpleLinearClassificationModel(torch.nn.Module):
        def __init__(self, n_features):
            super(TestFL.SimpleLinearClassificationModel, self).__init__()
            self.fc = torch.nn.Linear(n_features, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.fc(x))
        
    def splite_data(self, X, y, n_clients):
        # the data should owned by the clinets themselves rather than the server, here just for showing the concept
        total_samples = len(X)
        samples_per_client = total_samples // n_clients
        total_samples_used = samples_per_client * n_clients

        X_adjusted = X[:total_samples_used]
        y_adjusted = y[:total_samples_used]
        
        X_clients = torch.split(X_adjusted, samples_per_client)
        y_clients = torch.split(y_adjusted, samples_per_client)
        
        train_loader_clients = []
        for client_id in range(n_clients):
            train_loader_clients.append(torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_clients[client_id], y_clients[client_id]), batch_size=1, shuffle=True))
        return train_loader_clients


    def test_fl_regression(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=20)

        model_json = {
            "model_type": "regression",
            "learning_rate": 0.01,
            "optimizer": "adam",
            "criterion": "mse",
            "layers": [
                {
                    "type": "linear",
                    "in_features": X.shape[1],
                    "out_features": 1
                }
            ]
        }

        n_clients = 10
        clients = []
        for i in range(n_clients):
            # each client should have its own model
            model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)
            client = Client(model, criterion, optimizer)
            clients.append(client)

        model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)
        server = Server(model, optimizer, criterion, type=model_type, clients=clients)

        n_rounds = 10
        n_batch_size = 100
        n_iter = 10

        train_loader_clients = self.splite_data(X_train, y_train, n_clients)
        for client_id in range(n_clients):
            clients[client_id].setDataLoader(train_loader_clients[client_id], n_batch_size, n_rounds, n_iter)
        
        server.setTestLoader(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=n_batch_size, shuffle=True))
        server.train(n_rounds)
        
        server.summary()

    def test_fl_classification(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        model_json = {
            "model_type": "binary_classification",
            "learning_rate": 0.01,
            "optimizer": "adam",
            "criterion": "bce",
            "layers": [
                {
                    "type": "linear",
                    "in_features": X.shape[1],
                    "out_features": 1
                },
                {
                    "type": "sigmoid"
                }
            ]
        }
        model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)

        n_clients = 10
        clients = []
        for i in range(n_clients):
            # each client should have its own model
            model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)
            client = Client(model, criterion, optimizer)
            clients.append(client)

        model, model_type, optimizer, criterion = ModelFactory().create_model(model_json)
        server = Server(model, optimizer, criterion, type=model_type, clients=clients)

        n_rounds = 5
        n_batch_size = 2
        n_iter = 10

        train_loader_clients = self.splite_data(X_train, y_train, n_clients)
        for client_id in range(n_clients):
            clients[client_id].setDataLoader(train_loader_clients[client_id], n_rounds, n_batch_size, n_iter)
        
        server.setTestLoader(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=n_batch_size, shuffle=True))
        server.train(n_rounds)
        
        server.summary()

if __name__ == '__main__':
    unittest.main()