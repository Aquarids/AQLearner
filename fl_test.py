import torch
import torch.utils.data
import unittest
import sklearn.datasets
import sklearn.model_selection
import sklearn.preprocessing
from tqdm import tqdm
import numpy as np

import dl.metrics as Metrics
from fl.client import Client
from fl.server import Server, type_regresion, type_classification

class TestFL(unittest.TestCase):
    class SimpleLinearRegressionModel(torch.nn.Module):
        def __init__(self, n_features):
            super(TestFL.SimpleLinearRegressionModel, self).__init__()
            self.fc = torch.nn.Linear(n_features, 1)
        
        def forward(self, x):
            return self.fc(x)
        
    class SimpleLinearClassificationModel(torch.nn.Module):
        def __init__(self, n_features):
            super(TestFL.SimpleLinearClassificationModel, self).__init__()
            self.fc = torch.nn.Linear(n_features, 1)
        
        def forward(self, x):
            return torch.sigmoid(self.fc(x))
        
    def splite_data(self, X, y, n_clients, n_rounds):
        # the data should owned by the clinets themselves rather than the server, here just for showing the concept
        samples_per_client = len(X) // n_clients
        samples_per_round = samples_per_client // n_rounds
        samples_per_client_adjusted = samples_per_round * n_rounds
        
        total_samples_used = samples_per_client_adjusted * n_clients
        X_adjusted = X[:total_samples_used]
        y_adjusted = y[:total_samples_used]
        
        X_clients = torch.split(X_adjusted, samples_per_client_adjusted)
        y_clients = torch.split(y_adjusted, samples_per_client_adjusted)
        
        X_train_rounds, y_train_rounds = ([[] for _ in range(n_clients)] for _ in range(2))
        
        for client_idx, (X_client, y_client) in enumerate(zip(X_clients, y_clients)):
            rounds_X = torch.split(X_client, samples_per_round)
            rounds_y = torch.split(y_client, samples_per_round)
            
            for round_idx in range(n_rounds):
                X_train_rounds[client_idx].append(rounds_X[round_idx])
                y_train_rounds[client_idx].append(rounds_y[round_idx])

        return X_train_rounds, y_train_rounds


    def test_fl_regression(self):
        X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
        scaler = sklearn.preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=20)

        model = TestFL.SimpleLinearRegressionModel(n_features=X.shape[1])
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        n_clients = 10
        server = Server(model, optimizer, criterion, type=type_regresion)
        clients = [Client(model, criterion, optimizer) for _ in range(n_clients)]

        n_rounds = 10
        n_batch_size = 100
        n_iter = 10

        X_train_rounds, y_train_rounds = self.splite_data(X_train, y_train, n_clients, n_rounds)
        
        mse = []
        progress_bar = tqdm(range(n_rounds * n_clients), desc="Training progress")
        for round in range(n_rounds):
            gradients = []
            for client_id in range(n_clients):
                client = clients[client_id]
                X_train, y_train = X_train_rounds[client_id][round], y_train_rounds[client_id][round]
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=n_batch_size, shuffle=True)
                client.train(train_loader, n_iter)
                gradients.append(client.get_gradients())
                progress_bar.update(1)
            server.aggretate_gradients(gradients)

            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=n_batch_size, shuffle=True)
            y_pred_value, _ = server.predict(test_loader)
            y_test_value = []
            for _, y in test_loader:
                y_test_value += y.tolist()
            mse.append(Metrics.mse(np.array(y_test_value), np.array(y_pred_value)))

        progress_bar.close()
        print("MSE: ", mse)
        
        server.summary()

    def test_fl_classification(self):
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=42)

        model = TestFL.SimpleLinearClassificationModel(n_features=X.shape[1])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        n_clients = 10
        server = Server(model, optimizer, criterion, type=type_classification)
        clients = [Client(model, criterion, optimizer) for _ in range(n_clients)]

        n_rounds = 10
        n_batch_size = 100
        n_iter = 10

        X_train_rounds, y_train_rounds = self.splite_data(X_train, y_train, n_clients, n_rounds)
        
        accuracy = []
        progress_bar = tqdm(range(n_rounds * n_clients), desc="Training progress")
        for round in range(n_rounds):
            gradients = []
            for client_id in range(n_clients):
                client = clients[client_id]
                X_train, y_train = X_train_rounds[client_id][round], y_train_rounds[client_id][round]
                train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=n_batch_size, shuffle=True)
                client.train(train_loader, n_iter)
                gradients.append(client.get_gradients())
                progress_bar.update(1)
            server.aggretate_gradients(gradients)

            test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=n_batch_size, shuffle=True)
            y_pred_value, _ = server.predict(test_loader)
            y_test_value = []
            for _, y in test_loader:
                y_test_value += y.tolist()
            accuracy.append(Metrics.accuracy(np.array(y_test_value), np.array(y_pred_value)))

        progress_bar.close()
        print("Accuracy: ", accuracy)
        
        server.summary()

if __name__ == '__main__':
    unittest.main()