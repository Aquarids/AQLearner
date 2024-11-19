import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

class KeyGenerator(nn.Module):
    def __init__(self, input_size, key_size):
        super(KeyGenerator, self).__init__()
        self.fc = nn.Linear(input_size, key_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Eve(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Eve, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, encoded):
        return self.network(encoded)

    def _loss(output, target):
        return 1.0 - nn.BCELoss()(output, target)

input_size = 256 
hidden_size = 512
key_size = 256

bob = Autoencoder(input_size, hidden_size)
eve = Eve(input_size, hidden_size)
key_gen = KeyGenerator(input_size, key_size)

bob_optimizer = optim.Adam(bob.parameters(), lr=0.001)
eve_optimizer = optim.Adam(eve.parameters(), lr=0.001)
key_gen_optimizer = optim.Adam(key_gen.parameters(), lr=0.001)

def bob_loss(output, target):
    return nn.BCELoss()(output, target)

def eve_loss(output, target):
    return 1.0 - nn.BCELoss()(output, target)

def train(dataloader):

    for epoch in range(100):
        for data in dataloader:
            bob_optimizer.zero_grad()
            eve_optimizer.zero_grad()
            key_gen_optimizer.zero_grad()

            private_key = key_gen(data)
            encoded = bob(data + private_key)
            decoded = bob(encoded - private_key)

            eve_output = eve(encoded)

            loss_bob = bob_loss(decoded, data)
            loss_eve = eve_loss(eve_output, data)

            loss = loss_bob + loss_eve
            loss.backward()

            bob_optimizer.step()
            eve_optimizer.step()
            key_gen_optimizer.step()