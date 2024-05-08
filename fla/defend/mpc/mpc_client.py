import torch
import torch.utils.data

from fl.client import Client
from fla.defend.mpc.mpc_encryptor import MPCEncryptor


class MPCClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, type):
        super().__init__(model, criterion, optimizer, type)
        self.encryptor = MPCEncryptor()

    def get_noise(self):
        return self.encryptor.get_noise()

    def get_gradients(self, prev_grads=None, prev_noise=None):
        original_grads = super().get_gradients()
        encrpted_grads = self.encryptor.encrypt_grads(original_grads)

        cur_summed_grads = None
        cur_summed_noise = None
        if prev_grads is None:
            cur_summed_grads = encrpted_grads
            cur_summed_noise = self.encryptor.get_noise()
        else:
            cur_summed_grads = [
                prev_grad + cur_grad
                for prev_grad, cur_grad in zip(prev_grads, encrpted_grads)
            ]
            cur_summed_noise = prev_noise + self.encryptor.get_noise()
        return cur_summed_grads, cur_summed_noise

    def get_weights(self, prev_weights=None, prev_noise=None):
        original_weights = super().get_weights()
        encrypted_weights = self.encryptor.encrypt_weights(original_weights)

        cur_summed_weights = {}
        cur_summed_noise = None
        if prev_weights is None:
            cur_summed_weights = encrypted_weights
            cur_summed_noise = self.encryptor.get_noise()
        else:
            for key, value in encrypted_weights.items():
                cur_summed_weights[key] = prev_weights[key] + value
            cur_summed_noise = prev_noise + self.encryptor.get_noise()
        return cur_summed_weights, cur_summed_noise
