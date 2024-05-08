import random


class MPCEncryptor:

    def __init__(self, range=1):
        self.range = range
        self.reset()

    def reset(self):
        self.noise = random.uniform(-self.range, self.range)

    def encrypt_grads(self, grads):
        return [grad + self.noise for grad in grads]

    def encrypt_weights(self, weights):
        encrypted_weights = {}
        for key, value in weights.items():
            encrypted_weights[key] = value + self.noise
        return encrypted_weights

    def get_noise(self):
        return self.noise
