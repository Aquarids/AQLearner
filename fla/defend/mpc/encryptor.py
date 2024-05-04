import random

class Encryptor:
    def __init__(self, range=1):
        self.range = range
        self.reset()

    def reset(self):
        self.noise = random.uniform(-self.range, self.range)
        self.cur_summed_grads = None

    def encrypt(self, grads):
        return [grad + self.noise for grad in grads]

    def get_noise(self):
        return self.noise
    
    def sum_encrpted_grads(self, prev_grads, params):
        original_grads = [param.grad for param in params]
        encrpted_grads = self.encrypt(original_grads)

        if prev_grads is None:
            self.cur_summed_grads = encrpted_grads
        else:
            self.cur_summed_grads = [prev_grad + cur_grad for prev_grad, cur_grad in zip(prev_grads, encrpted_grads)]

    def get_summed_grads(self):
        return self.cur_summed_grads