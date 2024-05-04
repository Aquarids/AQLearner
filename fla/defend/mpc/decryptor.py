import torch

class Decryptor:
    def __init__(self):
        self.reset()
        self.max_diff = 0

    def reset(self):
        self.noises = []

    def add_noise(self, noise):
        self.noises.append(noise)

    def decrypt_sum_gradients(self, grads):
        summed_noises = sum(self.noises)
        decrypted_grads = [grad - summed_noises for grad in grads]
        return decrypted_grads
    
    def verfiy_sum_gradients(self, original_grads, decrypted_grads):
        original_grads_flatten = torch.tensor(self._flatten_grads_list(original_grads))
        decrypted_grads_flatten = torch.tensor(self._flatten_grads_list(decrypted_grads))
        max_diff = torch.max(torch.abs(original_grads_flatten - decrypted_grads_flatten))
        if max_diff > self.max_diff:
            self.max_diff = max_diff
        assert torch.allclose(original_grads_flatten, decrypted_grads_flatten, rtol=1e-06, atol=1e-06), "Gradients differ beyond tolerance"
    
    def _flatten_grads_list(self, grads):
        return [item for tensor in grads for item in tensor.view(-1).tolist()]