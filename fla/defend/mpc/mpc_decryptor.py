import torch


class MPCDecryptor:

    def __init__(self):
        self.max_diff = 0

    def decrypt_summed_gradients(self, grads, summed_noise):
        decrypted_grads = [grad - summed_noise for grad in grads]
        return decrypted_grads

    def decrypt_weights(self, weights, summed_noise):
        decrypted_weights = {}
        for key, value in weights.items():
            decrypted_weights[key] = value - summed_noise
        return decrypted_weights

    def verfiy_summed_gradients(self, original_grads, decrypted_grads):
        original_grads_flatten = torch.tensor(
            self._flatten_grads_list(original_grads))
        decrypted_grads_flatten = torch.tensor(
            self._flatten_grads_list(decrypted_grads))
        max_diff = torch.max(
            torch.abs(original_grads_flatten - decrypted_grads_flatten))
        if max_diff > self.max_diff:
            self.max_diff = max_diff
        assert torch.allclose(original_grads_flatten,
                              decrypted_grads_flatten,
                              rtol=1e-06,
                              atol=1e-06), "Gradients differ beyond tolerance"

    def _flatten_grads_list(self, grads):
        return [item for tensor in grads for item in tensor.view(-1).tolist()]
