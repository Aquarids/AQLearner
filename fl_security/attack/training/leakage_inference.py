import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


class LeakageInferenceAttack:

    def __init__(self, attack_model: torch.nn.Module, target_input_shape,
                 target_n_classes):
        self.attack_model = attack_model
        self.input_shape = target_input_shape
        self.n_classes = target_n_classes

    def reconstruct_inputs(self, target_grads, n_samples, lr=0.01, n_iter=100):
        recon_data = torch.randn(n_samples,
                                 *self.input_shape,
                                 requires_grad=True)
        recon_optimizer = torch.optim.SGD([recon_data], lr=lr)
        recon_criterion = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(range(n_iter), desc="Attacker reconstruct inputs")
        for _ in range(n_iter):
            recon_optimizer.zero_grad()
            recon_output = self.attack_model(recon_data)
            recon_loss = torch.tensor(0,
                                      requires_grad=True,
                                      dtype=torch.float32)
            for class_idx in range(self.n_classes):
                target_class = torch.tensor([class_idx] * n_samples)
                loss_class = recon_criterion(recon_output, target_class)
                self.attack_model.zero_grad()
                loss_class.backward(retain_graph=True)

                grad_diff = torch.tensor(0,
                                         dtype=torch.float32,
                                         requires_grad=True)
                attack_grads = [
                    param.grad for param in self.attack_model.parameters()
                ]
                for target_grad, attack_grad in zip(target_grads,
                                                    attack_grads):
                    grad_diff = grad_diff + torch.norm(target_grad -
                                                       attack_grad)

                recon_loss = recon_loss + grad_diff

            recon_loss.backward()
            recon_optimizer.step()
            progress_bar.update(1)
        progress_bar.close()

        return recon_data.detach()

    def visualize(self, original_data, reconstructed_data):
        n_image = original_data.shape[0]
        for i in range(n_image):
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(original_data[i].detach().numpy().reshape(
                self.input_shape[1:]))
            ax[0].set_title('Original')
            ax[1].imshow(reconstructed_data[i].detach().numpy().reshape(
                self.input_shape[1:]))
            ax[1].set_title('Reconstructed')
            plt.show()
