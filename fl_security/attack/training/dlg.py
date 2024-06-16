import torch
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt


class DLG:

    def __init__(self, attack_model: torch.nn.Module, target_input_shape,
                 target_n_classes):
        self.attack_model = attack_model
        self.input_shape = target_input_shape
        self.n_classes = target_n_classes

    def reconstruct_inputs_from_grads(self,
                                      target_grads,
                                      lr=0.01,
                                      n_iter=100):
        recon_data = torch.randn(1,
                                 *self.input_shape,
                                 requires_grad=True)
        recon_label = torch.randn(1, self.n_classes, requires_grad=True)

        recon_optimizer = torch.optim.LBFGS([recon_data, recon_label], lr=lr)
        recon_criterion = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(recon_data, recon_label)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=True)

        progress_bar = tqdm(range(n_iter * len(loader)),
                            desc="Attacker reconstruct inputs")

        self.attack_model.train()
        for _ in range(n_iter):
            for recon_data, label in loader:

                recon_optimizer.zero_grad()
                self.attack_model.zero_grad()

                recon_output = self.attack_model(recon_data)

                recon_loss = recon_criterion(recon_output, label)

                recon_grad = torch.autograd.grad(recon_loss,
                                                 self.attack_model.parameters(),
                                                 create_graph=True)

                def closure():
                    grad_diff = 0
                    for target_grad, attack_grad in zip(target_grads,
                                                        recon_grad):
                        if attack_grad is not None:
                            grad_diff += ((target_grad - attack_grad) ** 2).sum()

                    return grad_diff

                recon_optimizer.step(closure)
                
                progress_bar.update(1)
        progress_bar.close()

        return recon_data.detach()

    def visualize(self, original_data, reconstructed_data):
        n_image = original_data.shape[0]
        for i in range(n_image):
            self.visualize_simple(original_data[i], reconstructed_data[i])

    def visualize_simple(self, original_data, reconstructed_data):
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(original_data.detach().numpy().reshape(
            self.input_shape[1:]))
        ax[0].set_title('Original')
        ax[1].imshow(reconstructed_data.detach().numpy().reshape(
            self.input_shape[1:]))
        ax[1].set_title('Reconstructed')
        plt.show()
