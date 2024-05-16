import torch
import torch.utils.data
from tqdm import tqdm
import matplotlib.pyplot as plt


class LeakageInferenceAttack:

    def __init__(self, attack_model: torch.nn.Module, target_input_shape,
                 target_n_classes):
        self.attack_model = attack_model
        self.input_shape = target_input_shape
        self.n_classes = target_n_classes

    def reconstruct_inputs_from_grads(self,
                                      target_grads,
                                      labels,
                                      lr=0.01,
                                      n_iter=100,
                                      batch_size=10):
        n_samples = labels.shape[0]
        recon_data = torch.randn(n_samples,
                                 *self.input_shape,
                                 requires_grad=True)
        recon_optimizer = torch.optim.SGD([recon_data], lr=lr)
        recon_criterion = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(recon_data, labels)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

        progress_bar = tqdm(range(n_iter * len(loader)),
                            desc="Attacker reconstruct inputs")

        for _ in range(n_iter):
            for recon_data, labels in loader:
                recon_output = self.attack_model(recon_data)

                loss_class = recon_criterion(recon_output, labels)
                loss_class.backward(retain_graph=True)

                grad_diff = torch.tensor(0,
                                         dtype=torch.float32,
                                         requires_grad=True)
                attack_grads = [
                    param.grad for param in self.attack_model.parameters()
                ]
                self.attack_model.zero_grad()
                for target_grad, attack_grad in zip(target_grads,
                                                    attack_grads):
                    if attack_grad is not None:
                        grad_diff = grad_diff + torch.norm(target_grad -
                                                           attack_grad)
                grad_diff.backward()

                recon_optimizer.step()
                recon_optimizer.zero_grad()
                progress_bar.update(1)
        progress_bar.close()

        return recon_data.detach()

    def reconstruct_inputs_from_weights(self,
                                        target_weights,
                                        labels,
                                        lr=0.01,
                                        n_iter=100,
                                        batch_size=10):
        n_samples = labels.shape[0]
        recon_data = torch.randn(n_samples,
                                 *self.input_shape,
                                 requires_grad=True)
        recon_optimizer = torch.optim.SGD([recon_data], lr=lr)
        recon_criterion = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(recon_data, labels)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)

        progress_bar = tqdm(range(n_iter * len(loader)),
                            desc="Attacker reconstruct inputs")

        for _ in range(n_iter):
            for recon_data, labels in loader:
                recon_output = self.attack_model(recon_data)

                loss_class = recon_criterion(recon_output, labels)
                loss_class.backward(retain_graph=True)

                weight_diff = torch.tensor(0,
                                           dtype=torch.float32,
                                           requires_grad=True)
                attack_weights = self.attack_model.state_dict()
                self.attack_model.zero_grad()
                for target_key, target_weight in target_weights.items():
                    weight_diff = weight_diff + torch.norm(
                        target_weight - attack_weights[target_key])

                weight_diff.backward()

                recon_optimizer.step()
                recon_optimizer.zero_grad()
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
