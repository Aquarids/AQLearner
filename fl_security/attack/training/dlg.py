import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt


class DLG:

    class Utils:

        def weights_init(m):
            if hasattr(m, "weight"):
                m.weight.data.uniform_(-0.5, 0.5)
            if hasattr(m, "bias"):
                m.bias.data.uniform_(-0.5, 0.5)

        def label_to_onehot(target, num_classes=100):
            target = torch.unsqueeze(target, 1)
            onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
            onehot_target.scatter_(1, target, 1)
            return onehot_target

        def cross_entropy_for_onehot(pred, target):
            return torch.mean(torch.sum(- target * torch.functional.F.log_softmax(pred, dim=-1), 1))

    # LeNet as the attack model
    class LeNet(torch.nn.Module):

        def __init__(self):
            super(DLG.LeNet, self).__init__()
            act = torch.nn.Sigmoid
            self.body = torch.nn.Sequential(
                torch.nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
                act(),
                torch.nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
                act(),
                torch.nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
                act(),
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(768, 100)
            )

            # self.apply(DLG.Utils.weights_init)
            
        def forward(self, x):
            out = self.body(x)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

        def leak_grads(self, X, y):
            y_pred = self(X)
            loss = DLG.Utils.cross_entropy_for_onehot(y_pred, y)
            grads = torch.autograd.grad(loss, self.parameters())
            return [g.detach().clone() for g in grads]


    def __init__(self, target_model, input_shape, n_classes):
        self.shadow_model = DLG.LeNet()
        self.shadow_model.load_state_dict(target_model.state_dict())
        self.input_shape = input_shape
        self.n_classes = n_classes

    def reconstruct_inputs_from_grads(self,
                                      target_grads,
                                      lr=0.01,
                                      n_iter=100):
        recon_data = torch.randn(1,
                                 *self.input_shape,
                                 requires_grad=True)
        recon_label = torch.randn(1, self.n_classes, requires_grad=True)

        recon_optimizer = torch.optim.LBFGS([recon_data, recon_label], lr=lr)
        recon_criterion = DLG.Utils.cross_entropy_for_onehot

        progress_bar = tqdm(range(n_iter), desc="Attacker reconstruct inputs")
        def closure():
            recon_optimizer.zero_grad()
            recon_output = self.shadow_model(recon_data)
            onehot_recon_label = torch.nn.functional.softmax(recon_label, dim=-1)

            recon_loss = recon_criterion(recon_output, onehot_recon_label)
            recon_grad = torch.autograd.grad(recon_loss, self.shadow_model.parameters(), create_graph=True)

            grad_diff = 0
            for target_grad, attack_grad in zip(target_grads, recon_grad):
                grad_diff += ((target_grad - attack_grad) ** 2).sum()

            grad_diff.backward()

            progress_bar.set_postfix(loss=grad_diff.item())
            return grad_diff

        for _ in range(n_iter):
            recon_optimizer.step(closure)
            progress_bar.update(1)
                    
        progress_bar.close()

        return recon_data.detach()

    def visualize(self, original_data, reconstructed_data):
        fig, ax = plt.subplots(1, 2)
        to_pil_image = torchvision.transforms.ToPILImage()

        ax[0].imshow(to_pil_image(original_data))
        ax[0].set_title('Original')
        ax[1].imshow(to_pil_image(reconstructed_data))
        ax[1].set_title('Reconstructed')
        plt.show()
