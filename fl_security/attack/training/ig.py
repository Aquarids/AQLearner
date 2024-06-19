from fl_security.attack.training.idlg import iDLG

import torch
from collections import OrderedDict
from tqdm import tqdm

class IG(iDLG):

    class ConvNet(torch.nn.Module):

        def __init__(self, width=32, num_classes=10, num_channels=3):
            """Init with width and num classes."""
            super().__init__()
            self.model = torch.nn.Sequential(OrderedDict([
                ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
                ('bn0', torch.nn.BatchNorm2d(1 * width)),
                ('relu0', torch.nn.ReLU()),

                ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
                ('bn1', torch.nn.BatchNorm2d(2 * width)),
                ('relu1', torch.nn.ReLU()),

                ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
                ('bn2', torch.nn.BatchNorm2d(2 * width)),
                ('relu2', torch.nn.ReLU()),

                ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
                ('bn3', torch.nn.BatchNorm2d(4 * width)),
                ('relu3', torch.nn.ReLU()),

                ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('bn4', torch.nn.BatchNorm2d(4 * width)),
                ('relu4', torch.nn.ReLU()),

                ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('bn5', torch.nn.BatchNorm2d(4 * width)),
                ('relu5', torch.nn.ReLU()),

                ('pool0', torch.nn.MaxPool2d(3)),

                ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('bn6', torch.nn.BatchNorm2d(4 * width)),
                ('relu6', torch.nn.ReLU()),

                ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('bn6', torch.nn.BatchNorm2d(4 * width)),
                ('relu6', torch.nn.ReLU()),

                ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
                ('bn7', torch.nn.BatchNorm2d(4 * width)),
                ('relu7', torch.nn.ReLU()),

                ('pool1', torch.nn.MaxPool2d(3)),
                ('flatten', torch.nn.Flatten()),
                ('linear', torch.nn.Linear(36 * width, num_classes))
            ]))

        def forward(self, input):
            return self.model(input)
           
        def leak_grads(self, X, y):
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            y_pred = self(X)
            loss = criterion(y_pred, y)
            grads = torch.autograd.grad(loss, self.parameters())
            return [g.detach().clone() for g in grads]

    def __init__(self, target_model, input_shape, n_classes, tv_alpha=1e-4):
        self.shadow_model = target_model
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.total_variation = tv_alpha

    def reconstruct_inputs_from_grads(self, target_grads, lr=0.01, n_iter=100):


        recon_data = torch.randn(1,
                                 *self.input_shape,
                                 requires_grad=True)
        random_data = recon_data.detach().clone()
        recon_label = torch.randn(1, self.n_classes, requires_grad=True)
        pred_label = self.pred_true_label(target_grads)

        recon_optimizer = torch.optim.Adam([recon_data], lr=lr)
        recon_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        schedular = torch.optim.lr_scheduler.MultiStepLR(recon_optimizer, milestones=[n_iter // 2.667, n_iter // 1.6, n_iter // 1.142], gamma=0.1)

        
        def closure():
            recon_optimizer.zero_grad()

            recon_output = self.shadow_model(recon_data)
            recon_loss = recon_criterion(recon_output, pred_label)

            recon_grad = torch.autograd.grad(recon_loss, self.shadow_model.parameters(), create_graph=True)

            grad_diff = self._compute_cosine_cost(target_grads, recon_grad)
            grad_diff += self._total_variation(recon_data) * self.total_variation
            grad_diff.backward()

            return grad_diff

        progress_bar = tqdm(range(n_iter), desc="Attacker reconstruct inputs")
        for _ in range(n_iter):
            grad_diff = recon_optimizer.step(closure)
            progress_bar.set_postfix(loss=grad_diff.item())
            schedular.step()

            progress_bar.update(1)
                    
        progress_bar.close()
        
        print(f"recon label: {torch.argmax(recon_label).item()}")

        return random_data, recon_data.detach()
        
    def _compute_cosine_cost(self, target_grads, attack_grads):
        cosine_cost = 0
        for target_grad, attack_grad in zip(target_grads, attack_grads):
            cosine_cost += 1 - torch.nn.functional.cosine_similarity(target_grad.view(-1), attack_grad.view(-1), dim=0, eps=1e-10)
        return cosine_cost

    def _total_variation(self, data):
        dx = torch.mean(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]))
        dy = torch.mean(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]))
        return dx + dy


