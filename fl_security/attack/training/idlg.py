from fl_security.attack.training.dlg import DLG

import torch
from tqdm import tqdm

class iDLG(DLG):

    class LeNet(DLG.LeNet):
           
        def leak_grads(self, X, y):
            criterion = torch.nn.CrossEntropyLoss()
            y_pred = self(X)
            loss = criterion(y_pred, y)
            grads = torch.autograd.grad(loss, self.parameters())
            return [g.detach().clone() for g in grads]

    def __init__(self, target_model, input_shape, n_classes):
        self.shadow_model = target_model
        self.input_shape = input_shape
        self.n_classes = n_classes


    def pred_true_label(self, target_grads):
        # use the last activation layer, the diff of true label should be minimal
        layer_grad = torch.sum(target_grads[-2], dim=-1)
        label_pred = torch.argmin(layer_grad, dim=-1).detach().reshape((1,))
        return label_pred

    def reconstruct_inputs_from_grads(self,
                                      target_grads,
                                      lr=1,
                                      n_iter=100):
        recon_data = torch.randn(1,
                                 *self.input_shape,
                                 requires_grad=True)
        random_data = recon_data.detach().clone()
        recon_label = torch.randn(1, self.n_classes, requires_grad=True)
        pred_label = self.pred_true_label(target_grads)

        recon_optimizer = torch.optim.LBFGS([recon_data, ], lr=lr)
        recon_criterion = torch.nn.CrossEntropyLoss()

        progress_bar = tqdm(range(n_iter), desc="Attacker reconstruct inputs")
        def closure():
            recon_optimizer.zero_grad()

            recon_output = self.shadow_model(recon_data)
            recon_loss = recon_criterion(recon_output, pred_label)

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
        
        print(f"recon label: {torch.argmax(recon_label).item()}")

        return random_data, recon_data.detach()