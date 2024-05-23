from fl.controller import FLController, mode_avg_weight
from fl.maml.maml_client import MAMLClient


class MAMLController(FLController):

    def __init__(self, server, clients):
        super().__init__(server, clients)

    def train(self, n_rounds, mode):
        if mode != mode_avg_weight:
            raise ValueError(f"Unsupported mode: {mode}")
        self.avg_weight_train(n_rounds)
