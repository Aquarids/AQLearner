from fl.controller import FLController, mode_avg_weight
from fl.fed_adapt_opt.fed_adapt_opt_server import FedAdaptOptServer


class FedAdaptOptController(FLController):

    def __init__(self, server: FedAdaptOptServer, clients):
        super().__init__(server, clients)

    def train(self, n_rounds, mode):
        if mode != mode_avg_weight:
            raise ValueError(f"Unsupported mode: {mode}")
        self.avg_weight_train(n_rounds)
