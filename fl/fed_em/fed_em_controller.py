from fl.controller import FLController, mode_avg_vote
from fl.fed_em.fed_em_client import FedEMClient


class FedEMController(FLController):

    def __init__(self, server, clients: list[FedEMClient]):
        super().__init__(server, clients)

    def train(self, n_rounds, mode):
        if mode != mode_avg_vote:
            raise ValueError(f"Unsupported mode: {mode}")
        self.avg_vote_train(n_rounds)
