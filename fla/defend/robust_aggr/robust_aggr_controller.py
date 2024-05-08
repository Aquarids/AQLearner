from fl.controller import FLController
from fla.defend.robust_aggr.robust_aggr_server import RobustAggrServer, type_aggr_median, type_aggr_trimmed_mean, type_aggr_krum

from tqdm import tqdm

class MedianAggrFLController(FLController):
    def __init__(self, server: RobustAggrServer, clients):
        super().__init__(server, clients)

    def aggregate_grads(self, grads):
        self.server.aggretate_gradients(grads, type_aggr_median)

    def aggregate_weights(self, weights):
        self.server.aggregate_weights(weights, type_aggr_median)

class TrimmedMeanAggrFLController(FLController):
    def __init__(self, server: RobustAggrServer, clients, trim_ratio=0.1):
        super().__init__(server, clients)
        self.trim_ratio = trim_ratio

    def aggregate_grads(self, grads):
        self.server.aggretate_gradients(grads, type_aggr_trimmed_mean, trim_ratio=self.trim_ratio)

    def aggregate_weights(self, weights):
        self.server.aggregate_weights(weights, type_aggr_trimmed_mean, trim_ratio=self.trim_ratio)

class KrumAggrFLController(FLController):
    def __init__(self, server: RobustAggrServer, clients, n_malicious=1):
        super().__init__(server, clients)
        self.n_malicious = n_malicious

    def aggregate_weights(self, weights):
        self.server.aggregate_weights(weights, type_aggr_krum, n_malicious=self.n_malicious)