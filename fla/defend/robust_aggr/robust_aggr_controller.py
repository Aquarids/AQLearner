from fl.controller import FLController
from fla.defend.robust_aggr.robust_aggr_server import RobustAggrServer, type_aggr_median, type_aggr_trimmed_mean

from tqdm import tqdm

class MedianAggrFLController(FLController):
    def __init__(self, server: RobustAggrServer, clients):
        super().__init__(server, clients)

    def aggregate_grads(self, grads):
        self.server.aggretate_gradients(grads, type_aggr_median)

class TrimmedMeanAggrFLController(FLController):
    def __init__(self, server: RobustAggrServer, clients, trim_ratio=0.1):
        super().__init__(server, clients)
        self.trim_ratio = trim_ratio

    def aggregate_grads(self, grads):
        self.server.aggretate_gradients(grads, type_aggr_trimmed_mean, trim_ratio=self.trim_ratio)