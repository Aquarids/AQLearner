from fl.controller import FLController


class MaliciousFLController(FLController):

    def __init__(self, server, clients):
        super().__init__(server, clients)
        self.leaked_grads = []
        self.leaked_weights = []

    def aggregate_grads(self, grads):
        self.leaked_grads.append(grads)
        return super().aggregate_grads(grads)

    def aggregate_weights(self, weights):
        self.leaked_weights.append(weights)
        return super().aggregate_weights(weights)

    def get_leaked_grads(self):
        return self.leaked_grads

    def get_leaked_weights(self):
        return self.leaked_weights
