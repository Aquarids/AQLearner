from fl.controller import FLController
from fl_security.defend.dp.dp_server import OutputPerturbationServer
from fl_security.defend.dp.dp_client import InputPerturbationClient
from fl_security.defend.dp.dp_client import DPSGDClient


class OutputPerturbationFLController(FLController):

    def __init__(self, server: OutputPerturbationServer, clients, epsilon=0.1):
        super().__init__(server, clients)
        self.epsilon = epsilon

    def predict(self, loader):
        return self.server.predict(loader, epsilon=self.epsilon)


class InputPerturbationFLController(FLController):

    def __init__(self, server, clients: list[InputPerturbationClient]):
        super().__init__(server, clients)


class DPSGDFLController(FLController):

    def __init__(self, server, clients: list[DPSGDClient], sigma, clip_value,
                 delta):
        super().__init__(server, clients)
        self.sigma = sigma
        self.clip_value = clip_value
        self.delta = delta

    def client_train(self, client: DPSGDClient, round_idx):
        client.update_model(self.server.model.state_dict())
        client.train(round_idx,
                     sigma=self.sigma,
                     clip_value=self.clip_value,
                     delta=self.delta)
