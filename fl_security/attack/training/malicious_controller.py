from fl.controller import FLController


class MaliciousFLController(FLController):

    def __init__(self, server, clients):
        super().__init__(server, clients)
