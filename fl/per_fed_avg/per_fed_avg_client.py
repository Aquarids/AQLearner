from fl.client import Client

import torch


class PerFedAvgClient(Client):

    def __init__(self, model: torch.nn.Module, criterion, optimizer, type):
        super().__init__(model, criterion, optimizer, type)

    def personalize(self):
        # reuse the frain method for personalization
        self.train(-1)
