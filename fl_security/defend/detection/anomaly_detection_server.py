import torch

from fl.server import Server
from fl_security.defend.detection.anomaly_detection import AnomalyDetection


class AnomalyDetectionServer(Server):

    def __init__(self, model: torch.nn.Module, optimizer, criterion, type, k,
                 min_samples):
        super().__init__(model, optimizer, criterion, type)
        self.anomaly_detection = AnomalyDetection(k=k, min_samples=min_samples)

    def calculate_gradients(self, grads):
        anomaly_index = self.anomaly_detection.detect_grads(grads)
        normal_grads = [
            grads[i] for i in range(len(grads)) if i not in anomaly_index
        ]
        return super().calculate_gradients(normal_grads)

    def calculate_weights(self, weights):
        anomaly_index = self.anomaly_detection.detect_weights(weights)
        normal_weights = [
            weights[i] for i in range(len(weights)) if i not in anomaly_index
        ]
        return super().calculate_weights(normal_weights)
