import numpy as np
import torch

from ml.linear.knn import KNN
from ml.cluster.db_scan import DBScan

label_anomaly = -1


class AnomalyDetection:

    def __init__(self, k, min_samples=2):
        self.k = k
        self.dbscan = DBScan(min_samples=min_samples,
                             noise_label=label_anomaly)

    def extract_features(self, clients_data):
        aggregated_features = []
        for data in clients_data:
            features = []
            if isinstance(data, dict):
                # for weights
                data = data.values()

            for tensor in data:
                if tensor is not None and tensor.numel() > 0:
                    flattened = tensor.detach().cpu().numpy().flatten()
                    features.extend([
                        np.mean(flattened),
                        np.std(flattened),
                        np.min(flattened),
                        np.max(flattened),
                        np.percentile(flattened, 25),
                        np.percentile(flattened, 50),
                        np.percentile(flattened, 75)
                    ])
            if features:
                aggregated_features.append(features)
        return np.array(aggregated_features)

    def estimate_eps(self, features):

        dist_matrix = np.sqrt(((features[:, np.newaxis] -
                                features[np.newaxis, :])**2).sum(axis=2))
        sorted_dists = np.sort(dist_matrix, axis=1)
        kth_distances = sorted_dists[:, self.k]
        sorted_kth_distances = np.sort(kth_distances)

        # Determine the 'elbow' point
        gradients = np.diff(sorted_kth_distances)
        max_curvature_index = np.argmax(np.diff(gradients))

        estimated_eps = sorted_kth_distances[max_curvature_index]
        return estimated_eps

    def detect_anomalies(self, clients_data):
        features_matrix = self.extract_features(clients_data)
        if features_matrix.size == 0:
            return []
        self.dbscan.eps = self.estimate_eps(features_matrix)
        self.dbscan.fit(features_matrix)
        anomalies_index = self.dbscan.get_noise_index()
        print(f"Detect anomalies index: {anomalies_index}")
        return anomalies_index

    def detect_grads(self, clients_grads):
        return self.detect_anomalies(clients_grads)

    def detect_weights(self, clients_weights):
        return self.detect_anomalies(clients_weights)
