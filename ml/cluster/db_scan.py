import numpy as np
import learn_math


class DBScan:

    def __init__(self, eps=0.5, min_samples=5, noise_label=-1):
        self.eps = eps
        self.min_samples = min_samples
        self.noise_label = noise_label
        self.labels = None
        self.visited = None
        self.X = None

    def fit(self, X):
        self.X = X
        self.labels = np.full(X.shape[0], -1)
        self.visited = np.full(X.shape[0], False)
        cluster = 0
        for i in range(X.shape[0]):
            if self.visited[i]:
                continue
            self.visited[i] = True
            neighbors = self._get_neighbors(i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = self.noise_label
            else:
                cluster += 1
                self._expand_cluster(i, neighbors, cluster)

    def _get_neighbors(self, i):
        neighbors = []
        for j in range(self.X.shape[0]):
            if np.linalg.norm(self.X[i] - self.X[j]) < self.eps:
                neighbors.append(j)
        return neighbors

    def _expand_cluster(self, i, neighbors, cluster):
        self.labels[i] = cluster
        for j in neighbors:
            if not self.visited[j]:
                self.visited[j] = True
                new_neighbors = self._get_neighbors(j)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            if self.labels[j] == -1:
                self.labels[j] = cluster

    def get_labels(self):
        return self.labels

    def get_noise_index(self):
        return np.where(self.labels == self.noise_label)[0]
