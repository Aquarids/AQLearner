import numpy as np
import learn_math


class KMeans:

    def __init__(self, n_clusters, n_iter=300):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.centroids = None
        self.labels = None

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0],
                                            self.n_clusters,
                                            replace=False)]
        for _ in range(self.n_iter):
            clusters = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, clusters)
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids
        self.labels = clusters

    def _assign_clusters(self, X):
        clusters = []
        for x in X:
            distances = np.linalg.norm(x - self.centroids, axis=1)
            clusters.append(np.argmin(distances))
        return np.array(clusters)

    def _update_centroids(self, X, clusters):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster = X[clusters == i]
            new_centroids.append(np.mean(cluster, axis=0))
        return np.array(new_centroids)

    def get_labels(self):
        return self.labels
