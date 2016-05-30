import Orange
import numpy as np
from Orange.distance import Euclidean


class Kmeans:

    def __init__(self, data, centroids=None, distance_metric=Orange.distance.Euclidean):
        self.data = data
        self.centroids = np.array(centroids) if centroids is not None else np.empty((0, 2))
        self.centroids_before = None
        self.centroids_belonging_points = [None] * self.k
        self.distance_metric = distance_metric

        # if len od data is less than number of centroids take only len(data) centroids
        if len(self.data) < self.k:
            self.centroids = self.centroids[:len(self.data)]

    @property
    def k(self):
        return len(self.centroids)

    def step(self):
        self.centroids_before = np.copy(self.centroids)
        d = self.data.X
        dist = self.distance_metric(d, self.centroids)
        closest_centroid = np.argmin(dist, axis=1)
        for i in range(len(self.centroids)):
            c_points = d[closest_centroid == i]
            self.centroids_belonging_points[i] = c_points
            self.centroids[i, :] = np.average(c_points, axis=0)
        # delete centroids that do not belong to any point
        self.centroids = self.centroids[~np.isnan(self.centroids).any(axis=1)]

    def random_positioning(self):
        idx = np.random.choice(len(self.data), np.random.randint(1, np.min((5, len(self.data)))))
        return np.mean(self.data.X[idx], axis=0)

    def add_centroids(self, points=None):
        if points is not None:
            self.centroids = np.vstack((self.centroids, np.array(points)))
        else:  # if no point provided add one centroid
            self.centroids = np.vstack((self.centroids, self.random_positioning()))
        self.centroids_belonging_points = self.centroids_belonging_points + \
            [None] * (self.k - len(self.centroids_belonging_points))
        self.set_centroid_belonging()

    def delete_centroids(self):
        self.centroids = self.centroids[:-1]
        self.centroids_belonging_points = self.centroids_belonging_points[:-1]
        self.set_centroid_belonging()

    def move_centroid(self, _index, x, y):
        self.centroids[_index, :] = np.array([x, y])
        self.set_centroid_belonging()

    def set_centroid_belonging(self):
        d = self.data.X
        dist = self.distance_metric(d, self.centroids)
        closest_centroid = np.argmin(dist, axis=1)
        for i in range(len(self.centroids)):
            c_points = d[closest_centroid == i]
            self.centroids_belonging_points[i] = c_points
