import Orange
import numpy as np
from Orange.distance import Euclidean


class Kmeans:

    def __init__(self, data, centroids=None, distance_metric=Orange.distance.Euclidean):
        self.data = data
        self.centroids = np.array(centroids) if centroids is not None else np.empty((0, 2))
        self.distance_metric = distance_metric
        self.stepNo = 0
        self.previous_centroids_belonging_points = None

    @property
    def k(self):
        return len(self.centroids)

    @property
    def centroids_belonging_points(self):
        if self.stepNo % 2 == 1:
            return self.previous_centroids_belonging_points
        else:
            d = self.data.X
            dist = self.distance_metric(d, self.centroids)
            closest_centroid = np.argmin(dist, axis=1)
            return [d[closest_centroid == i] for i in range(len(self.centroids))]



    def step(self):
        if self.stepNo % 2 == 0:
            points = self.centroids_belonging_points
            self.previous_centroids_belonging_points = points
            for i in range(len(self.centroids)):
                c_points = points[i]
                self.centroids[i, :] = np.average(c_points, axis=0)
            # delete centroids that do not belong to any point
            self.centroids = self.centroids[~np.isnan(self.centroids).any(axis=1)]
        self.stepNo += 1

    def random_positioning(self):
        idx = np.random.choice(len(self.data), np.random.randint(1, np.min((5, len(self.data) + 1))))
        return np.mean(self.data.X[idx], axis=0)

    def add_centroids(self, points=None):
        if points is not None:
            self.centroids = np.vstack((self.centroids, np.array(points)))
        else:  # if no point provided add one centroid
            self.centroids = np.vstack((self.centroids, self.random_positioning()))

    def delete_centroids(self):
        self.centroids = self.centroids[:-1]

    def move_centroid(self, _index, x, y):
        self.centroids[_index, :] = np.array([x, y])

