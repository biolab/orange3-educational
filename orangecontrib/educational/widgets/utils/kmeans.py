import Orange
import numpy as np
from Orange.distance import Euclidean
from random import sample


class Kmeans:

    def __init__(self, data, centroids=None, distance_metric=Orange.distance.Euclidean):
        self.k = len(centroids) if centroids is not None else 0
        self.data = data
        self.centroids = np.array(centroids) if centroids is not None else np.empty((0, 2))
        self.centroids_before = None
        self.centroids_belonging_points = [None] * self.k
        self.distance_metric = distance_metric
        # if len od data is less than number of centroids
        if len(self.data) < len(self.centroids):
            self.centroids = self.centroids[:len(self.data)]

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
        else:
            self.centroids = np.vstack((self.centroids, self.random_positioning()))
        self.k = len(self.centroids)
        self.centroids_belonging_points = self.centroids_belonging_points + \
            [None] * (self.k - len(self.centroids_belonging_points))
        self.set_centroid_belonging()

    def delete_centroids(self):
        self.centroids = self.centroids[:-1]
        self.k = len(self.centroids)
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


# if __name__ == "__main__":
    # domain = Domain([ContinuousVariable("x"), ContinuousVariable("y")])
    # table = Table(domain, np.array([[1, 1], [2, 2], [1, 2], [1.5, 1.5],
    #                                 [5, 5], [5.5, 5], [4.9, 4.8], [5.5, 5.5],
    #                                 [1, 6], [0, 6], [0.5, 6], [0.5, 7]]))
    #
    #
    # kmeans = Kmeans(table, 3)
    # plt.plot(kmeans.data.X[:,0], kmeans.data.X[:,1], "ro")
    # plt.plot(kmeans.centroids[:,0], kmeans.centroids[:,1], "go")
    # plt.savefig("kmeans.png")
    # plt.clf()
    #
    # for i in range(5):
    #     kmeans.step()
    #     plt.plot(kmeans.data.X[:,0], kmeans.data.X[:,1], "ro")
    #     plt.plot(kmeans.centroids[:,0], kmeans.centroids[:,1], "go")
    #     plt.savefig("kmeans%d.png" % i)
    #     plt.clf()