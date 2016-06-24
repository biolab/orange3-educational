import Orange
import numpy as np
from Orange.distance import Euclidean


class Kmeans:
    """
    K-Means algorithm

    Parameters
    ----------
    data: Orange.data.Table
        Data used for k-means
    centroids: list or numpy.array
        List of centroids
    distance_metric: Orange.distance
        Distance used to measure distance to point in k-means
    """

    max_iter = 100
    threshold = 1e-3

    def __init__(self, data, centroids=None, distance_metric=Orange.distance.Euclidean):
        self.data = data
        self.centroids = np.array(centroids) if centroids is not None else np.empty((0, 2))
        self.distance_metric = distance_metric
        self.stepNo = 0
        self.clusters = self.find_clusters(self.centroids)
        self.centroids_history = []
        self.centroids_moved = False

    @property
    def k(self):
        """
        :return: Number of clusters
        :type: int
        """
        return len(self.centroids) if self.centroids is not None else 0

    @property
    def centroids_belonging_points(self):
        """
        :return: List of lists that contains each clusters points
        :type: list of numpy.arrays
        """
        d = self.data.X
        return [d[ self.clusters == i] for i in range(len(self.centroids))]

    @property
    def converged(self):
        """
        :return: True if k-means converged else False.
        :type: boolean
        """
        if len(self.centroids_history) == 0 or len(self.centroids) != len(self.centroids_history[-1]):
            return False
        dist = (np.sum(np.sqrt(np.sum(np.power((self.centroids - self.centroids_history[-1]), 2), axis=1))) /
                len(self.centroids))
        return dist < self.threshold or self.stepNo > self.max_iter

    @property
    def step_completed(self):
        """
        :return: True if booth phases of step (centroids moved and points assigned to new centroids)
        :type: boolean
        """
        return self.stepNo % 2 == 0

    def set_data(self, data):
        """
        Function called when data changed on imput
        :param data: Data used for k-means
        :type data: Orange.data.Table or None
        """
        if data is not None and len(data) > 0:
            self.data = data
            self.clusters = self.find_clusters(self.centroids)

            # with different data it does not make sense to have history, algorithm from begining
            self.centroids_history = []
            self.stepNo = 0
        else:
            self.data = None
            self.clusters = None
            self.centroids_history = []
            self.stepNo = 0
        self.centroids_moved = False

    def find_clusters(self, centroids):
        """
        Function calculates new clusters to data points
        :param centroids: Centroids
        :type centroids: numpy.array
        :return: Clusters indices for every data point
        :type: numpy.array
        """
        if self.k > 0:
            d = self.data.X
            dist = self.distance_metric(d, centroids)
            return np.argmin(dist, axis=1)
        else:
            return None

    def step(self):
        """
        Half of the step of k-means
        """
        if self.step_completed:
            if len(self.centroids_history) < self.stepNo // 2 + 1:
                self.centroids_history.append(np.copy(self.centroids))
            else:
                self.centroids_history[self.stepNo // 2] = np.copy(self.centroids)
            d = self.data.X
            points = [d[self.clusters == i] for i in range(len(self.centroids))]
            for i in range(len(self.centroids)):
                c_points = points[i]
                self.centroids[i, :] = np.average(c_points, axis=0) if len(c_points) > 0 else np.nan
            # reinitialize empty centroids

            nan_c = np.isnan(self.centroids).any(axis=1)
            if np.count_nonzero(nan_c) > 0:
                self.centroids[nan_c] = self.random_positioning(np.count_nonzero(nan_c))
            self.centroids_moved = True
        else:
            self.clusters = self.find_clusters(self.centroids)
            self.centroids_moved = False
        self.stepNo += 1

    def step_back(self):
        """
        Half of the step back of k-means
        :return:
        """
        if self.stepNo > 0:
            if not self.step_completed:
                self.centroids = self.centroids_history[self.stepNo // 2]
                self.centroids_moved = True
            else:
                self.clusters = self.find_clusters(self.centroids_history[self.stepNo // 2 - 1])
                self.centroids_moved = False
            self.stepNo -= 1

    def random_positioning(self, no_centroids):
        """
        Calculates new centroid using random positioning
        :param no_centroids: number of centroids to calculate
        :type no_centroids: int
        :return: new centroid
        :type: np.array
        """
        if no_centroids <= 0:
            return np.array([])
        centroids = np.empty((no_centroids, 2))
        for i in range(no_centroids):
            idx = np.random.choice(len(self.data), np.random.randint(1, np.min((5, len(self.data) + 1))))
            centroids[i, :] = np.mean(self.data.X[idx], axis=0)
        return centroids

    def add_centroids(self, points=None):
        """
        Add new centroid/s. Using points if provided else random positioning
        :param points: Centroids or number of them
        :type: list or numpy.array or int
        """
        if points is None:  # if no point provided add one centroid
            self.centroids = np.vstack((self.centroids, self.random_positioning(1)))
        elif isinstance(points, int):  # if int provided add as much of them
            self.centroids = np.vstack((self.centroids, self.random_positioning(points)))
        else:   # else it is array of new centroids
            self.centroids = np.vstack((self.centroids, np.array(points)))
        self.clusters = self.find_clusters(self.centroids)
        self.centroids_moved = False
        if not self.step_completed:
            self.stepNo += 1

    def delete_centroids(self, num):
        """
        Remove last centroid
        :param num: number of deleted centroids
        :type num: int
        """
        self.centroids = self.centroids[:(-num if num <= len(self.centroids) else len(self.centroids))]
        self.clusters = self.find_clusters(self.centroids)

    def move_centroid(self, _index, x, y):
        """
        Move centroid with index to position x, y
        :param _index: centroid index
        :param x: x position
        :type x: float
        :param y: y position
        :type y: float
        """
        self.centroids[_index, :] = np.array([x, y])
        self.centroids_moved = False
        self.clusters = self.find_clusters(self.centroids)
        if not self.step_completed:
            self.stepNo += 1
