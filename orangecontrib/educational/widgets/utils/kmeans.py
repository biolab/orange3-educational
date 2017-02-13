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
    threshold = 1e-1

    def __init__(self, data,
                 centroids=None, distance_metric=Orange.distance.Euclidean):
        self.data = data
        self.centroids = (np.array(centroids)
                          if centroids is not None else np.empty((0, 2)))
        self.distance_metric = distance_metric
        self.step_no = 0
        self.clusters = self.find_clusters(self.centroids)
        self.centroids_history = [np.copy(self.centroids)]
        self.centroids_moved = False

    @property
    def k(self):
        return len(self.centroids) if self.centroids is not None else 0

    @property
    def centroids_belonging_points(self):
        d = self.data.X
        return [d[self.clusters == i] for i in range(len(self.centroids))]

    @property
    def converged(self):
        """
        Function check if algorithm already converged
        """
        if len(self.centroids_history) <= 1 or \
                (len(self.centroids) != len(self.centroids_history[self.step_no - 2])) or \
                not self.step_completed:
            return False
        dist = (np.sum(
            np.sqrt(np.sum(
                np.power(
                    (self.centroids - self.centroids_history[self.step_no - 2]),
                    2),
                axis=1))) / len(self.centroids))

        return dist < self.threshold or self.step_no > self.max_iter

    @property
    def step_completed(self):
        return self.step_no % 2 == 0

    def set_data(self, data):
        """
        Function called when data changed on input

        Parameters
        ----------
        data : Orange.data.Table or None
            Data used for k-means
        """
        self.__init__(data, self.centroids, distance_metric=self.distance_metric)

    def find_clusters(self, centroids):
        """
        Function calculates new clusters to data points
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
            d = self.data.X
            points = [d[self.clusters == i] for i in range(len(self.centroids))]
            for i in range(len(self.centroids)):
                c_points = points[i]
                self.centroids[i, :] = (np.average(c_points, axis=0)
                                        if len(c_points) > 0 else np.nan)
            # reinitialize empty centroids

            nan_c = np.isnan(self.centroids).any(axis=1)
            if np.count_nonzero(nan_c) > 0:
                self.centroids[nan_c] = self.random_positioning(
                    np.count_nonzero(nan_c))
            self.centroids_moved = True
        else:
            self.clusters = self.find_clusters(self.centroids)
            self.centroids_moved = False
        self.step_no += 1
        self.centroids_history = self.set_list(
            self.centroids_history, self.step_no, np.copy(self.centroids))

    def step_back(self):
        """
        Half of the step back of k-means
        """
        if self.step_no > 0:
            if not self.step_completed:
                self.centroids = np.copy(
                    self.centroids_history[self.step_no - 1])
                self.centroids_moved = True
            else:
                self.centroids = np.copy(
                    self.centroids_history[self.step_no - 1])
                self.clusters = self.find_clusters(
                    self.centroids_history[self.step_no - 2])
                self.centroids_moved = False
            self.step_no -= 1

    def random_positioning(self, no_centroids):
        """
        Calculates new centroid using random positioning

        Parameters
        ----------
        no_centroids : int
            number of centroids to calculate

        Returns
        -------
        np.array
            new centroid
        """
        if no_centroids <= 0:
            return np.array([])
        centroids = np.empty((no_centroids, 2))
        for i in range(no_centroids):
            idx = np.random.choice(
                len(self.data), np.random.randint(
                    1, np.min((5, len(self.data) + 1))))
            centroids[i, :] = np.mean(self.data.X[idx], axis=0)
        return centroids

    def recompute_clusters(self):
        """
        Function recomputes belonging points to centroid and increases step_no
        """
        self.clusters = self.find_clusters(self.centroids)
        self.centroids_moved = False
        if not self.step_completed:
            self.step_no += 1
        self.centroids_history = self.set_list(
            self.centroids_history, self.step_no, np.copy(self.centroids))

    def add_centroids(self, points=None):
        """
        Add new centroid/s. Using points if provided else random positioning

        Parameters
        ----------
        points : list or numpy.array or int or None
            Centroids or number of them
        """
        if points is None:  # if no point provided add one centroid
            self.centroids = np.vstack(
                (self.centroids, self.random_positioning(1)))
        elif isinstance(points, int):  # if int provided add as much of them
            self.centroids = np.vstack(
                (self.centroids, self.random_positioning(points)))
        else:   # else it is array of new centroids
            self.centroids = np.vstack((self.centroids, np.array(points)))
        self.recompute_clusters()

    def delete_centroids(self, num):
        """
        Remove last num centroids
        """
        self.centroids = self.centroids[:(-num if num <= len(self.centroids)
                                          else len(self.centroids))]
        self.recompute_clusters()

    def move_centroid(self, _index, x, y):
        """
        Move centroid with index to position x, y
        """
        self.centroids[_index, :] = np.array([x, y])
        self.centroids_moved = False
        self.recompute_clusters()

    @staticmethod
    def set_list(l, i, v):
        try:
            l[i] = v
        except IndexError:
            assert i == len(l)
            l.append(v)
        return l
