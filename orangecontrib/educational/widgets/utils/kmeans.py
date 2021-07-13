from functools import wraps
from types import FunctionType
from typing import NamedTuple, List

import numpy as np

import Orange


HistoryEntry = NamedTuple("HistoryEntry", (("step", FunctionType),
                                           ("centroids", np.ndarray),
                                           ("clusters", np.ndarray)))


def historic(reassign=True):
    def decorator(f):
        @wraps(f)
        def historian(self, *args, **kwargs):
            # store decorated function (not `f`) to enable comparisons with
            # Kmeans.<whatever>
            self._store_history(historian)
            f(self, *args, **kwargs)
            if reassign:
                self.clusters = self._find_clusters()
        return historian
    return decorator if isinstance(reassign, bool) else decorator(reassign)


class Kmeans:
    """
    K-Means algorithm

    Parameters
    ----------
    data (np.ndarray): two-column d ata used for k-means
    centroids (int, list or numpy.array or int): number of coordinates of centroids
    """
    def __init__(self, data, centroids=3):
        self.data = None
        self.centroids = None
        self.history: List[HistoryEntry] = []
        self.clusters = None
        self.reset(data, centroids)

    @property
    def k(self):
        return len(self.centroids)

    @property
    def waits_reassignment(self):
        return self.history and self.history[-1].step == Kmeans.move_centroids

    @property
    def converged(self):
        """
        Clustering converged if the last three steps were assignment, moving,
        and assignment, with membership assignments being the same.
        """
        if len(self.history) < 3:
            return False
        a, b, c = self.history[-3:]
        return a.step == c.step == Kmeans.assign_membership \
               and b.step == Kmeans.move_centroids \
               and np.all(a.clusters == c.clusters)

    def _find_clusters(self):
        dist = Orange.distance.Euclidean(self.data, self.centroids)
        return np.argmin(dist, axis=1)

    def _store_history(self, step):
        self.history.append(
            HistoryEntry(step, np.copy(self.centroids), np.copy(self.clusters)))

    def step_back(self):
        if self.history:
            _, self.centroids, self.clusters = self.history.pop()

    def reset(self, data, centroids=3):
        self.data = data
        self.centroids = np.array(
                centroids if not isinstance(centroids, int)
                else [self._random_position() for _ in range(centroids)])
        self.clusters = self._find_clusters()
        self.history = []

    @historic
    def assign_membership(self):
        pass

    @historic(reassign=False)
    def move_centroids(self):
        for i in range(self.k):
            points = self.data[self.clusters == i]
            self.centroids[i, :] = np.average(points, axis=0) if points.size \
                                   else self._random_position()

    @historic
    def add_centroid(self, x=None, y=None):
        assert (x is None) == (y is None)
        new = self._random_position() if x is None else np.array([x, y])
        self.centroids = np.vstack((self.centroids, new))

    @historic
    def delete_centroid(self, num):
        self.centroids = np.vstack((self.centroids[:num],
                                    self.centroids[num + 1:]))

    @historic
    def move_centroid(self, _index, x, y):
        self.centroids[_index, :] = np.array([x, y])

    def _random_position(self):
        n = len(self.data)
        sample = np.random.choice(n, np.min((5, n)))
        return np.mean(self.data[sample], axis=0)
