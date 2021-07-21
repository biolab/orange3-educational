import unittest

from orangecontrib.educational.widgets.utils.kmeans import Kmeans
import numpy as np


class TestKmeans(unittest.TestCase):
    def setUp(self):
        self.data = np.arange(20).reshape(10, 2)
        self.kmeans = Kmeans(self.data)

    def test_converged(self):
        """It doesn't converge in a half-step, but does eventually"""
        self.kmeans.move_centroid(0, 0, 1)
        self.kmeans.move_centroid(1, 2, 3)
        self.kmeans.move_centroid(2, 4, 5)
        i = -1
        for i in range(50):
            self.kmeans.move_centroids()
            self.kmeans.assign_membership()
            if self.kmeans.converged:
                break
        self.assertGreater(i, 1)
        self.assertTrue(self.kmeans.converged)

    def test_reset(self):
        assert self.kmeans.k == len(self.kmeans.centroids) == 3
        centroids = self.kmeans.centroids.copy()
        self.kmeans.reset(self.data[:9])
        np.testing.assert_equal(self.kmeans.data, self.data[:9])
        self.assertFalse(np.all(centroids == self.kmeans.centroids.copy()))

        self.kmeans.reset(self.data, 5)
        np.testing.assert_equal(self.kmeans.data, self.data)
        self.assertEqual(self.kmeans.k, 5)

    def test_find_clusters(self):
        # if there is only one cluster, all points belong to it
        self.kmeans.reset(self.data, 1)
        np.testing.assert_equal(self.kmeans._find_clusters(), 0)

        # Have two centroids that split the data
        self.kmeans.reset(self.data, 2)
        self.kmeans.move_centroid(1, 4, 5)
        self.kmeans.move_centroid(0, 6, 7)
        np.testing.assert_equal(self.kmeans._find_clusters(), [1] * 3 + [0] * 7)

    def test_steps(self):
        self.kmeans.reset(self.data, 2)
        self.kmeans.centroids = np.array([[6, 7], [4, 5]])
        self.kmeans.assign_membership()
        np.testing.assert_equal(self.kmeans.clusters, [1] * 3 + [0] * 7)

        self.kmeans.move_centroids()
        np.testing.assert_equal(self.kmeans.centroids, [[12, 13], [2, 3]])

    def test_history(self):
        assert not self.kmeans.history

        centroids = self.kmeans.centroids.copy()
        self.kmeans.move_centroids()
        self.assertIs(self.kmeans.history[0].step, Kmeans.move_centroids)

        clusters = self.kmeans.clusters.copy()
        self.kmeans.assign_membership()
        self.assertIs(self.kmeans.history[1].step, Kmeans.assign_membership)

        self.assertEqual(len(self.kmeans.history), 2)

        self.kmeans.step_back()
        self.assertEqual(len(self.kmeans.history), 1)
        self.assertIs(self.kmeans.history[0].step, Kmeans.move_centroids)
        np.testing.assert_equal(self.kmeans.clusters, clusters)

        self.kmeans.step_back()
        self.assertEqual(len(self.kmeans.history), 0)
        np.testing.assert_equal(self.kmeans.centroids, centroids)

    def test_add_centroid(self):
        centroids = self.kmeans.centroids.tolist()
        assert len(centroids) == 3
        self.kmeans.add_centroid(4, 5)
        self.assertEqual(self.kmeans.k, 4)
        self.assertEqual(self.kmeans.centroids.tolist(), centroids + [[4, 5]])
        self.assertIs(self.kmeans.history[-1].step, Kmeans.add_centroid)

        self.kmeans.step_back()
        self.assertEqual(self.kmeans.centroids.tolist(), centroids)

    def test_delete_centroid(self):
        centroids = self.kmeans.centroids.tolist()
        assert len(centroids) == 3
        self.kmeans.delete_centroid(1)
        self.assertEqual(self.kmeans.k, 2)
        self.assertEqual(self.kmeans.centroids.tolist(),
                         centroids[:1] + centroids[2:])
        self.assertIs(self.kmeans.history[-1].step, Kmeans.delete_centroid)

        self.kmeans.step_back()
        self.assertEqual(self.kmeans.centroids.tolist(), centroids)

    def test_move_centroid(self):
        c0, c1, c2 = self.kmeans.centroids.tolist()
        cn = [3, 3.2]
        self.kmeans.move_centroid(1, *cn)
        self.assertEqual(self.kmeans.centroids.tolist(), [c0, cn, c2])
        self.assertEqual(self.kmeans.k, 3)
        self.assertIs(self.kmeans.history[-1].step, Kmeans.move_centroid)

        self.kmeans.step_back()
        self.assertEqual(self.kmeans.centroids.tolist(), [c0, c1, c2])


if __name__ == "__main__":
    unittest.main()
