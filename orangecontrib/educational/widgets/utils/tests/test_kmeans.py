import unittest
from Orange.data import Table, Domain
from orangecontrib.educational.widgets.utils.kmeans import Kmeans
import numpy as np


class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.data = Table('iris')
        new_domain = Domain(self.data.domain.attributes[:2])
        self.data = Table.from_table(new_domain, self.data)
        # self.centroids = [[5.2, 3.1], [6.5, 3], [7, 4]]
        self.kmeans = Kmeans(self.data)

    def test_k(self):
        centroids = [[5.2, 3.1], [6.5, 3], [7, 4]]
        self.kmeans.add_centroids(centroids)
        self.assertEqual(self.kmeans.k, 3)
        self.assertEqual(self.kmeans.k, len(self.kmeans.centroids))

    def test_converged(self):
        centroids = [[5.2, 3.1], [6.5, 3], [7, 4]]
        self.kmeans.add_centroids(centroids)
        self.assertFalse(self.kmeans.converged)

        self.kmeans.step()
        self.assertFalse(self.kmeans.converged)
        # step not complete so false every odd state

        self.kmeans.step()
        # it is even step so maybe converged but it depends on example
        # unable to test
        self.kmeans.step()
        self.assertFalse(self.kmeans.converged)

        # check if false every not completed step
        for i in range(self.kmeans.max_iter // 2 + 1):
            self.kmeans.step()
            self.kmeans.step()
            self.assertFalse(self.kmeans.converged)

        # it converged because of max iter
        self.kmeans.step()
        self.assertTrue(self.kmeans.converged)

    def test_centroids_belonging_points(self):
        centroids = [[5.2, 3.6]]
        self.kmeans.add_centroids(centroids)

        # if only one cluster all points in 0th element of first dimension
        np.testing.assert_equal(
            self.kmeans.centroids_belonging_points, np.array([self.data.X]))

        # try with two clusters and less data
        self.kmeans.set_data(self.data[:3])
        self.kmeans.add_centroids([[4.7, 3.0]])
        desired_array = np.array([np.array([[5.100, 3.500]]),
                                  np.array([[4.900, 3.000],
                                            [4.700, 3.200]])])
        for i, arr in enumerate(self.kmeans.centroids_belonging_points):
            np.testing.assert_equal(arr, desired_array[i])

    def test_step_completed(self):
        centroids = [[5.2, 3.1], [6.5, 3], [7, 4]]
        self.kmeans.add_centroids(centroids)
        self.assertEqual(self.kmeans.step_completed, True)
        self.kmeans.step()
        self.assertEqual(self.kmeans.step_completed, False)
        self.kmeans.step()
        self.assertEqual(self.kmeans.step_completed, True)

    def test_set_data(self):
        self.kmeans.set_data(self.data)
        self.assertEqual(self.kmeans.data, self.data)
        self.assertEqual(self.kmeans.centroids_history[0].size, 0)
        self.assertEqual(self.kmeans.step_no, 0)
        self.assertEqual(self.kmeans.step_completed, True)
        self.assertEqual(self.kmeans.centroids_moved, False)

        # try with none data
        self.kmeans.set_data(None)
        self.assertEqual(self.kmeans.data, None)
        self.assertEqual(self.kmeans.centroids_history[0].size, 0)
        self.assertEqual(self.kmeans.clusters, None)
        self.assertEqual(self.kmeans.step_no, 0)
        self.assertEqual(self.kmeans.step_completed, True)
        self.assertEqual(self.kmeans.centroids_moved, False)

    def test_find_clusters(self):
        self.kmeans.add_centroids([[5.2, 3.6]])

        # if only one cluster all points in 0th element of first dimension
        np.testing.assert_equal(
            self.kmeans.find_clusters(self.kmeans.centroids),
            np.zeros(len(self.data)))

        # try with two clusters and less data
        self.kmeans.set_data(self.data[:3])
        self.kmeans.add_centroids([[4.7, 3.0]])
        np.testing.assert_equal(
            self.kmeans.find_clusters(self.kmeans.centroids),
            np.array([0, 1, 1]))

    def test_step(self):
        centroids = [[5.2, 3.1], [6.5, 3], [7, 4]]
        self.kmeans.add_centroids(centroids)
        centroids_before = np.copy(self.kmeans.centroids)
        clusters_before = np.copy(self.kmeans.clusters)
        self.kmeans.step()
        self.assertEqual(self.kmeans.step_completed, False)
        self.assertEqual(self.kmeans.centroids_moved, True)
        np.testing.assert_equal(centroids_before,
                                self.kmeans.centroids_history[-2])
        # clusters doesnt change in every odd step
        np.testing.assert_equal(clusters_before, self.kmeans.clusters)

        centroids_before = np.copy(self.kmeans.centroids)
        self.kmeans.step()
        self.assertEqual(self.kmeans.step_completed, True)
        self.assertEqual(self.kmeans.centroids_moved, False)
        np.testing.assert_equal(centroids_before, self.kmeans.centroids)

        centroids_before = np.copy(self.kmeans.centroids)
        self.kmeans.step()
        self.kmeans.step_back()
        np.testing.assert_equal(centroids_before, self.kmeans.centroids)

    def test_step_back(self):
        centroids = [[5.2, 3.1], [6.5, 3], [7, 4]]
        self.kmeans.add_centroids(centroids)

        # check if nothing happens when step = 0
        centroids_before = np.copy(self.kmeans.centroids)
        clusters_before = np.copy(self.kmeans.clusters)
        self.kmeans.step_back()
        np.testing.assert_equal(centroids_before, self.kmeans.centroids)
        np.testing.assert_equal(clusters_before, self.kmeans.clusters)

        # check if centroids remain in even step
        self.kmeans.step()
        self.kmeans.step()

        centroids_before = self.kmeans.centroids
        self.kmeans.step_back()
        np.testing.assert_equal(centroids_before, self.kmeans.centroids)
        self.assertEqual(self.kmeans.step_completed, False)
        self.assertEqual(self.kmeans.centroids_moved, False)

        # check if clusters remain in even step
        clusters_before = self.kmeans.clusters
        self.kmeans.step_back()
        np.testing.assert_equal(clusters_before, self.kmeans.clusters)
        self.assertEqual(self.kmeans.step_completed, True)
        self.assertEqual(self.kmeans.centroids_moved, True)

    def test_random_positioning(self):
        self.assertEqual(self.kmeans.random_positioning(4).shape, (4, 2))
        self.assertEqual(self.kmeans.random_positioning(1).shape, (1, 2))
        self.assertEqual(self.kmeans.random_positioning(0).shape, (0,))
        self.assertEqual(self.kmeans.random_positioning(-1).shape, (0,))

    def test_add_centroids(self):
        self.kmeans.add_centroids([[5.2, 3.1]])
        self.assertEqual(self.kmeans.k, 1)
        self.kmeans.add_centroids([[6.5, 3], [7, 4]])
        self.assertEqual(self.kmeans.k, 3)
        self.kmeans.add_centroids(2)
        self.assertEqual(self.kmeans.k, 5)
        self.kmeans.add_centroids()
        self.assertEqual(self.kmeans.k, 6)

        step_before = self.kmeans.step_no
        self.assertEqual(step_before, self.kmeans.step_no)
        self.kmeans.step()
        self.kmeans.add_centroids()
        self.assertEqual(step_before + 2, self.kmeans.step_no)
        self.assertEqual(self.kmeans.centroids_moved, False)

    def test_delete_centroids(self):
        self.kmeans.add_centroids([[6.5, 3], [7, 4], [5.2, 3.1]])
        self.kmeans.delete_centroids(1)
        self.assertEqual(self.kmeans.k, 2)
        self.kmeans.delete_centroids(2)
        self.assertEqual(self.kmeans.k, 0)
        self.kmeans.delete_centroids(2)
        self.assertEqual(self.kmeans.k, 0)

    def test_move_centroid(self):
        self.kmeans.add_centroids([[6.5, 3], [7, 4], [5.2, 3.1]])
        self.kmeans.move_centroid(1, 3, 3.2)
        np.testing.assert_equal(self.kmeans.centroids[1], np.array([3, 3.2]))
        self.assertEqual(self.kmeans.k, 3)

        self.kmeans.step()
        self.kmeans.move_centroid(2, 3.2, 3.4)
        self.assertEqual(self.kmeans.centroids_moved, False)
        self.assertEqual(self.kmeans.step_no, 2)

    def test_set_list(self):
        # adding to end
        self.assertEqual(self.kmeans.set_list([2, 1], 2, 1), [2, 1, 1])
        # changing the element in the last place
        self.assertEqual(self.kmeans.set_list([2, 1], 1, 3), [2, 3])
        # changing the element in the middle place
        self.assertEqual(self.kmeans.set_list([2, 1, 3], 1, 3), [2, 3, 3])
