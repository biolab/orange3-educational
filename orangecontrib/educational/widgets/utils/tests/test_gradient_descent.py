import unittest
from Orange.data import Table
from orangecontrib.educational.widgets.utils.gradient_descent import \
    GradientDescent
from numpy.testing import *
import numpy as np


class TestGradientDescent(unittest.TestCase):

    def setUp(self):
        self.iris = Table('iris')
        # new_domain = Domain(self.data.domain.attributes[:2])
        # self.data = Table(new_domain, self.data)
        self.gradient_descent = GradientDescent()

    def test_set_data(self):
        """
        Test set data
        """

        # check if None on beginning
        self.assertIsNone(self.gradient_descent.x, None)
        self.assertIsNone(self.gradient_descent.y, None)
        self.assertIsNone(self.gradient_descent.domain, None)

        # check if correct data are provided
        self.gradient_descent.set_data(self.iris)

        assert_array_equal(self.gradient_descent.x, self.iris.X)
        assert_array_equal(self.gradient_descent.y, self.iris.Y)
        self.assertEqual(self.gradient_descent.domain, self.iris.domain)

        # check data remove
        self.gradient_descent.set_data(None)

        self.assertIsNone(self.gradient_descent.x, None)
        self.assertIsNone(self.gradient_descent.y, None)
        self.assertIsNone(self.gradient_descent.domain, None)

    def test_set_theta(self):
        """
        Check set theta
        """
        gd = self.gradient_descent

        # theta must be none on beginning
        self.assertIsNone(gd.theta, None)

        # check if theta set correctly
        # theta from np array
        gd.set_theta(np.array([1, 2]))
        assert_array_equal(gd.theta, np.array([1, 2]))
        # history of 0 have to be equal theta
        assert_array_equal(gd.history[0][0], np.array([1, 2]))
        # step no have to reset to 0
        self.assertEqual(gd.step_no, 0)

        # theta from list
        gd.set_theta([2, 3])
        assert_array_equal(gd.theta, np.array([2, 3]))
        assert_array_equal(gd.history[0][0], np.array([2, 3]))
        self.assertEqual(gd.step_no, 0)

        # theta None
        gd.set_theta(None)
        self.assertIsNone(gd.theta)

        # theta anything else
        gd.set_theta("abc")
        self.assertIsNone(gd.theta)

    def test_set_alpha(self):
        """
        Check if alpha set correctly
        """
        gd = self.gradient_descent

        # check alpha 0.1 in the beginning
        self.assertEqual(gd.alpha, 0.1)

        # check if alpha set correctly
        gd.set_alpha(0.2)
        self.assertEqual(gd.alpha, 0.2)

        # check if alpha removed correctly
        gd.set_alpha(None)
        self.assertIsNone(gd.alpha)

    def test_model(self):
        """
        Test if model is correct
        """
        self.assertRaises(
            NotImplementedError, lambda: self.gradient_descent.model)

    def test_j(self):
        """
        Test cost function j
        """
        self.assertRaises(NotImplementedError, self.gradient_descent.j, [1, 1])

    def test_dj(self):
        """
        Test gradient function
        """
        self.assertRaises(NotImplementedError, self.gradient_descent.dj, [1, 1])

    def test_optimized(self):
        """
        Test if optimized works well
        """
        self.gradient_descent.set_data(self.iris)
        self.assertRaises(NotImplementedError, self.gradient_descent.optimized)

    def test_set_list(self):
        """
        Test set list
        """
        gd = self.gradient_descent

        # test adding Nones if list too short
        self.assertEqual(gd.set_list([], 2, 1), [None, None, 1])
        # test adding Nones if list too short
        self.assertEqual(gd.set_list([2], 2, 1), [2, None, 1])
        # adding to end
        self.assertEqual(gd.set_list([2, 1], 2, 1), [2, 1, 1])
        # changing the element in the last place
        self.assertEqual(gd.set_list([2, 1], 1, 3), [2, 3])
        # changing the element in the middle place
        self.assertEqual(gd.set_list([2, 1, 3], 1, 3), [2, 3, 3])
