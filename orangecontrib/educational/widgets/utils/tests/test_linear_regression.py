import unittest
from Orange.data import Table, Domain
from Orange.preprocess import Normalize

from orangecontrib.educational.widgets.utils.linear_regression import \
    LinearRegression
from numpy.testing import *
import numpy as np


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.housing = Normalize()(Table('housing'))
        self.linear_regression = LinearRegression()

    def test_model(self):
        """
        Test if model is correct
        """
        lr = self.linear_regression

        # test if model None when no data
        lr.set_theta([1, 2])
        self.assertIsNone(lr.model)

        # test if model None when no theta
        lr.set_theta(None)
        lr.set_data(self.housing)
        self.assertIsNone(lr.model)

        # test if model None when no theta and no Data
        lr.set_data(None)
        self.assertIsNone(lr.model)

        theta = np.ones(len(self.housing.domain.attributes))
        # test when model is not none
        lr.set_data(self.housing)
        lr.set_theta(theta)
        model = lr.model

        # test parameters are ok
        self.assertIsNotNone(model)
        assert_array_equal(model.theta, theta)
        self.assertEqual(model.name, "Linear Regression")

        # test class returns correct predictions
        values = model(self.housing)
        self.assertEqual(len(values), len(self.housing))

        # test with intercept
        theta = np.ones(len(self.housing.domain.attributes) + 1)
        lr.set_theta(theta)
        lr.intercept = True
        model = lr.model

        assert_array_equal(model.theta, theta)

        # test class returns correct predictions
        values = model(self.housing)
        self.assertEqual(len(values), len(self.housing))

    def test_converged(self):
        """
        Test convergence flag or the algorithm
        """
        lr = self.linear_regression
        lr.set_data(self.housing)
        theta = np.ones(len(self.housing.domain.attributes))
        lr.set_theta(theta)
        lr.set_alpha(0.1)
        # we found out for example in test convergence is faster with this alpha

        # it can not converge in the first step
        self.assertFalse(lr.converged)

        # it converge when distance between current theta and this is < 1e-2
        converge = False
        while not converge:
            lr.step()
            converge = (np.sum(
                np.abs(lr.theta - lr.history[lr.step_no - 1][0])) /
                        len(lr.theta) < 1e-2)
            self.assertEqual(lr.converged, converge)

    def test_step(self):
        """
        Test step method
        """
        lr = self.linear_regression

        theta = np.ones(len(self.housing.domain.attributes))
        lr.set_theta(theta)
        lr.set_data(self.housing)

        # check beginning
        self.assertEqual(lr.step_no, 0)

        # perform step
        lr.step()

        # check if parameters are fine
        self.assertEqual(len(lr.theta), len(theta))
        assert_array_equal(lr.history[1][0], lr.theta)
        self.assertEqual(lr.step_no, 1)

        # perform step
        lr.step()

        # check if parameters are fine
        self.assertEqual(len(lr.theta), len(theta))
        assert_array_equal(lr.history[2][0], lr.theta)
        self.assertEqual(lr.step_no, 2)

        # check for stochastic
        lr.stochastic = True

        # perform step
        lr.step()
        self.assertEqual(len(lr.theta), len(theta))
        assert_array_equal(lr.history[3][0], lr.theta)

        # check if stochastic_i indices are ok
        self.assertEqual(
            lr.history[3][1], lr.stochastic_i)

        # reset algorithm
        lr.set_data(self.housing)

        # wait for shuffle and check if fine
        shuffle = False
        while not shuffle:
            lr.step()
            shuffle = lr.history[lr.step_no][2] is not None
            if shuffle:
                self.assertEqual(len(lr.x), len(self.housing))
                self.assertEqual(len(lr.y), len(self.housing))

        # check when intercept
        theta = np.ones(len(self.housing.domain.attributes) + 1)
        lr.set_theta(theta)
        lr.intercept = True
        lr.set_data(self.housing)

        # check beginning
        self.assertEqual(lr.step_no, 0)

        # perform step
        lr.step()

        # check if parameters are fine
        self.assertEqual(len(lr.theta), len(theta))
        assert_array_equal(lr.history[1][0], lr.theta)
        self.assertEqual(lr.step_no, 1)

        # perform step
        lr.step()

        # check if parameters are fine
        self.assertEqual(len(lr.theta), len(theta))
        assert_array_equal(lr.history[2][0], lr.theta)
        self.assertEqual(lr.step_no, 2)

        # check for stochastic
        lr.stochastic = True

        # perform step
        lr.step()
        self.assertEqual(len(lr.theta), len(theta))
        assert_array_equal(lr.history[3][0], lr.theta)

        # check if stochastic_i indices are ok
        self.assertEqual(
            lr.history[3][1], lr.stochastic_i)

        # reset algorithm
        lr.set_data(self.housing)

        # wait for shuffle and check if fine
        shuffle = False
        while not shuffle:
            lr.step()
            shuffle = lr.history[lr.step_no][2] is not None
            if shuffle:
                self.assertEqual(len(lr.x), len(self.housing))
                self.assertEqual(len(lr.y), len(self.housing))

    def test_step_back(self):
        """
        Test step back function
        """
        lr = self.linear_regression
        theta = np.ones(len(self.housing.domain.attributes))

        lr.set_data(self.housing)
        lr.set_theta(np.copy(theta))

        # check no step back when no step done before
        lr.step_back()
        assert_array_equal(lr.theta, theta)
        self.assertEqual(lr.step_no, 0)

        # perform step and step back
        lr.step()
        lr.step_back()
        assert_array_equal(lr.theta, theta)
        self.assertEqual(lr.step_no, 0)

        lr.step()
        theta1 = np.copy(lr.theta)
        lr.step()
        lr.step_back()

        assert_array_equal(lr.theta, theta1)
        self.assertEqual(lr.step_no, 1)

        lr.step_back()

        assert_array_equal(lr.theta, theta)
        self.assertEqual(lr.step_no, 0)

        # test for stochastic
        lr.stochastic = True

        lr.step()
        lr.step_back()
        self.assertEqual(lr.stochastic_i, 0)
        self.assertEqual(lr.step_no, 0)

        lr.step()
        lr.step()
        lr.step_back()

        self.assertEqual(lr.stochastic_i, lr.stochastic_step_size)
        self.assertEqual(lr.step_no, 1)

        lr.step_back()

        self.assertEqual(lr.stochastic_i, 0)
        self.assertEqual(lr.step_no, 0)

        # wait for shuffle and check if fine
        shuffle = False
        before = np.copy(lr.x)
        while not shuffle:
            lr.step()
            shuffle = lr.history[lr.step_no][2] is not None

        lr.step_back()
        assert_array_equal(lr.x, before)

        # with intercept
        lr = self.linear_regression
        theta = np.ones(len(self.housing.domain.attributes) + 1)

        lr.intercept = True
        lr.set_data(self.housing)
        lr.set_theta(np.copy(theta))

        # check no step back when no step done before
        lr.step_back()
        assert_array_equal(lr.theta, theta)
        self.assertEqual(lr.step_no, 0)

        # perform step and step back
        lr.step()
        lr.step_back()
        assert_array_equal(lr.theta, theta)
        self.assertEqual(lr.step_no, 0)

        lr.step()
        theta1 = np.copy(lr.theta)
        lr.step()
        lr.step_back()

        assert_array_equal(lr.theta, theta1)
        self.assertEqual(lr.step_no, 1)

        lr.step_back()

        assert_array_equal(lr.theta, theta)
        self.assertEqual(lr.step_no, 0)

        # test for stochastic
        lr.stochastic = True

        lr.step()
        lr.step_back()
        self.assertEqual(lr.stochastic_i, 0)
        self.assertEqual(lr.step_no, 0)

        lr.step()
        lr.step()
        lr.step_back()

        self.assertEqual(lr.stochastic_i, lr.stochastic_step_size)
        self.assertEqual(lr.step_no, 1)

        lr.step_back()

        self.assertEqual(lr.stochastic_i, 0)
        self.assertEqual(lr.step_no, 0)

        # wait for shuffle and check if fine
        shuffle = False
        before = np.copy(lr.x)
        while not shuffle:
            lr.step()
            shuffle = lr.history[lr.step_no][2] is not None

        lr.step_back()
        assert_array_equal(lr.x, before)

    def test_j(self):
        """
        Test cost function j
        """
        lr = self.linear_regression

        lr.set_data(self.housing)

        theta = np.ones(len(self.housing.domain.attributes))
        # test with one theta and with list of thetas
        self.assertEqual(type(lr.j(theta)), np.float64)
        self.assertEqual(
            len(lr.j(np.vstack((theta, theta * 2)))), 2)

    def test_dj(self):
        """
        Test gradient function
        """
        lr = self.linear_regression

        theta = np.ones(len(self.housing.domain.attributes))
        lr.set_data(self.housing)
        # check length with stochastic and usual
        self.assertEqual(len(lr.dj(theta)), len(theta))
        self.assertTupleEqual(
            lr.dj(theta, stochastic=True).shape, (30, len(theta)))

        lr.stochastic_step_size = 2
        self.assertTupleEqual(
            lr.dj(theta, stochastic=True).shape, (2, len(theta)))

        lr.stochastic_step_size = 1
        self.assertTupleEqual(
            lr.dj(theta, stochastic=True).shape, (1, len(theta)))

    def test_optimized(self):
        """
        Test if optimized works well
        """
        lr = self.linear_regression

        lr.set_data(self.housing)
        op_theta = lr.optimized()
        self.assertEqual(len(op_theta), len(self.housing.domain.attributes))

        # check if really minimal, function is monotonic so everywhere around
        # j should be higher
        attr_x = self.housing.domain['CRIM']
        attr_y = self.housing.domain['ZN']
        cols = []
        for attr in (attr_x, attr_y) if attr_y is not None else (attr_x, ):
            subset = self.housing[:, attr]
            cols.append(subset.X)
        x = np.column_stack(cols)

        domain = Domain([attr_x, attr_y], self.housing.domain.class_var)
        data = Normalize(transform_class=True)(Table(domain, x, self.housing.Y))

        lr.set_data(data)
        op_theta = lr.optimized()

        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([1, 0])))
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([0, 1])))
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([-1, 0])))
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([0, -1])))
