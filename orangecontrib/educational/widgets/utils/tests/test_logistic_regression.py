import unittest
from Orange.data import Table, Domain, DiscreteVariable
from orangecontrib.educational.widgets.utils.logistic_regression import \
    LogisticRegression
from numpy.testing import *
import numpy as np


class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.iris = Table('iris')
        class_var = DiscreteVariable(
            "iris", values=self.iris.domain.class_var.values[:2]
        )
        domain = Domain(self.iris.domain.attributes, class_var)
        self.iris = Table.from_table(domain, self.iris[:100])
        self.logistic_regression = LogisticRegression()

    def test_model(self):
        """
        Test if model is correct
        """
        lr = self.logistic_regression

        # test if model None when no data
        lr.set_theta([1, 2])
        self.assertIsNone(lr.model)

        # test if model None when no theta
        lr.set_theta(None)
        lr.set_data(self.iris)
        self.assertIsNone(lr.model)

        # test if model None when no theta and no Data
        lr.set_data(None)
        self.assertIsNone(lr.model)

        # test when model is not none
        lr.set_data(self.iris)
        lr.set_theta([1, 1, 1, 1])
        model = lr.model

        # test parameters are ok
        self.assertIsNotNone(model)
        assert_array_equal(model.theta, np.array([1, 1, 1, 1]))
        self.assertEqual(model.name, "Logistic Regression")

        # test class returns correct predictions
        values, probabilities = model(self.iris, ret=2)
        self.assertEqual(len(values), len(self.iris))
        self.assertEqual(len(probabilities), len(self.iris))
        # values have to be 0 if prob <0.5 else 1
        assert_array_equal(values, np.around(probabilities)[:, 1])

    def test_converged(self):
        """
        Test convergence flag or the algorithm
        """
        lr = self.logistic_regression
        lr.set_data(self.iris)
        lr.set_theta([1., 1., 1., 1.])
        lr.set_alpha(100)
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
        lr = self.logistic_regression

        lr.set_theta([1., 1., 1., 1.])
        lr.set_data(self.iris)

        # check beginning
        self.assertEqual(lr.step_no, 0)

        # perform step
        lr.step()

        # check if parameters are fine
        self.assertEqual(len(lr.theta), 4)
        assert_array_equal(lr.history[1][0], lr.theta)

        # perform step
        lr.step()

        # check if parameters are fine
        self.assertEqual(len(lr.theta), 4)
        assert_array_equal(lr.history[2][0], lr.theta)

        # check for stochastic
        lr.stochastic = True

        # perform step
        lr.step()
        self.assertEqual(len(lr.theta), 4)
        assert_array_equal(lr.history[3][0], lr.theta)

        # check if stochastic_i indices are ok
        self.assertEqual(
            lr.history[3][1], lr.stochastic_i)

        # reset algorithm
        lr.set_data(self.iris)

        # wait for shuffle and check if fine
        shuffle = False
        while not shuffle:
            lr.step()
            shuffle = lr.history[lr.step_no][2] is not None
            if shuffle:
                self.assertEqual(len(lr.x), len(self.iris))
                self.assertEqual(len(lr.y), len(self.iris))

    def test_step_back(self):
        """
        Test step back function
        """
        lr = self.logistic_regression
        theta = [1., 1., 1., 1.]

        lr.set_data(self.iris)
        lr.set_theta(theta)

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
        lr = self.logistic_regression

        lr.set_data(self.iris)

        # test with one theta and with list of thetas
        self.assertEqual(type(lr.j(np.array([1., 1., 1., 1.]))), np.float64)
        self.assertEqual(
            len(lr.j(np.array([[1., 1., 1., 1.], [2, 2, 2, 2]]))), 2)

    def test_dj(self):
        """
        Test gradient function
        """
        lr = self.logistic_regression

        lr.set_data(self.iris)
        # check length with stochastic and usual
        self.assertEqual(len(lr.dj(np.array([1, 1, 1, 1]))), 4)
        self.assertTupleEqual(
            lr.dj(np.array([1, 1, 1, 1]), stochastic=True).shape, (30, 4))

        lr.stochastic_step_size = 2
        self.assertTupleEqual(
            lr.dj(np.array([1, 1, 1, 1]), stochastic=True).shape, (2, 4))

        lr.stochastic_step_size = 1
        self.assertTupleEqual(
            lr.dj(np.array([1, 1, 1, 1]), stochastic=True).shape, (1, 4))

    def test_optimized(self):
        """
        Test if optimized works well
        """
        lr = self.logistic_regression

        lr.set_data(self.iris)
        op_theta = lr.optimized()
        self.assertEqual(len(op_theta), 4)

        # check if really minimal, function is monotonic so everywhere around
        # j should be higher
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([1, 0, 0, 0])))
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([0, 1, 0, 0])))
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([0, 0, 1, 0])))
        self.assertLessEqual(
            lr.j(op_theta), lr.j(op_theta + np.array([0, 0, 0, 1])))

    def test_g(self):
        """
        Test sigmoid function
        """
        lr = self.logistic_regression

        # test length
        self.assertEqual(type(lr.g(1)), np.float64)
        self.assertEqual(len(lr.g(np.array([1, 1]))), 2)
        self.assertEqual(len(lr.g(np.array([1, 1, 1]))), 3)
        self.assertEqual(len(lr.g(np.array([1, 1, 1, 1]))), 4)

        # test correctness, function between 0 and 1
        self.assertGreaterEqual(lr.g(-10000), 0)
        self.assertGreaterEqual(lr.g(-1000), 0)
        self.assertGreaterEqual(lr.g(-10), 0)
        self.assertGreaterEqual(lr.g(-1), 0)
        self.assertGreaterEqual(lr.g(0), 0)
        self.assertGreaterEqual(lr.g(1), 0)
        self.assertGreaterEqual(lr.g(10), 0)
        self.assertGreaterEqual(lr.g(1000), 0)
        self.assertGreaterEqual(lr.g(10000), 0)

        self.assertLessEqual(lr.g(-10000), 1)
        self.assertLessEqual(lr.g(-1000), 1)
        self.assertLessEqual(lr.g(-10), 1)
        self.assertLessEqual(lr.g(-1), 1)
        self.assertLessEqual(lr.g(0), 1)
        self.assertLessEqual(lr.g(1), 1)
        self.assertLessEqual(lr.g(10), 1)
        self.assertLessEqual(lr.g(1000), 1)
        self.assertLessEqual(lr.g(10000), 1)

    def test_set_list(self):
        """
        Test set list
        """
        lr = self.logistic_regression

        # test adding Nones if list too short
        self.assertEqual(lr.set_list([], 2, 1), [None, None, 1])
        # test adding Nones if list too short
        self.assertEqual(lr.set_list([2], 2, 1), [2, None, 1])
        # adding to end
        self.assertEqual(lr.set_list([2, 1], 2, 1), [2, 1, 1])
        # changing the element in the last place
        self.assertEqual(lr.set_list([2, 1], 1, 3), [2, 3])
        # changing the element in the middle place
        self.assertEqual(lr.set_list([2, 1, 3], 1, 3), [2, 3, 3])
