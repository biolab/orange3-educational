import unittest
from Orange.data import Table, Domain
from orangecontrib.educational.widgets.utils.logistic_regression import \
    LogisticRegression
from numpy.testing import *
import numpy as np

class TestKmeans(unittest.TestCase):

    def setUp(self):
        self.iris = Table('iris')
        # new_domain = Domain(self.data.domain.attributes[:2])
        # self.data = Table(new_domain, self.data)
        self.logistic_regression = LogisticRegression()

    def test_set_data(self):
        """
        Test set data
        """

        # check if None on beginning
        self.assertIsNone(self.logistic_regression.x, None)
        self.assertIsNone(self.logistic_regression.y, None)
        self.assertIsNone(self.logistic_regression.domain, None)

        # check if correct data are provided
        self.logistic_regression.set_data(self.iris)

        assert_array_equal(self.logistic_regression.x, self.iris.X)
        assert_array_equal(self.logistic_regression.y, self.iris.Y)
        self.assertEqual(self.logistic_regression.domain, self.iris.domain)

        # check data remove
        self.logistic_regression.set_data(None)

        self.assertIsNone(self.logistic_regression.x, None)
        self.assertIsNone(self.logistic_regression.y, None)
        self.assertIsNone(self.logistic_regression.domain, None)

    def test_set_theta(self):
        """
        Check set theta
        """

        lr = self.logistic_regression

        # theta must be none on beginning
        self.assertIsNone(lr.theta, None)

        # check if theta set correctly
        # theta from np array
        lr.set_theta(np.array([1, 2]))
        assert_array_equal(lr.theta, np.array([1, 2]))
        # history of 0 have to be equal theta
        assert_array_equal(lr.history[0][0], np.array([1, 2]))
        # step no have to reset to 0
        self.assertEqual(lr.step_no, 0)

        # theta from list
        lr.set_theta([2, 3])
        assert_array_equal(lr.theta, np.array([2, 3]))
        assert_array_equal(lr.history[0][0], np.array([2, 3]))
        self.assertEqual(lr.step_no, 0)

        # theta None
        lr.set_theta(None)
        self.assertIsNone(lr.theta)

        # theta anything else
        lr.set_theta("abc")
        self.assertIsNone(lr.theta)

    def test_set_alpha(self):
        """
        Check if alpha set correctly
        """
        lr = self.logistic_regression

        # check alpha 0.1 in the beginning
        self.assertEqual(lr.alpha, 0.1)

        # check if alpha set correctly
        lr.set_alpha(0.2)
        self.assertEqual(lr.alpha, 0.2)

        # check if alpha removed correctly
        lr.set_alpha(None)
        self.assertIsNone(lr.alpha)

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
        assert_array_equal(values, np.around(probabilities)[:,1])

    def test_converged(self):
        """
        Test convergence flag or the algorithm
        """
        lr = self.logistic_regression
        lr.set_data(self.iris)
        lr.set_theta([1., 1., 1., 1.])
        lr.set_alpha(1)
        # we found out for example in test convergence is faster with this alpha

        # it can not converge in the first step
        self.assertFalse(lr.converged)

        # it converge when distance between current theta and this is < 1e-2
        converge = False
        while not converge:
            lr.step()
            converge = np.sum(
                np.abs(lr.theta - lr.history[lr.step_no - 1][0])) < 1e-2
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
        self.assertEqual(lr.history[3][1], lr.stochastic_i)

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
        theta1 = np.copy(lr.theta)
        lr.step()
        lr.step_back()

        self.assertEqual(lr.stochastic_i, lr.stochastic_num_steps)
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






