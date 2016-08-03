import numpy as np

from Orange.classification import Model
from scipy.optimize import fmin_l_bfgs_b


class LogisticRegression:

    x = None
    y = None
    theta = None
    domain = None
    step_no = 0

    def __init__(self, alpha=0.1, theta=None, data=None):
        self.history = []
        self.set_alpha(alpha)
        self.set_data(data)
        self.set_theta(theta)

    def set_data(self, data):
        if data is not None:
            self.x = data.X
            self.y = data.Y
            self.domain = data.domain

    def set_theta(self, theta):
        if isinstance(theta, (np.ndarray, np.generic)):
            self.theta = theta
        elif isinstance(theta, list):
            self.theta = np.array(theta)
        else:
            self.theta = None
        self.history = self.set_list(self.history, 0, np.copy(self.theta))
        self.step_no = 0

    def set_alpha(self, alpha):
        self.alpha = alpha

    @property
    def model(self):
        return LogisticRegressionModel(self.theta, self.domain)

    @property
    def converged(self):
        if self.step_no == 0:
            return False
        return np.sum(np.abs(self.theta - self.history[self.step_no - 1])) < 1e-2

    def step(self):
        self.step_no += 1
        grad = self.dj(self.theta)
        self.theta -= self.alpha * grad
        self.history = self.set_list(self.history, self.step_no, np.copy(self.theta))

    def step_back(self):
        if self.step_no > 0:
            self.step_no -= 1
            self.theta = np.copy(self.history[self.step_no])

    def j(self, theta):
        """
        Cost function for logistic regression
        """
        # TODO: modify for more thetas
        yh = self.g(self.x.dot(theta))
        # return -sum(np.log(self.y * yh + (1 - self.y) * (1 - yh))) / len(yh)
        return -sum(self.y * np.log(yh) + (1 - self.y) * np.log(1 - yh)) / len(yh)

    def dj(self, theta):
        """
        Gradient of the cost function with L2 regularization
        """
        return (self.g(self.x.dot(theta)) - self.y).dot(self.x)

    def optimized(self):
        """
        Function performs model training
        """
        res = fmin_l_bfgs_b(self.j,
                            np.zeros(self.x.shape[1]),
                            self.dj)
        return res[0]

    @staticmethod
    def g(z):
        """
        sigmoid function

        Parameters
        ----------
        z : array_like
            values to evaluate with function
        """

        # limit values in z to avoid log with 0 produced by values almost 0
        z_mod = np.minimum(z, 100 * np.ones(len(z)))
        z_mod = np.maximum(z_mod, -100 * np.ones(len(z)))

        return 1.0 / (1 + np.exp(- z_mod))

    @staticmethod
    def set_list(l, i, v):
        try:
            l[i] = v
        except IndexError:
            for _ in range(i-len(l)):
                l.append(None)
            l.append(v)
        return l

class LogisticRegressionModel(Model):

    def __init__(self, theta, domain):
        super().__init__(domain)
        self.theta = theta

    def predict_storage(self, data):
        return LogisticRegression.g(data.X.dot(self.theta))
