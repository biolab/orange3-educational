import numpy as np

from Orange.classification import Model
from scipy.optimize import fmin_l_bfgs_b


class LogisticRegression:

    x = None
    y = None
    theta = None
    alpha = None
    domain = None
    step_no = 0
    stochastic_i = 0
    stochastic_num_steps = 30  # number of steps in one step

    def __init__(self, alpha=0.1, theta=None, data=None, stochastic=False):
        self.history = []
        self.set_alpha(alpha)
        self.set_data(data)
        self.set_theta(theta)
        self.stochastic = stochastic

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
        self.history = self.set_list(self.history, 0, (np.copy(self.theta), 0))
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
        return (np.sum(np.abs(self.theta - self.history[self.step_no - 1][0])) <
                (1e-2 if not self.stochastic else 1e-5))

    def step(self):
        self.step_no += 1
        grad = self.dj(self.theta, self.stochastic)
        self.theta -= self.alpha * grad

        self.stochastic_i += self.stochastic_num_steps

        seed = None  # seed that will be stored to revert the shuffle
        if self.stochastic_i >= len(self.x):
            self.stochastic_i = 0
            seed = np.random.randint(100)  # random seed
            np.random.seed(seed)  # set seed of permutation used to shuffle
            indices = np.random.permutation(len(self.x))
            self.x = self.x[indices]  # permutation
            self.y = self.y[indices]

        self.history = self.set_list(
            self.history, self.step_no,
            (np.copy(self.theta), self.stochastic_i, seed))

    def step_back(self):
        if self.step_no > 0:
            self.step_no -= 1
            self.theta = np.copy(self.history[self.step_no][0])
            self.stochastic_i = self.history[self.step_no][1]
            seed = self.history[self.step_no + 1][2]
            if seed is not None:  # it means data had been permuted on this pos
                np.random.seed(seed)  # use same seed to revert
                indices = np.random.permutation(len(self.x))
                indices_reverse = np.argsort(indices)
                # indices of sorted indices gives us reversing shuffle list
                self.x = self.x[indices_reverse]
                self.y = self.y[indices_reverse]

    def j(self, theta):
        """
        Cost function for logistic regression
        """
        yh = self.g(self.x.dot(theta.T)).T
        y = self.y
        return -np.sum(
            (self.y * np.log(yh) + (1 - y) * np.log(1 - yh)).T, axis=0) / len(y)

    def dj(self, theta, stochastic=False):
        """
        Gradient of the cost function with L2 regularization
        """
        if stochastic:
            ns = self.stochastic_num_steps
            x = self.x[self.stochastic_i : self.stochastic_i + ns]
            y = self.y[self.stochastic_i : self.stochastic_i + ns]
            return x.T.dot(self.g(x.dot(theta)) - y)
        else:
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
        z_mod = np.minimum(z, 100)
        z_mod = np.maximum(z_mod, -100)

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
