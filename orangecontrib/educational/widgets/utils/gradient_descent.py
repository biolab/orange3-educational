import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from Orange.classification import Model


class GradientDescent:
    """
    Logistic regression algorithm with custom cost and gradient function,
    which allow to perform algorithm step by step

    Parameters
    ----------
    alpha : float
        Learning rate
    theta : array_like(float, ndim=1)
        Logistic function parameters
    data : Orange.data.Table
        Data
    """

    x = None
    y = None
    theta = None
    alpha = None
    domain = None
    step_no = 0
    stochastic_i = 0
    stochastic_step_size = 30  # number of steps in one step
    regularization_rate = 0.001
    # very small regularization rate to avoid big parameters

    def __init__(self, alpha=0.1, theta=None, data=None, stochastic=False,
                 step_size=30, intercept=False):
        self.history = []
        self.intercept=intercept
        self.set_alpha(alpha)
        self.set_data(data)
        self.set_theta(theta)
        self.stochastic = stochastic
        self.stochastic_step_size = step_size

    def set_data(self, data):
        """
        Function set the data
        """
        if data is not None:
            x = data.X
            self.x = np.c_[np.ones(len(x)), x] if self.intercept else x
            self.y = data.Y
            self.domain = data.domain
        else:
            self.x = None
            self.y = None
            self.domain = None

    def set_theta(self, theta):
        """
        Function sets theta. Can be called from constructor or outside.
        """
        if isinstance(theta, (np.ndarray, np.generic)):
            self.theta = theta
        elif isinstance(theta, list):
            self.theta = np.array(theta)
        else:
            self.theta = None
        self.history = self.set_list(
            self.history, 0, (np.copy(self.theta), 0, None))
        self.step_no = 0

    def set_alpha(self, alpha):
        """
        Function sets alpha and can be called from constructor or from outside.
        """
        self.alpha = alpha

    @property
    def model(self):
        """
        Function returns model based on current parameters.
        """
        raise NotImplementedError("Subclasses should implement this!")

    @property
    def converged(self):
        """
        Function returns True if gradient descent already converged.
        """
        if self.step_no == 0:
            return False
        return (np.sum(np.abs(self.theta - self.history[self.step_no - 1][0])) /
                len(self.theta) < (1e-2 if not self.stochastic else 1e-5))

    def step(self):
        """
        Function performs one step of the gradient descent
        """
        self.step_no += 1

        seed = None  # seed that will be stored to revert the shuffle
        # if we came around all data set index to zero and permute data
        if self.stochastic_i + self.stochastic_step_size >= len(self.x):
            self.stochastic_i = 0

            # shuffle data
            seed = np.random.randint(100)  # random seed
            np.random.seed(seed)  # set seed of permutation used to shuffle
            indices = np.random.permutation(len(self.x))
            self.x = self.x[indices]  # permutation
            self.y = self.y[indices]

        # calculates gradient and modify theta
        grad = self.dj(self.theta, self.stochastic)
        if self.stochastic:
            self.theta -= np.sum(np.float64(self.alpha) * grad, axis=0)
        else:
            self.theta -= self.alpha * grad

        # increase index used by stochastic gradient descent
        self.stochastic_i += self.stochastic_step_size

        # save history for step back
        self.history = self.set_list(
            self.history, self.step_no,
            (np.copy(self.theta), self.stochastic_i, seed))

    def step_back(self):
        if self.step_no > 0:
            self.step_no -= 1

            # modify theta
            self.theta = np.copy(self.history[self.step_no][0])

            # modify index for stochastic gradient descent
            self.stochastic_i = self.history[self.step_no][1]

            # if necessary restore data shuffle
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
        raise NotImplementedError("Subclasses should implement this!")

    def dj(self, theta, stochastic=False):
        """
        Gradient of the cost function for logistic regression
        """
        raise NotImplementedError("Subclasses should implement this!")

    def optimized(self):
        """
        Function performs whole model training. Not step by step.
        """

        res = fmin_l_bfgs_b(self.j,
                            np.zeros(self.x.shape[1]),
                            self.dj)
        return res[0]

    @staticmethod
    def set_list(l, i, v):
        """
        Function sets i-th value in list to v. If i does not exist in l
        it is initialized else value is modified
        """
        try:
            l[i] = v
        except IndexError:
            for _ in range(i-len(l)):
                l.append(None)
            l.append(v)
        return l
