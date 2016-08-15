import numpy as np

from Orange.classification import Model

from orangecontrib.educational.widgets.utils.gradient_descent import \
    GradientDescent


class LinearRegression(GradientDescent):
    """
    Logistic regression algorithm with custom cost and gradient function,
    which allow to perform algorithm step by step
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def model(self):
        """
        Function returns model based on current parameters.
        """
        if self.theta is None or self.domain is None:
            return None
        else:
            return LinearRegressionModel(
                self.theta, self.domain, intercept=self.intercept)

    def j(self, theta):
        """
        Cost function for logistic regression
        """
        h = self.h(self.x, theta)
        return 1.0 / 2.0 * np.sum(np.square(h - self.y).T, axis=0) / len(self.y)

    def dj(self, theta, stochastic=False):
        """
        Gradient of the cost function for logistic regression
        """
        if stochastic:
            ns = self.stochastic_step_size
            x = self.x[self.stochastic_i: self.stochastic_i + ns]
            y = self.y[self.stochastic_i: self.stochastic_i + ns]
            h = self.h(x, theta)
            return x * (h - y)[:, None] / len(y)
        else:
            x = self.x
            y = self.y
            h = self.h(x, theta)
            return x.T.dot(h - y) / len(y)

    @staticmethod
    def h(x, theta):
        return x.dot(theta.T).T


class LinearRegressionModel(Model):

    def __init__(self, theta, domain, intercept=False):
        super().__init__(domain)
        self.theta = theta
        self.name = "Linear Regression"
        self.intercept = intercept

    def predict_storage(self, data):
        x = np.c_[np.ones(len(data.X)), data.X] if self.intercept else data.X
        return LinearRegression.h(x, self.theta)
