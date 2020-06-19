import numpy as np

from Orange.classification import Model

from orangecontrib.educational.widgets.utils.gradient_descent import \
    GradientDescent


class LogisticRegression(GradientDescent):
    """
    Logistic regression algorithm with custom cost and gradient function,
    which allow to perform algorithm step by step
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data=data, **kwargs)
        self._check_data(data)

    @staticmethod
    def _check_data(data):
        if data is not None and len(data.domain.class_var.values) > 2:
            raise ValueError(
                "This implementation is made for class with only two values"
            )

    @property
    def model(self):
        """
        Function returns model based on current parameters.
        """
        if self.theta is None or self.domain is None:
            return None
        else:
            return LogisticRegressionModel(self.theta, self.domain)

    def set_data(self, data):
        self._check_data(data)
        super().set_data(data)

    def j(self, theta):
        """
        Cost function for logistic regression
        """
        h = self.g(self.x.dot(theta.T)).T
        y = self.y
        return ((-np.sum((y * np.log(h) + (1 - y) * np.log(1 - h)).T, axis=0) +
                self.regularization_rate * np.sum(np.square(theta.T), axis=0)) /
                len(y))

    def dj(self, theta, stochastic=False):
        """
        Gradient of the cost function for logistic regression
        """
        if stochastic:
            ns = self.stochastic_step_size
            x = self.x[self.stochastic_i: self.stochastic_i + ns]
            y = self.y[self.stochastic_i: self.stochastic_i + ns]
            return x * (self.g(x.dot(theta)) - y)[:, None] / len(y)
        else:
            return ((self.g(self.x.dot(theta)) - self.y).dot(self.x) +
                   self.regularization_rate * theta) / len(self.y)

    @staticmethod
    def g(z):
        """
        Sigmoid function

        Parameters
        ----------
        z : array_like(float)
            values to evaluate with function
        """

        # limit values in z to avoid log with 0 produced by values almost 0
        z_mod = np.minimum(z, 20)
        z_mod = np.maximum(z_mod, -20)

        return 1.0 / (1 + np.exp(- z_mod))


class LogisticRegressionModel(Model):

    def __init__(self, theta, domain):
        super().__init__(domain)
        self.theta = theta
        self.name = "Logistic Regression"

    def predict_storage(self, data):
        probabilities = LogisticRegression.g(data.X.dot(self.theta))
        values = np.around(probabilities)
        probabilities0 = 1 - probabilities
        probabilities = np.column_stack((probabilities0, probabilities))
        return values, probabilities
