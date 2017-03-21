import numpy as np
import copy

__all__ = ['SGD', 'Momentum', 'NesterovMomentum', 'AdaGrad', 'RMSProp',
           'AdaDelta', 'Adam', 'Adamax', 'create_opt']


def create_opt(opt2copy, learning_rate=None):
    opt = copy.copy(opt2copy)  # Shallow copy
    if learning_rate:
        opt.learning_rate = learning_rate
    return opt


class SGD:
    """Stochastic Gradient Descent (SGD) updates

    Generates update expressions of the form:

    * ``param := param - learning_rate * gradient``

    Args:
        learning_rate: float, optional
            The learning rate controlling the size of update steps

    """

    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        self.name = 'Stochastic Gradient Descent'

    def update(self, grads, params, indices=None):
        """SGD updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices in params to update

        """
        if indices is None:
            indices = np.arange(len(params))

        params[indices] -= self.learning_rate * grads

    def __str__(self):
        return self.name


class Momentum:
    """Stochastic Gradient Descent (SGD) updates with momentum

    Generates update expressions of the form:

    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := param + velocity``

    Args:
        learning_rate: float
                The learning rate controlling the size of update steps
        momentum: float, optional
            The amount of momentum to apply. Higher momentum results in
            smoothing over more update steps. Defaults to 0.9.

    Notes:
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.

    See Also:
        apply_momentum: Generic function applying momentum to updates
        nesterov_momentum: Nesterov's variant of SGD with momentum

    """

    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.name = 'Momentum'

    def update(self, grads, params, indices=None):
        """Momentum updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array
                Indices in params to update

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.velocity is None:
            self.velocity = np.zeros(params.shape)

        self.velocity[indices] = \
            self.momentum * self.velocity[indices] - self.learning_rate * grads
        params[indices] += self.velocity[indices]

    def __str__(self):
        return self.name


class NesterovMomentum:
    """Stochastic Gradient Descent (SGD) updates with Nesterov momentum

    Generates update expressions of the form:

    * ``param_ahead := param + momentum * velocity``
    * ``velocity := momentum * velocity - learning_rate * gradient_ahead``
    * ``param := param + velocity``

    In order to express the update to look as similar to vanilla SGD, this can
    be written as:

    * ``v_prev := velocity``
    * ``velocity := momentum * velocity - learning_rate * gradient``
    * ``param := -momentum * v_prev + (1 + momentum) * velocity``

    Args:
        learning_rate : float
            The learning rate controlling the size of update steps
        momentum: float, optional
            The amount of momentum to apply. Higher momentum results in
            smoothing over more update steps. Defaults to 0.9.

    Notes:
        Higher momentum also results in larger update steps. To counter that,
        you can optionally scale your learning rate by `1 - momentum`.

        The classic formulation of Nesterov momentum (or Nesterov accelerated
        gradient) requires the gradient to be evaluated at the predicted next
        position in parameter space. Here, we use the formulation described at
        https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617,
        which allows the gradient to be evaluated at the current parameters.

    See Also:
        apply_nesterov_momentum: Function applying momentum to updates

    """

    def __init__(self, learning_rate=1.0, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None
        self.name = "Nesterov's Accelerated Momentum"

    def update(self, grads, params, indices=None):
        """NAG updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices in params to update

        Returns
            updates: list of float
                Variables updated with the gradients

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.velocity is None:
            self.velocity = np.zeros(params.shape)

        v_prev = self.velocity[indices]
        self.velocity[indices] = \
            self.momentum * self.velocity[indices] - self.learning_rate * grads
        params[indices] += -self.momentum * v_prev + \
                           (1 + self.momentum) * self.velocity[indices]

    def __str__(self):
        return self.name


class AdaGrad:
    """AdaGrad updates

    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.

    * ``param := param - learning_rate * gradient``

    Args:
        learning_rate : float or symbolic scalar
            The learning rate controlling the size of update steps
        epsilon: float or symbolic scalar
            Small value added for numerical stability

    Notes:
        Using step size eta Adagrad calculates the learning rate for feature i
        at time step t as:

        .. math:: \\eta_{t,i} = \\frac{\\eta}
           {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}

        as such the learning rate is monotonically decreasing.

        Epsilon is not included in the typical formula, see [2]_.

    References:
        .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
               Adaptive subgradient methods for online learning and stochastic
               optimization. JMLR, 12:2121-2159.

        .. [2] Chris Dyer:
               Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf

    """

    def __init__(self, learning_rate=1.0, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.accu = None
        self.name = 'AdaGrad'

    def update(self, grads, params, indices=None):
        """AdaGrad updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices in params to update

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.accu is None:
            self.accu = np.zeros(params.shape)

        self.accu[indices] += grads ** 2
        den = np.sqrt(self.accu[indices] + self.epsilon)
        params[indices] -= self.learning_rate * grads/den

    def __str__(self):
        return self.name


class RMSProp:
    """RMSProp

    Scale learning rates by dividing with the moving average of the root mean
    squared (RMS) gradients. See [3]_ for further description.

    Args:
        learning_rate: float
            The learning rate controlling the size of update steps
        rho: float
            Gradient moving average decay factor
        epsilon: float
            Small value added for numerical stability

    Notes:
        `rho` should be between 0 and 1. A value of `rho` close to 1 will decay
        the moving average slowly and a value close to 0 will decay the moving
        average fast.

        Using the step size :math:`\\eta` and a decay factor :math:`\\rho` the
        learning rate :math:`\\eta_t` is calculated as:

        .. math::
           r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
           \\eta_t &= \\frac{\\eta}{\\sqrt{r_t + \\epsilon}}

    References:
        .. [3] Tieleman, T. and Hinton, G. (2012):
               Neural Networks for Machine Learning, Lecture 6.5 - rmsprop.
               Coursera. http://www.youtube.com/watch?v=O3sxAc4hxZU
               (formula @5:20)

    """

    def __init__(self, learning_rate=1.0, rho=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.accu = None
        self.name = 'RMSProp'

    def update(self, grads, params, indices=None):
        """RMSProp updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices in params to update

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.accu is None:
            self.accu = np.zeros(params.shape)

        self.accu[indices] = \
            self.rho * self.accu[indices] + (1 - self.rho) * grads ** 2
        params[indices] -= self.learning_rate * grads /\
                           np.sqrt(self.accu[indices] + self.epsilon)

    def __str__(self):
        return self.name


class AdaDelta:
    """AdaDelta

    Scale learning rates by a the ratio of accumulated gradients to accumulated
    step sizes, see [4]_ and notes for further description.

    Args:
        learning_rate: float
            The learning rate controlling the size of update steps
        rho: float
            Squared gradient moving average decay factor
        epsilon: float
            Small value added for numerical stability

    Notes:
        rho should be between 0 and 1. A value of rho close to 1 will decay the
        moving average slowly and a value close to 0 will decay the moving
        average fast.

        rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to
        work for multiple datasets (MNIST, speech).

        In the paper, no learning rate is considered (so learning_rate=1.0).
        Probably best to keep it at this value.
        epsilon is important for the very first update (so the numerator does
        not become 0).

        Using the step size eta and a decay factor rho the learning rate is
        calculated as:

        .. math::
           r_t &= \\rho r_{t-1} + (1-\\rho)*g^2\\\\
           \\eta_t &= \\eta \\frac{\\sqrt{s_{t-1} + \\epsilon}}
                                 {\sqrt{r_t + \epsilon}}\\\\
           s_t &= \\rho s_{t-1} + (1-\\rho)*g^2

    References:
        .. [4] Zeiler, M. D. (2012):
               ADADELTA: An Adaptive Learning Rate Method.
               arXiv Preprint arXiv:1212.5701.
    """

    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.accu = None
        self.delta_accu = None
        self.name = 'AdaDelta'

    def update(self, grads, params, indices=None):
        """AdaDelta updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices in params to update

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.accu is None or self.delta_accu is None:
            self.accu = np.zeros(params.shape)
            self.delta_accu = np.zeros(params.shape)

        self.accu[indices] = self.rho * self.accu[indices] + \
                             (1 - self.rho) * grads ** 2

        # compute parameter update, using the 'old' delta_accu
        update = grads * np.sqrt(self.delta_accu[indices] + self.epsilon) / \
                 np.sqrt(self.accu[indices] + self.epsilon)
        params[indices] -= self.learning_rate * update

        # update delta_accu (as accu, but accumulating updates)
        delta_accu_new = \
            self.rho * self.delta_accu[indices] + (1 - self.rho) * update ** 2
        self.delta_accu[indices] = delta_accu_new

        return params

    def __str__(self):
        return self.name


class Adam:
    """Adam

    Adam updates implemented as in [5]_.

    Args:
        learning_rate : float
            The learning rate controlling the size of update steps
        beta_1 : float
            Exponential decay rate for the first moment estimates.
        beta_2 : float
            Exponential decay rate for the second moment estimates.
        epsilon : float
            Constant for numerical stability.

    Notes:
        The paper [5]_ includes an additional hyperparameter lambda. This is
        only needed to prove convergence of the algorithm and has no practical
        use, it is therefore omitted here.

    References:
        .. [5] Kingma, Diederik, and Jimmy Ba (2014):
               Adam: A Method for Stochastic Optimization.
               arXiv preprint arXiv:1412.6980.

    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
         epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t_prev = 0
        self.m_prev = None
        self.v_prev = None
        self.name = 'Adam'

    def update(self, grads, params, indices=None):
        """Adam updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices of parameters ('params') to update. If None (default),
                all parameters will be updated.

        Returns
            updates: list of float
                Variables updated with the gradients

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.m_prev is None or self.v_prev is None:
            self.m_prev = np.zeros(params.shape)
            self.v_prev = np.zeros(params.shape)

        t = self.t_prev + 1

        # To understand the coefficients plot this:
        #     sqrt(1-0.999^x)*(1-0.9^x)
        # or this:
        #     (1-0.999^x)*(1-0.9^x)
        # Computing bias-corrected first and second moment estimates to
        # counteract the effect of vt and mt been biased towards zero
        a_t = self.learning_rate * np.sqrt(1 - self.beta2 ** t) / \
              (1 - self.beta1 ** t)

        self.m_prev[indices] = self.beta1 * self.m_prev[indices] + \
                               (1 - self.beta1) * grads
        self.v_prev[indices] = self.beta2 * self.v_prev[indices] + \
                               (1 - self.beta2) * grads ** 2
        params[indices] -= a_t * self.m_prev[indices] / \
                           (np.sqrt(self.v_prev[indices]) + self.epsilon)

        self.t_prev = t

    def __str__(self):
        return self.name

class Adamax:
    """Adamax

    Adamax updates implemented as in [6]_. This is a variant of of the Adam
    algorithm based on the infinity norm.

    Args:
        learning_rate : float
            The learning rate controlling the size of update steps
        beta_1 : float
            Exponential decay rate for the first moment estimates.
        beta_2 : float
            Exponential decay rate for the second moment estimates.
        epsilon : float
            Constant for numerical stability.

    References:
        .. [6] Kingma, Diederik, and Jimmy Ba (2014):
               Adam: A Method for Stochastic Optimization.
               arXiv preprint arXiv:1412.6980.

    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
         epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t_prev = 0
        self.m_prev = None
        self.u_prev = None
        self.name = 'Adamax'

    def update(self, grads, params, indices=None):
        """Adamax updates

        Args:
            grads: array
                List of gradient expressions
            params: array
                The variables to generate update expressions for
            indices: array, optional
                Indices of parameters ('params') to update. If None (default),
                all parameters will be updated.

        Returns
            updates: list of float
                Variables updated with the gradients

        """

        if indices is None:
            indices = np.arange(len(params))

        if self.m_prev is None or self.u_prev is None:
            self.m_prev = np.zeros(params.shape)
            self.u_prev = np.zeros(params.shape)

        t = self.t_prev + 1
        a_t = self.learning_rate/(1 - self.beta1**t)

        self.m_prev[indices] = self.beta1 * self.m_prev[indices] + \
                               (1 - self.beta1) * grads
        self.u_prev[indices] = np.maximum(self.beta2 * self.u_prev[indices],
                                          np.abs(grads))
        params[indices] -= a_t * self.m_prev[indices] / \
                           (self.u_prev[indices] + self.epsilon)

        self.t_prev = t

    def __str__(self):
        return self.name