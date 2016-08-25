import orangecontrib.educational.optimizers as opt

import numpy as np
import collections
import unittest
import copy

__optimizers__ = [
    opt.SGD(learning_rate=0.1),
    opt.Momentum(learning_rate=0.1, momentum=0.5),
    opt.NesterovMomentum(learning_rate=0.1, momentum=0.5),
    opt.AdaGrad(learning_rate=0.1),
    opt.RMSProp(learning_rate=0.01, rho=0.9, epsilon=1e-6),
    opt.AdaDelta(learning_rate=1, rho=0.95, epsilon=1e-6),
    opt.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8),
    opt.Adamax(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    ]


def dxf(X):
    tensor = [0.1, 0.2, 0.3]
    return tensor * X * 2  # f(x) = x^2 -> dx(f(x)) = 2x


def dxf2(X):
    return X*2


class TestOptimizers(unittest.TestCase):

    # These tests compare results on a toy problem to values
    # calculated by the torch.optim package, using this script:
    # https://gist.github.com/ebenolson/931e879ed38f257253d2

    # Note: In 'optim.rmsprop' the running average 'm' (state.m) is initialized
    # to ones. So we modified their code to initialize it to zero.

    # Torch's reason: https://github.com/torch/optim/issues/87
    # Our reason: "Follow the most common initialization"
    torch_values = collections.OrderedDict()
    torch_values['sgd'] = [0.81707280688755,
                           0.6648326359915,
                           0.5386151140949]

    torch_values['momentum'] = [0.6848486952183,
                                0.44803321781003,
                                0.27431190123502]

    torch_values['nesterov_momentum'] = [0.67466543592725,
                                         0.44108468114241,
                                         0.2769002108997]

    torch_values['adagrad'] = [0.55373120047759,
                               0.55373120041518,
                               0.55373120039438]

    torch_values['rmsprop'] = [0.83205403985348,
                               0.83205322744821,
                               0.83205295664444]

    torch_values['adadelta'] = [0.95453237704725,
                                0.9545237471374,
                                0.95452214847397]

    torch_values['adam'] = [0.90034973381771,
                            0.90034969365796,
                            0.90034968027137]

    torch_values['adamax'] = [0.90211749000754,
                              0.90211748762402,
                              0.90211748682951]

    def test_torch(self):
        for _opt, torch in zip(__optimizers__, self.torch_values.values()):
            print('Testing: %s' % _opt.__str__())  # Testing __str__

            opt_copy = copy.copy(_opt)  # Shallow copy (test dependent)
            x = np.asarray([1., 1., 1.], dtype=np.float64)  # Initial state

            for _ in range(10):
                opt_copy.update(dxf(x), x)
            np.testing.assert_almost_equal(x, torch, decimal=4)

    def test_convergence(self):
        res = []
        x0 = 100

        # Test SGD optimizers too
        for _opt in __optimizers__:
            opt_copy = copy.copy(_opt)  # Shallow copy (test dependent)
            x = np.asarray([x0], dtype=np.float64)  # Initial state
            for _ in range(10):
                opt_copy.update(dxf2(x), x)
            res.append(x)

        test = list(
            map(lambda t: abs(t) < x0, res))
        self.assertTrue(all(test))

    def test_opt_cloner(self):  # Dumb test. Just for coverage
        opt_1 = opt.create_opt(opt.SGD())
        opt_1.learning_rate = 0.5
        opt_2 = opt.create_opt(opt_1, opt_1.learning_rate)

        self.assertIsInstance(opt_1, opt.SGD)
        self.assertIsInstance(opt_2, opt.SGD)
        self.assertTrue(opt_1.learning_rate == opt_2.learning_rate)

    def test_params(self):  # Dumb test. Just for coverage
        x0 = [10, 20, 30]
        indices = np.arange(len(x0))

        for _opt in __optimizers__:
            opt_copy = copy.copy(_opt)  # Shallow copy (test dependent)
            x = np.asarray(x0, dtype=np.float64)  # Initial state

            for _ in range(10):
                opt_copy.update(dxf2(x), x, indices)

if __name__ == "__main__":
    # # Test all
    unittest.main()

    # # Test single test
    # suite = unittest.TestSuite()
    # suite.addTest(TestOptimizers("test_convergence"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
