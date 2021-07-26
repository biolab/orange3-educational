import unittest
from unittest.mock import Mock

import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_equal

from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.classification import \
    LogisticRegressionLearner, TreeLearner, SVMLearner
from Orange.preprocess.preprocess import Continuize, Discretize
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.educational.widgets.owpolynomialclassification import \
    OWPolynomialClassification, GRID_SIZE
from orangecontrib.educational.widgets.utils.polynomialtransform import \
    PolynomialTransform


class TestOWPolynomialClassificationNoGrid(WidgetTest):
    # Tests with mocked computation of probability grid and contours
    # These are never tested and just slow everything down.
    def setUp(self):
        # type: OWPolynomialClassification
        self.widget = self.create_widget(OWPolynomialClassification)

        def plot_gradient():
            self.widget.probabilities_grid = self.widget.model and np.zeros((GRID_SIZE, GRID_SIZE))

        self.widget.plot_gradient = plot_gradient
        self.widget.plot_contour = Mock()

        self.iris = Table.from_file("iris")

    def test_init(self):
        w = self.widget
        self.assertIsInstance(w.learner, LogisticRegressionLearner)
        self.assertIsInstance(self.get_output(w.Outputs.learner),
                              LogisticRegressionLearner)
        self.assertIsNone(self.get_output(w.Outputs.model))
        self.assertIsNone(self.get_output(w.Outputs.coefficients))

    def test_set_learner(self):
        w = self.widget
        self.assertIsInstance(w.learner, LogisticRegressionLearner)
        self.assertIsInstance(self.get_output(w.Outputs.learner),
                              LogisticRegressionLearner)

        self.send_signal(w.Inputs.learner, SVMLearner())
        self.assertIsInstance(self.get_output(w.Outputs.learner), SVMLearner)

        self.send_signal(w.Inputs.learner, None)
        self.assertIsInstance(w.learner, LogisticRegressionLearner)
        self.assertIsInstance(self.get_output(w.Outputs.learner),
                              LogisticRegressionLearner)

    def test_set_preprocessor(self):
        w = self.widget

        preprocessor = Continuize()
        self.send_signal(w.Inputs.preprocessor, preprocessor)

        for learner in (None, SVMLearner(), None):
            self.send_signal(w.Inputs.learner, learner)

            self.assertEqual(w.preprocessors, [preprocessor])
            preprocessors = self.get_output(w.Outputs.learner).preprocessors
            self.assertIn(preprocessor, preprocessors)
            self.assertTrue(isinstance(pp, PolynomialTransform)
                            for pp in preprocessors)

            self.send_signal(w.Inputs.preprocessor, None)
            self.assertIn(w.preprocessors, [[], None])
            self.assertNotIn(preprocessor, self.get_output(w.Outputs.learner).preprocessors)
            self.assertTrue(isinstance(pp, PolynomialTransform)
                            for pp in preprocessors)

            self.send_signal(w.Inputs.preprocessor, preprocessor)
            self.assertEqual(w.preprocessors, [preprocessor])
            self.assertIn(preprocessor, self.get_output(w.Outputs.learner).preprocessors)
            self.assertTrue(isinstance(pp, PolynomialTransform)
                            for pp in preprocessors)

    def test_set_data(self):
        w = self.widget
        attr0, attr1 = self.iris.domain.attributes[:2]
        class_vals = self.iris.domain.class_var.values

        self.send_signal(w.Inputs.data, self.iris[::15])

        self.assertEqual(w.var_model.rowCount(), 4)
        self.assertEqual(w.controls.target_class.count(), len(class_vals))
        self.assertIs(w.attr_x, attr0)
        self.assertIs(w.attr_y, attr1)
        self.assertEqual(w.target_class, class_vals[0])

        # remove data set
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(w.var_model.rowCount(), 0)
        self.assertEqual(w.controls.target_class.count(), 0)

    def _set_iris(self):
        # Set some data so the widget goes up and test can check that it is
        # torn down when erroneous data is received
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.assert_all_up()

    def assert_all_up(self):
        w = self.widget
        self.assertNotEqual(len(w.var_model), 0)
        self.assertNotEqual(w.controls.target_class.count(), 0)
        self.assertIsNotNone(w.data)
        self.assertIsNotNone(w.selected_data)
        self.assertIsNotNone(w.model)
        self.assertIsNotNone(w.probabilities_grid)

    def assert_all_down(self):
        w = self.widget
        self.assertEqual(len(w.var_model), 0)
        self.assertEqual(w.controls.target_class.count(), 0)
        self.assertIsNone(w.data)
        self.assertIsNone(w.selected_data)
        self.assertIsNone(w.model)
        self.assertIsNone(w.probabilities_grid)

    def test_set_data_no_class(self):
        w = self.widget

        self._set_iris()

        table_no_class = self.iris.transform(Domain(self.iris.domain.attributes, []))
        self.send_signal(w.Inputs.data, table_no_class)
        self.assert_all_down()

    def test_set_data_regression(self):
        w = self.widget
        self._set_iris()

        table_regr = Table.from_numpy(
            Domain(self.iris.domain.attributes[:3],
                   self.iris.domain.attributes[3]),
            self.iris.X[:, :3], self.iris.X[:, 3])
        self.send_signal(w.Inputs.data, table_regr)
        self.assert_all_down()

    def test_set_data_one_class(self):
        w = self.widget
        self._set_iris()

        table_one_class = Table.from_numpy(
            Domain(self.iris.domain.attributes,
                   DiscreteVariable("a", values=("k", ))),
            self.iris.X, np.zeros(150))
        self.send_signal(w.Inputs.data, table_one_class)

        self.assertEqual(len(w.var_model), 4)
        self.assertEqual(w.controls.target_class.count(), 1)
        self.assertIsNotNone(w.data)
        self.assertIsNotNone(w.selected_data)
        self.assertIsNone(w.model)
        self.assertIsNone(w.probabilities_grid)
        self.assertTrue(w.Error.fitting_failed.is_shown())

    def test_set_data_wrong_var_number(self):
        w = self.widget

        self._set_iris()

        table_no_enough_cont = Table.from_numpy(
            Domain(
                [ContinuousVariable("x"),
                 DiscreteVariable("y", values=("a", "b"))],
                DiscreteVariable("a", values=("a", "b"))),
            [[1, 0], [2, 1]], [0, 0])
        self.send_signal(w.Inputs.data, table_no_enough_cont)

        self.assert_all_down()
        self.assertTrue(w.Error.num_features.is_shown())

    def test_select_data(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)

        w.select_data()
        np.testing.assert_equal(w.selected_data.X, self.iris.X[:, :2])
        np.testing.assert_equal(w.selected_data.Y, [1] * 50 + [0] * 100)

        w.attr_x, w.attr_y = self.iris.domain[1:3]
        w.select_data()
        np.testing.assert_equal(w.selected_data.X, self.iris.X[:, 1:3])
        np.testing.assert_equal(w.selected_data.Y, [1] * 50 + [0] * 100)
        self.assertIsNotNone(w.model)
        self.assertIsNotNone(w.probabilities_grid)

        # data without valid rows
        data = Table.from_numpy(
            Domain([ContinuousVariable('a'), ContinuousVariable('b')],
                   DiscreteVariable('c', values=('a', 'b'))),
            [[1, None], [None, 1]], [0, 1]
        )
        self.send_signal(w.Inputs.data, data)
        w.select_data()
        self.assertIsNone(w.selected_data)
        self.assertIsNone(w.model)
        self.assertIsNone(w.probabilities_grid)

    def test_update_model(self):
        w = self.widget
        self.assertIsNone(w.model)
        self.assertIsNone(self.get_output(w.Outputs.model))

        self.send_signal(w.Inputs.data, self.iris)
        self.assertIsNotNone(w.model)
        self.assertEqual(w.model, self.get_output(w.Outputs.model))

        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.model)
        self.assertIsNone(self.get_output(w.Outputs.model))

    def test_send_coefficients(self):
        w = self.widget

        self.assertIsNone(self.get_output(w.Outputs.coefficients))

        self.send_signal(w.Inputs.data, self.iris)

        for w.degree in range(1, 6):
            w._on_degree_changed()
            self.assertEqual(
                len(self.get_output(w.Outputs.coefficients)),
                (w.degree + 1) * (w.degree + 2) // 2,
                f"at degree={w.degree}")
            self.assertEqual(
                len(self.get_output(w.Outputs.data).domain.attributes),
                (w.degree + 1) * (w.degree + 2) // 2 - 1,
                f"at degree={w.degree}")

        # change learner which does not have coefficients
        learner = TreeLearner
        self.send_signal(w.Inputs.learner, learner())
        self.assertIsNone(self.get_output(w.Outputs.coefficients))
        self.assertIsNotNone(self.get_output(w.Outputs.data))

        self.send_signal(w.Inputs.learner, None)
        for w.degree in range(1, 6):
            w._on_degree_changed()
            self.assertEqual(len(self.get_output(w.Outputs.coefficients)),
                             (w.degree + 1) * (w.degree + 2) // 2,
                             f"at degree={w.degree}")

    def test_bad_learner(self):
        w = self.widget

        self.assertFalse(w.Error.fitting_failed.is_shown())
        learner = LogisticRegressionLearner()
        learner.preprocessors = [Discretize()]
        self.send_signal(w.Inputs.learner, learner)
        self.send_signal(w.Inputs.data, self.iris)
        self.assertTrue(w.Error.fitting_failed.is_shown())
        learner.preprocessors = []
        self.send_signal(w.Inputs.learner, learner)
        self.assertFalse(w.Error.fitting_failed.is_shown())

    def test_sparse(self):
        w = self.widget

        def send_sparse_data(data):
            data.X = sp.csr_matrix(data.X)
            data.Y = sp.csr_matrix(data.Y)
            self.send_signal(w.Inputs.data, data)

        # one class variable
        send_sparse_data(Table.from_file("iris")[::15])

        # two class variables
        data = Table.from_file("iris")[::15]
        domain = Domain(
            attributes=data.domain.attributes[:3],
            class_vars=data.domain.attributes[3:] + data.domain.class_vars
        )
        send_sparse_data(data.transform(domain))

    def test_non_in_data(self):
        w = self.widget
        self.iris.Y[:10] = np.nan
        self.iris.X[-4:, 0] = np.nan

        self.send_signal(w.Inputs.data, self.iris)
        np.testing.assert_equal(w.selected_data.X, self.iris.X[:-4, :2])

    def test_same_variable(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)

        assert w.selected_data is not None
        assert w.attr_x is w.data.domain[0]

        w.attr_y = w.data.domain[0]
        w._on_attr_changed()
        self.assertIsNone(w.selected_data)
        self.assertTrue(w.Error.same_variable.is_shown())

        w.attr_y = w.data.domain[1]
        w._on_attr_changed()
        self.assertIsNotNone(w.selected_data)
        self.assertFalse(w.Error.same_variable.is_shown())

        self.send_signal(w.Inputs.data, None)
        self.assertFalse(w.Error.same_variable.is_shown())

        self.send_signal(w.Inputs.data, self.iris)
        w.attr_y = w.data.domain[0]
        w._on_attr_changed()
        self.assertTrue(w.Error.same_variable.is_shown())

        self.send_signal(w.Inputs.data, Table("housing"))
        self.assertFalse(w.Error.same_variable.is_shown())


class TestOWPolynomialClassification(WidgetTest):
    # Tests that compute the probability grid and contours, so the code is
    # run at least a few times
    def setUp(self):
        # type: OWPolynomialClassification
        self.widget = self.create_widget(OWPolynomialClassification)
        self.iris = Table.from_file("iris")

    def test_blur_grid(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)
        # here we can check that 0.5 remains same
        assert_array_equal(w.probabilities_grid == 0.5,
                           w.blur_grid(w.probabilities_grid) == 0.5)

    def test_send_report(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        w.send_report()


if __name__ == "__main__":
    unittest.main()
