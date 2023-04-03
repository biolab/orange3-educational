import unittest

import numpy as np

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.educational.widgets.owpolynomialregression \
    import OWPolynomialRegression, PolynomialFeatures, RegressTo0, \
    TempMeanModel, PolynomialLearnerWrapper
from Orange.regression import (LinearRegressionLearner,
                               RandomForestRegressionLearner)
from Orange.regression.tree import TreeLearner as TreeRegressionLearner
from Orange.preprocess.preprocess import Normalize


class TestOWPolynomialRegression(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPolynomialRegression)   # type: OWPolynomialRegression
        self.data = Table.from_file("iris")
        self.data_housing = Table.from_file("housing")

    def test_set_data(self):
        variables = self.data.domain.variables
        class_variables = self.data.domain.class_vars
        continuous_variables = [var for var in variables if var.is_continuous]

        self.widget.set_data(self.data)

        self.assertEqual(self.data, self.widget.data)
        self.assertEqual(continuous_variables, list(self.widget.var_model))
        self.assertEqual(self.widget.x_var, continuous_variables[0])
        self.assertEqual(self.widget.y_var, continuous_variables[1])

        # check for none data
        self.widget.set_data(None)
        self.assertEqual(self.widget.data, None)

        # data set with continuous class to check if nclass > 0
        variables = self.data_housing.domain.variables
        continuous_variables = [var for var in variables if var.is_continuous]

        self.widget.set_data(self.data_housing)
        self.assertEqual(self.data_housing, self.widget.data)
        self.assertEqual(continuous_variables, list(self.widget.var_model))
        self.assertEqual(self.widget.x_var, continuous_variables[0])
        self.assertEqual(self.widget.y_var, self.data_housing.domain.class_var)

        # check with data with all none
        data = Table.from_list(
            Domain([ContinuousVariable('a'),
                    ContinuousVariable('b')]),
            [[None, None], [None, None]])
        self.widget.set_data(data)
        self.widget.apply()
        self.assertIsNone(self.widget.plot_item)
        self.assertIsNone(self.widget.scatterplot_item)

    def test_add_main_layout(self):
        w = self.widget
        self.assertEqual(w.data, None)
        self.assertEqual(w.preprocessors, None)
        self.assertEqual(w.learner, None)
        self.assertEqual(w.scatterplot_item, None)
        self.assertEqual(w.plot_item, None)

        self.assertEqual(
            w.controls.regressor_name.text(), "Regressor: Linear Regression")

    def test_send_report(self):
        # check if nothing happens when polynomialexpansion is None
        self.widget.polynomialexpansion = None
        self.widget.send_report()
        self.assertNotIn("class='caption'", self.widget.report_html)
        self.widget.polynomialexpansion = 1

        self.widget.send_report()
        self.assertEqual(self.widget.report_html, "")

        self.widget.set_data(self.data)
        self.widget.send_report()

        self.assertNotEqual(self.widget.report_html, "")

    def test_clear(self):
        self.widget.set_data(self.data)
        self.widget.clear()

        self.assertEqual(self.widget.data, None)
        #just check if clear function also call clear_plot
        self.assertEqual(self.widget.plot_item, None)
        self.assertEqual(self.widget.scatterplot_item, None)
        self.assertEqual(len(self.widget.error_plot_items), 0)

    def test_clear_plot(self):
        self.widget.set_data(self.data)
        self.widget.apply()
        self.widget.clear_plot()

        self.assertEqual(self.widget.plot_item, None)
        self.assertEqual(self.widget.scatterplot_item, None)

    def test_set_learner(self):
        w = self.widget

        lin = LinearRegressionLearner
        lin.name = "Linear Regression"
        self.widget.set_learner(lin)
        self.assertEqual(self.widget.learner, lin)

        self.assertEqual(
            w.controls.regressor_name.text(), "Regressor: Linear Regression")

        tree = TreeRegressionLearner
        tree.name = "Tree Learner"

        self.widget.set_learner(tree)
        self.assertEqual(self.widget.learner, tree)
        self.assertEqual(
            w.controls.regressor_name.text(), "Regressor: Tree Learner")

    def test_plot_scatter_points(self):
        x_data = [1, 2, 3]
        y_data = [2, 3, 4]

        self.widget.plot_scatter_points(x_data, y_data)

        self.assertNotEqual(self.widget.scatterplot_item, None)

        # check case when scatter plot allready exist
        x_data = [2, 3, 5]
        y_data = [2, 3, 10]

        self.widget.plot_scatter_points(x_data, y_data)

        self.assertNotEqual(self.widget.scatterplot_item, None)

    def test_plot_regression_line(self):
        x_data = [1, 2, 3]
        y_data = [2, 3, 4]

        self.widget.plot_regression_line(x_data, y_data)

        self.assertNotEqual(self.widget.plot_item, None)

        # check case when regression_line already exist
        x_data = [2, 3, 5]
        y_data = [2, 3, 10]

        self.widget.plot_regression_line(x_data, y_data)

        self.assertNotEqual(self.widget.plot_item, None)

    def test_plot_error_bars(self):
        w = self.widget

        check = w.controls.error_bars_enabled
        check.click()

        x_data = [1, 2, 3]
        y_data = [2, 3, 4]
        y_data_fake = [1, 2, 4]

        self.widget.plot_error_bars(x_data, y_data, y_data_fake)
        self.assertEqual(len(w.error_plot_items), len(x_data))

        check.click()

        self.widget.plot_error_bars(x_data, y_data, y_data_fake)
        self.assertEqual(len(w.error_plot_items), 0)

        check.click()

        self.send_signal(w.Inputs.data, self.data)
        self.assertEqual(len(w.error_plot_items), len(self.data))

    def test_apply(self):
        self.widget.set_data(self.data)
        self.widget.apply()

        self.assertNotEqual(self.widget.plot_item, None)
        self.assertNotEqual(self.widget.scatterplot_item, None)

        self.widget.set_data(None)
        self.widget.apply()
        # TODO: output will be checked when it available in GuiTest

        # check if function does not change plots that are None according to test_set_data
        self.assertEqual(self.widget.plot_item, None)
        self.assertEqual(self.widget.scatterplot_item, None)

        self.widget.set_data(self.data)
        self.widget.set_learner(LinearRegressionLearner())
        self.widget.apply()

        self.assertNotEqual(self.widget.plot_item, None)
        self.assertNotEqual(self.widget.scatterplot_item, None)

        self.widget.set_learner(RandomForestRegressionLearner())
        self.widget.apply()

        self.assertNotEqual(self.widget.plot_item, None)
        self.assertNotEqual(self.widget.scatterplot_item, None)

        self.widget.set_preprocessor(Normalize())
        self.assertNotEqual(self.widget.plot_item, None)
        self.assertNotEqual(self.widget.scatterplot_item, None)

    def test_data_output(self):
        """
        Check if correct data on output
        """
        w = self.widget
        spin =w.controls.polynomialexpansion

        self.assertIsNone(self.get_output(w.Outputs.data))

        u, x, y, z = (ContinuousVariable(n) for n in "uxyz")
        domain = Domain([u, x], y)
        data = Table.from_numpy(
            domain,
            [[1, 1], [0, 2], [np.nan, 3], [-1, np.nan], [2, 4]],
            [3, 5, 7, 7, np.nan])

        spin.setValue(0)
        self.send_signal(w.Inputs.data, data)
        w.x_var = x
        w.y_var = y

        w.fit_intercept = False
        spin.setValue(1)
        np.testing.assert_almost_equal(
            self.get_output(w.Outputs.data).X.T,
            [[1, 2, 3]])

        spin.setValue(2)
        np.testing.assert_almost_equal(
            self.get_output(w.Outputs.data).X.T,
            [[1, 2, 3], [1, 4, 9]])

        spin.setValue(3)
        np.testing.assert_almost_equal(
            self.get_output(w.Outputs.data).X.T,
            [[1, 2, 3], [1, 4, 9], [1, 8, 27]])

        w.fit_intercept = True
        spin.setValue(1)
        np.testing.assert_almost_equal(
            self.get_output(w.Outputs.data).X.T,
            [[1, 1, 1], [1, 2, 3]])

        spin.setValue(2)
        np.testing.assert_almost_equal(
            self.get_output(w.Outputs.data).X.T,
            [[1, 1, 1], [1, 2, 3], [1, 4, 9]])

        spin.setValue(3)
        np.testing.assert_almost_equal(
            self.get_output(w.Outputs.data).X.T,
            [[1, 1, 1], [1, 2, 3], [1, 4, 9], [1, 8, 27]])

        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.data))

    def test_data_nan_row(self):
        """
        When some rows are nan in attributes array widget crashes.
        GH-43
        """
        data = self.data[::50].copy()
        with data.unlocked(data.X):
            data.X[0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)

    def test_coefficients(self):
        w = self.widget

        u, x, y, z = (ContinuousVariable(n) for n in "uxyz")
        domain = Domain([u, x], y)
        data = Table.from_numpy(
            domain,
            [[1, 1], [0, 2], [np.nan, 3], [-1, np.nan], [2, 4]],
            [3, 5, 7, 7, np.nan])

        self.send_signal(w.Inputs.data, data)
        w.x_var = x
        w.y_var = y

        spin = self.widget.controls.polynomialexpansion
        intercept_cb = self.widget.controls.fit_intercept
        intercept_cb.setChecked(True)

        spin.setValue(0)
        coef = self.get_output(w.Outputs.coefficients)
        np.testing.assert_almost_equal(coef.X, [[5]])

        spin.setValue(1)
        coef = self.get_output(w.Outputs.coefficients)
        np.testing.assert_almost_equal(coef.X.T, [[1, 2]])

        spin.setValue(2)
        coef = self.get_output(w.Outputs.coefficients)
        np.testing.assert_almost_equal(coef.X.T, [[1, 2, 0]])

        intercept_cb.setChecked(False)
        spin.setValue(0)
        coef = self.get_output(w.Outputs.coefficients)
        self.assertIsNone(coef)

        spin.setValue(1)
        coef = self.get_output(w.Outputs.coefficients)
        # I haven't computed this value manually, I just copied it
        np.testing.assert_almost_equal(coef.X.T, [[2.4285714]])

        spin.setValue(2)
        coef = self.get_output(w.Outputs.coefficients)
        # I haven't computed these values manually, I just copied them
        np.testing.assert_almost_equal(coef.X.T, [[ 3.1052632, -0.2631579]])

        self.send_signal(w.Inputs.learner, LinearRegressionLearner(fit_intercept=True))

        spin.setValue(0)
        coef = self.get_output(w.Outputs.coefficients)
        np.testing.assert_almost_equal(coef.X.T, [[5]])

        spin.setValue(1)
        coef = self.get_output(w.Outputs.coefficients)
        np.testing.assert_almost_equal(coef.X.T, [[1, 2]])

        spin.setValue(2)
        coef = self.get_output(w.Outputs.coefficients)
        np.testing.assert_almost_equal(coef.X.T, [[1, 2, 0]])

        self.send_signal(w.Inputs.learner, LinearRegressionLearner(fit_intercept=False))

        spin.setValue(0)
        coef = self.get_output(w.Outputs.coefficients)
        self.assertIsNone(coef)

        spin.setValue(1)
        coef = self.get_output(w.Outputs.coefficients)
        # I haven't computed this value manually, I just copied it
        np.testing.assert_almost_equal(coef.X.T, [[2.4285714]])

        spin.setValue(2)
        coef = self.get_output(w.Outputs.coefficients)
        # I haven't computed these values manually, I just copied them
        np.testing.assert_almost_equal(coef.X.T, [[ 3.1052632, -0.2631579]])

    def test_large_polynomial_values(self):
        x, y = (ContinuousVariable(n) for n in "xy")
        data = Table.from_numpy(Domain([x], y), [[1], [30]], [3, 5])
        self.send_signal(self.widget.Inputs.data, data)

        self.widget.controls.polynomialexpansion.setValue(9)
        self.assertFalse(self.widget.Warning.large_diffs.is_shown())
        self.widget.controls.polynomialexpansion.setValue(10)
        self.assertTrue(self.widget.Warning.large_diffs.is_shown())
        self.widget.controls.polynomialexpansion.setValue(9)
        self.assertFalse(self.widget.Warning.large_diffs.is_shown())


class PolynomialFeaturesTest(unittest.TestCase):
    def test_1d(self):
        x, y, z = (ContinuousVariable(n) for n in "xyz")
        domain = Domain([x], y, [z])
        data = Table.from_numpy(
            domain,
            [[1], [2], [3]], [1, 2, 3], [[1], [2], [3]])
        data2 = Table.from_numpy(
            domain,
            [[3], [4], [5]], [1, 2, 3], [[1], [2], [3]])
        tf = PolynomialFeatures(1, False)(data)
        self.assertIs(tf.domain.class_var, y)
        np.testing.assert_equal(tf.Y, [1, 2, 3])
        self.assertEqual(tf.domain.metas, (z, ))
        np.testing.assert_equal(tf.metas.T, [[1, 2, 3]])

        np.testing.assert_equal(tf.X.T, [[1, 2, 3]])
        np.testing.assert_equal(data2.transform(tf.domain).X.T, [[3, 4, 5]])

        tf = PolynomialFeatures(1, True)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 1, 1], [1, 2, 3]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[1, 1, 1], [3, 4, 5]])

        tf = PolynomialFeatures(2, True)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 1, 1], [1, 2, 3], [1, 4, 9]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[1, 1, 1], [3, 4, 5], [9, 16, 25]])

        tf = PolynomialFeatures(2, False)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 2, 3], [1, 4, 9]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[3, 4, 5], [9, 16, 25]])

        tf = PolynomialFeatures(3, True)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 1, 1], [1, 2, 3], [1, 4, 9], [1, 8, 27]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[1, 1, 1], [3, 4, 5], [9, 16, 25], [27, 64, 125]])

        tf = PolynomialFeatures(3, False)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 2, 3], [1, 4, 9], [1, 8, 27]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[3, 4, 5], [9, 16, 25], [27, 64, 125]])

        tf = PolynomialFeatures(0, True)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 1, 1]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[1, 1, 1]])

        tf = PolynomialFeatures(0, False)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[0, 0, 0]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[0, 0, 0]])

    def test_nd(self):
        x, y, z = (ContinuousVariable(n) for n in "xyz")
        domain = Domain([x, y, z])
        data = Table.from_numpy(
            domain,
            [[1, 2, 3], [4, 5, 6]])
        data2 = Table.from_numpy(
            domain,
            [[1, 3, 5]])

        tf = PolynomialFeatures(1, False)(data)
        np.testing.assert_equal(tf.X, data.X)
        np.testing.assert_equal(data2.transform(tf.domain).X, data2.X)

        tf = PolynomialFeatures(1, True)(data)
        np.testing.assert_equal(tf.X, [[1, 1, 2, 3], [1, 4, 5, 6]])
        np.testing.assert_equal(data2.transform(tf.domain).X, [[1, 1, 3, 5]])

        tf = PolynomialFeatures(2, False)(data)
        np.testing.assert_equal(
            tf.X,
            [[1, 2, 3, 1, 2, 3, 4, 6, 9],
             [4, 5, 6, 16, 20, 24, 25, 30, 36]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X,
            [[1, 3, 5, 1, 3, 5, 9, 15, 25]])

        tf = PolynomialFeatures(2, True)(data)
        np.testing.assert_equal(
            tf.X,
            [[1, 1, 2, 3, 1, 2, 3, 4, 6, 9],
             [1, 4, 5, 6, 16, 20, 24, 25, 30, 36]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X,
            [[1, 1, 3, 5, 1, 3, 5, 9, 15, 25]])

        tf = PolynomialFeatures(0, True)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[1, 1]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[1]])

        tf = PolynomialFeatures(0, False)(data)
        np.testing.assert_equal(
            tf.X.T,
            [[0, 0]])
        np.testing.assert_equal(
            data2.transform(tf.domain).X.T,
            [[0]])


class ModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Table("iris")[:10]

    def test_regressto0(self):
        model = RegressTo0()(self.data)
        prediction = model(self.data[:5])
        np.testing.assert_equal(prediction, [0] * 5)

    def test_tempmeanmodel(self):
        model = TempMeanModel(42)
        prediction = model(self.data[:5])
        np.testing.assert_equal(prediction, [42] * 5)

    def test_polynomiallearnerwrapper(self):
        u, x, y, z = (ContinuousVariable(n) for n in "uxyz")
        domain = Domain([u, x], y)
        data = Table.from_numpy(
            domain,
            [[1, 1], [0, 2], [np.nan, 3], [-1, np.nan], [2, 4]],
            [3, 5, 7, 7, np.nan])
        data2 = Table.from_numpy(
            Domain([x, z]),
            [[5, 6], [7, 8]])

        learner = PolynomialLearnerWrapper(
            x, y, 1, LinearRegressionLearner(fit_intercept=False),
            fit_intercept=True, preprocessors=None)
        model = learner(data)
        np.testing.assert_almost_equal(model.coefficients, [1, 2])
        np.testing.assert_almost_equal(model(data2), [11, 15])

        learner = PolynomialLearnerWrapper(
            x, y, 1, LinearRegressionLearner(fit_intercept=True),
            fit_intercept=False, preprocessors=False)
        model = learner(data)
        np.testing.assert_almost_equal(model.coefficients, [2])
        self.assertAlmostEqual(model.intercept, 1)
        np.testing.assert_almost_equal(model(data2), [11, 15])

        learner = PolynomialLearnerWrapper(
            x, y, 1, LinearRegressionLearner(fit_intercept=False),
            fit_intercept=False, preprocessors=None)
        model = learner(data)
        # I haven't computed this value manually, just copied the result
        # But it must be something a bit larger than 2, hence...
        np.testing.assert_almost_equal(model.coefficients, [2.4285714])
        self.assertEqual(model.intercept, 0)
        np.testing.assert_almost_equal(model(data2), [12.1428571, 17])

        learner = PolynomialLearnerWrapper(
            x, y, 2, LinearRegressionLearner(fit_intercept=False),
            fit_intercept=False, preprocessors=False)
        model = learner(data)
        # I haven't computed this value manually, just copied the result
        np.testing.assert_almost_equal(model.coefficients,
                                       [3.1052632, -0.2631579])
        self.assertEqual(model.intercept, 0)
        np.testing.assert_almost_equal(model(data2), [8.9473684, 8.8421053])


if __name__ == "__main__":
    unittest.main()
