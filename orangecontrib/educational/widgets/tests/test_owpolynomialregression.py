import numpy as np

from Orange.data import Table, Domain, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.educational.widgets.owpolynomialregression \
    import OWPolynomialRegression
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

        self.widget.set_preprocessor((Normalize(),))
        self.assertNotEqual(self.widget.plot_item, None)
        self.assertNotEqual(self.widget.scatterplot_item, None)

    def test_data_output(self):
        """
        Check if correct data on output
        """
        w = self.widget

        self.assertIsNone(self.get_output(w.Outputs.data))
        self.widget.set_data(self.data)
        spin = self.widget.controls.polynomialexpansion
        spin.setValue(1)
        self.widget.send_data()
        self.assertEqual(len(self.get_output(w.Outputs.data).domain.attributes), 2)

        spin.setValue(2)
        self.widget.send_data()
        self.assertEqual(len(self.get_output(w.Outputs.data).domain.attributes), 3)

        spin.setValue(3)
        self.widget.send_data()
        self.assertEqual(len(self.get_output(w.Outputs.data).domain.attributes), 4)

        spin.setValue(4)
        self.widget.send_data()
        self.assertEqual(len(self.get_output(w.Outputs.data).domain.attributes), 5)

        self.widget.set_data(None)
        self.widget.send_data()
        self.assertIsNone(self.get_output(w.Outputs.data))

    def test_data_nan_row(self):
        """
        When some rows are nan in attributes array widget crashes.
        GH-43
        """
        data = self.data[::50]
        data.X[0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)


if __name__ == "__main__":
    import unittest
    unittest.main()
