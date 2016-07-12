from Orange.widgets.tests.base import GuiTest
from orangecontrib.educational.widgets.owunivariateregression import OWUnivariateRegression
from Orange.data.table import Table
from Orange.regression import LinearRegressionLearner

class TestOWUnivariateRegression(GuiTest):

    def setUp(self):
        self.widget = OWUnivariateRegression()
        self.data = Table("iris")

    def test_set_data(self):
        variables = self.data.domain.variables
        class_variables = self.data.domain.class_vars
        continuous_variables = [var for var in variables if var.is_continuous]

        self.widget.set_data(self.data)

        self.assertEqual(self.data, self.widget.data)
        self.assertEqual(continuous_variables, self.widget.x_var_model._list)
        self.assertEqual(continuous_variables, self.widget.y_var_model._list)
        self.assertEqual(self.widget.x_var_index, 0)
        self.assertEqual(self.widget.y_var_index,
                         len(continuous_variables) - 1
                         if len(class_variables) == 0
                         else len(continuous_variables) - len(class_variables))

    def test_add_main_layout(self):
        self.assertEqual(self.widget.data, None)
        self.assertEqual(self.widget.preprocessors, None)
        self.assertEqual(self.widget.learner, None)
        self.assertEqual(self.widget.scatterplot_item, None)
        self.assertEqual(self.widget.plot_item, None)
        self.assertEqual(self.widget.x_label, 'x')
        self.assertEqual(self.widget.y_label, 'y')

    def test_send_report(self):
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

    def test_clear_plot(self):
        self.widget.set_data(self.data)
        self.widget.apply()
        self.widget.clear_plot()

        self.assertEqual(self.widget.plot_item, None)
        self.assertEqual(self.widget.scatterplot_item, None)

    def test_set_learner(self):
        self.widget.set_learner(LinearRegressionLearner)
        self.assertEqual(self.widget.learner, LinearRegressionLearner)

    def test_plot_scatter_points(self):
        x_data = [1, 2, 3]
        y_data = [2, 3, 4]

        self.widget.plot_scatter_points(x_data, y_data)

        self.assertEqual(self.widget.n_points, len(x_data))
        self.assertNotEqual(self.widget.scatterplot_item, None)

    def test_plot_regression_line(self):
        x_data = [1, 2, 3]
        y_data = [2, 3, 4]

        self.widget.plot_regression_line(x_data, y_data)

        self.assertNotEqual(self.widget.plot_item, None)

    def test_apply(self):
        self.widget.set_data(self.data)
        self.widget.apply()

        self.assertNotEqual(self.widget.plot_item, None)
        self.assertNotEqual(self.widget.scatterplot_item, None)
