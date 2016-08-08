from numpy.testing import *
import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.educational.widgets.owgradientdescent import \
    OWGradientDescent


class TestOWGradientDescent(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWGradientDescent)
        self.iris = Table('iris')

    def test_set_data(self):
        """
        Test set data
        """
        w = self.widget

        # test on init
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)

        # call with none data
        self.send_signal("Data", None)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)

        # call with no class variable
        table_no_class = Table(
            Domain([ContinuousVariable("x"), ContinuousVariable("y")]),
            [[1, 2], [2, 3]])
        self.send_signal("Data", table_no_class)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Warning.no_class.is_shown())

        # with only one class value
        table_one_class = Table(
            Domain([ContinuousVariable("x"), ContinuousVariable("y")],
                   DiscreteVariable("a", values=["k"])),
            [[1, 2], [2, 3]], [0, 0])
        self.send_signal("Data", table_one_class)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Warning.no_class.is_shown())

        # not enough continuous variables
        table_no_enough_cont = Table(
            Domain(
                [ContinuousVariable("x"),
                 DiscreteVariable("y", values=["a", "b"])],
                ContinuousVariable("a")),
            [[1, 0], [2, 1]], [0, 0])
        self.send_signal("Data", table_no_enough_cont)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Warning.to_few_features.is_shown())

        # init with ok data
        num_continuous_attributes = sum(
            True for var in self.iris.domain.attributes
            if isinstance(var, ContinuousVariable))

        self.send_signal("Data", self.iris)
        self.assertEqual(w.cbx.count(), num_continuous_attributes)
        self.assertEqual(w.cby.count(), num_continuous_attributes)
        self.assertEqual(
            w.target_class_combobox.count(),
            len(self.iris.domain.class_var.values))
        self.assertEqual(w.cbx.currentText(), self.iris.domain[0].name)
        self.assertEqual(w.cby.currentText(), self.iris.domain[1].name)
        self.assertEqual(
            w.target_class_combobox.currentText(),
            self.iris.domain.class_var.values[0])

        self.assertEqual(w.attr_x, self.iris.domain[0].name)
        self.assertEqual(w.attr_y, self.iris.domain[1].name)
        self.assertEqual(w.target_class, self.iris.domain.class_var.values[0])

        # change showed attributes
        w.attr_x = self.iris.domain[1].name
        w.attr_y = self.iris.domain[2].name
        w.target_class = self.iris.domain.class_var.values[1]

        self.assertEqual(w.cbx.currentText(), self.iris.domain[1].name)
        self.assertEqual(w.cby.currentText(), self.iris.domain[2].name)
        self.assertEqual(
            w.target_class_combobox.currentText(),
            self.iris.domain.class_var.values[1])

        self.assertEqual(w.attr_x, self.iris.domain[1].name)
        self.assertEqual(w.attr_y, self.iris.domain[2].name)
        self.assertEqual(w.target_class, self.iris.domain.class_var.values[1])

        # remove data
        self.send_signal("Data", None)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)

    def test_restart(self):
        """
        Test if restart works fine
        """
        w = self.widget

        # check if init is as expected
        self.assertIsNone(w.selected_data)
        self.assertIsNone(w.learner)

        # with data
        self.send_signal("Data", self.iris)
        self.assertEqual(len(w.selected_data), len(self.iris))
        assert_array_equal(w.learner.x, w.selected_data.X)
        assert_array_equal(w.learner.y, w.selected_data.Y)
        assert_array_equal(w.learner.domain, w.selected_data.domain)
        self.assertEqual(w.learner.alpha, w.alpha)
        self.assertEqual(w.learner.stochastic, False)
        self.assertEqual(w.learner.stochastic_step_size, w.step_size)

        # again no data
        self.send_signal("Data", None)
        self.assertIsNone(w.selected_data)
        self.assertIsNone(w.learner)

    def test_change_alpha(self):
        """
        Function check if alpha is changing correctly
        """
        w = self.widget

        # to define learner
        self.send_signal("Data", self.iris)

        # check init alpha
        self.assertEqual(w.learner.alpha, 0.1)

        # change alpha
        w.alpha_spin.setValue(1)
        self.assertEqual(w.learner.alpha, 1)
        w.alpha_spin.setValue(0.3)
        self.assertEqual(w.learner.alpha, 0.3)

        # just check if nothing happens when no learner
        self.send_signal("Data", None)
        self.assertIsNone(w.learner)
        w.alpha_spin.setValue(5)

    def test_change_stochastic(self):
        """
        Test changing stochastic
        """
        w = self.widget

        # define learner
        self.send_signal("Data", self.iris)

        # check init
        self.assertFalse(w.learner.stochastic)

        # change stochastic
        w.stochastic_checkbox.click()
        self.assertTrue(w.learner.stochastic)
        w.stochastic_checkbox.click()
        self.assertFalse(w.learner.stochastic)

        # just check if nothing happens when no learner
        self.send_signal("Data", None)
        self.assertIsNone(w.learner)
        w.stochastic_checkbox.click()

    def test_change_step(self):
        """
        Function check if change step works correctly
        """
        w = self.widget

        # to define learner
        self.send_signal("Data", self.iris)

        # check init alpha
        self.assertEqual(w.learner.stochastic_step_size, 30)

        # change alpha
        w.step_size_spin.setValue(50)
        self.assertEqual(w.learner.stochastic_step_size, 50)
        w.step_size_spin.setValue(40)
        self.assertEqual(w.learner.stochastic_step_size, 40)

        # just check if nothing happens when no learner
        self.send_signal("Data", None)
        self.assertIsNone(w.learner)
        w.step_size_spin.setValue(40)

    def test_change_theta(self):
        """
        Test setting theta
        """
        w = self.widget

        # to define learner
        self.send_signal("Data", self.iris)

        # check init alpha
        self.assertIsNone(w.learner.theta)

        # change alpha
        w.change_theta(1, 1)
        assert_array_equal(w.learner.theta, [1, 1])
        w.scatter.chart_clicked(1, 2)
        assert_array_equal(w.learner.theta, [1, 2])

        # just check if nothing happens when no learner
        self.send_signal("Data", None)
        self.assertIsNone(w.learner)
        w.change_theta(1, 1)

    def test_step(self):
        """
        Test step
        """
        w = self.widget

        # test function not crashes when no data and learner
        w.step()

        self.send_signal("Data", self.iris)

        # test theta set when none
        self.assertIsNone(w.learner.theta)
        w.step()
        self.assertIsNotNone(w.learner.theta)

        # check theta is changing when step
        old_theta = np.copy(w.learner.theta)
        w.step()
        self.assertNotEqual(sum(old_theta - w.learner.theta), 0)

    def test_step_back(self):
        """
        Test stepping back
        """
        w = self.widget

        # test function not crashes when no data and learner
        w.step_back()

        self.send_signal("Data", self.iris)

        # test step back not performed when step_no == 0
        old_theta = np.copy(w.learner.theta)
        w.step_back()
        assert_array_equal(w.learner.theta, old_theta)

        # test same theta when step performed
        w.change_theta(1.0, 1.0)
        theta = np.copy(w.learner.theta)
        w.step()
        w.step_back()
        assert_array_equal(theta, w.learner.theta)

        w.change_theta(1.0, 1.0)
        theta1 = np.copy(w.learner.theta)
        w.step()
        theta2 = np.copy(w.learner.theta)
        w.step()
        theta3 = np.copy(w.learner.theta)
        w.step()
        w.step_back()
        assert_array_equal(theta3, w.learner.theta)
        w.step_back()
        assert_array_equal(theta2, w.learner.theta)
        w.step_back()
        assert_array_equal(theta1, w.learner.theta)
        w.step_back()
        assert_array_equal(theta1, w.learner.theta)

        # test for stochastic
        w.stochastic_checkbox.click()

        w.change_theta(1.0, 1.0)
        theta = np.copy(w.learner.theta)
        w.step()
        w.step_back()
        assert_array_equal(theta, w.learner.theta)

        w.change_theta(1.0, 1.0)
        theta1 = np.copy(w.learner.theta)
        w.step()
        theta2 = np.copy(w.learner.theta)
        w.step()
        theta3 = np.copy(w.learner.theta)
        w.step()
        w.step_back()
        assert_array_equal(theta3, w.learner.theta)
        w.step_back()
        assert_array_equal(theta2, w.learner.theta)
        w.step_back()
        assert_array_equal(theta1, w.learner.theta)
        w.step_back()
        assert_array_equal(theta1, w.learner.theta)

        # test mix stochastic and normal
        # now it is stochastic

        w.change_theta(1.0, 1.0)
        theta1 = np.copy(w.learner.theta)
        w.step()
        theta2 = np.copy(w.learner.theta)
        w.step()
        w.stochastic_checkbox.click()
        theta3 = np.copy(w.learner.theta)
        w.step()
        w.step_back()
        assert_array_equal(theta3, w.learner.theta)
        w.step_back()
        assert_array_equal(theta2, w.learner.theta)
        w.step_back()
        w.stochastic_checkbox.click()
        assert_array_equal(theta1, w.learner.theta)
        w.step_back()
        assert_array_equal(theta1, w.learner.theta)

    def test_replot(self):
        """
        Test replot function and all functions connected with it
        """
        w = self.widget
        # nothing happens when no data
        w.replot()

        self.assertIsNone(w.cost_grid)
        self.assertEqual(w.scatter.count_replots, 1)

        self.send_signal("Data", self.iris)
        self.assertTupleEqual(w.cost_grid.shape, (w.grid_size, w.grid_size))
        self.assertEqual(w.scatter.count_replots, 2)

        # when step no new re-plots
        w.step()
        self.assertEqual(w.scatter.count_replots, 2)

        # triggered new re-plot
        self.send_signal("Data", self.iris)
        self.assertTupleEqual(w.cost_grid.shape, (w.grid_size, w.grid_size))
        self.assertEqual(w.scatter.count_replots, 3)

    def test_select_data(self):
        """
        Test select data function
        """
        w = self.widget

        # test for none data
        self.send_signal("Data", None)

        self.assertIsNone(w.select_data())  # result is none

        # test on iris
        self.send_signal("Data", self.iris)
        self.assertEqual(len(w.select_data()), len(self.iris))
        self.assertEqual(len(w.select_data().domain.attributes), 2)
        self.assertEqual(len(w.select_data().domain.class_var.values), 2)
        self.assertEqual(w.select_data().domain.class_var.values[1], 'Others')
        self.assertEqual(w.select_data().domain.attributes[0].name, w.attr_x)
        self.assertEqual(w.select_data().domain.attributes[1].name, w.attr_y)
        self.assertEqual(
            w.select_data().domain.class_var.values[0], w.target_class)

    def test_autoplay(self):
        """
        Test autoplay functionalities
        """
        w = self.widget

        # test if not chrashes when data is none
        w.auto_play()

        # set data
        self.send_signal("Data", self.iris)

        # check init
        self.assertFalse(w.auto_play_enabled)
        self.assertEqual(w.auto_play_button.text(), w.auto_play_button_text[0])
        self.assertTrue((w.step_box.isEnabled()))
        self.assertTrue((w.options_box.isEnabled()))
        self.assertTrue((w.properties_box.isEnabled()))

        # auto play on
        w.auto_play()
        self.assertTrue(w.auto_play_enabled)
        self.assertEqual(w.auto_play_button.text(), w.auto_play_button_text[1])
        self.assertFalse((w.step_box.isEnabled()))
        self.assertFalse((w.options_box.isEnabled()))
        self.assertFalse((w.properties_box.isEnabled()))

        # stop auto play
        w.auto_play()
        self.assertFalse(w.auto_play_enabled)
        self.assertEqual(w.auto_play_button.text(), w.auto_play_button_text[0])
        self.assertTrue((w.step_box.isEnabled()))
        self.assertTrue((w.options_box.isEnabled()))
        self.assertTrue((w.properties_box.isEnabled()))

    def test_disable_controls(self):
        """
        Test disabling controls
        """
        w = self.widget

        # check init
        self.assertTrue((w.step_box.isEnabled()))
        self.assertTrue((w.options_box.isEnabled()))
        self.assertTrue((w.properties_box.isEnabled()))

        # disable
        w.disable_controls(True)
        self.assertFalse((w.step_box.isEnabled()))
        self.assertFalse((w.options_box.isEnabled()))
        self.assertFalse((w.properties_box.isEnabled()))

        w.disable_controls(True)
        self.assertFalse((w.step_box.isEnabled()))
        self.assertFalse((w.options_box.isEnabled()))
        self.assertFalse((w.properties_box.isEnabled()))

        # enable
        w.disable_controls(False)
        self.assertTrue((w.step_box.isEnabled()))
        self.assertTrue((w.options_box.isEnabled()))
        self.assertTrue((w.properties_box.isEnabled()))

        w.disable_controls(False)
        self.assertTrue((w.step_box.isEnabled()))
        self.assertTrue((w.options_box.isEnabled()))
        self.assertTrue((w.properties_box.isEnabled()))

    def test_send_model(self):
        """
        Test sending model
        """
        w = self.widget

        # when no learner
        self.assertIsNone(self.get_output("Classifier"))

        # when learner but no theta
        self.send_signal("Data", self.iris)
        self.assertIsNone(self.get_output("Classifier"))

        # when everything fine
        w.change_theta(1., 1.)
        assert_array_equal(self.get_output("Classifier").theta, [1., 1.])

        # when data deleted
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output("Classifier"))

    def test_send_coefficients(self):
        w = self.widget

        # when no learner
        self.assertIsNone(self.get_output("Coefficients"))

        # when learner but no theta
        self.send_signal("Data", self.iris)
        self.assertIsNone(self.get_output("Coefficients"))

        # when everything fine
        w.change_theta(1., 1.)
        coef_out = self.get_output("Coefficients")
        self.assertEqual(len(coef_out), 2)
        self.assertEqual(len(coef_out.domain.attributes), 1)
        self.assertEqual(coef_out.domain.attributes[0].name, "Coefficients")
        self.assertEqual(len(coef_out.domain.metas), 1)
        self.assertEqual(coef_out.domain.metas[0].name, "Name")

        # when data deleted
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output("Coefficients"))

    def test_send_data(self):
        """
        Test sending selected data to output
        """
        w = self.widget

        # when no data
        self.assertIsNone(self.get_output("Data"))

        # when everything fine
        self.send_signal("Data", self.iris)
        w.change_theta(1., 1.)
        assert_array_equal(self.get_output("Data"), w.selected_data)

        # when data deleted
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output("Data"))