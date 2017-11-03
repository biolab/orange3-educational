import unittest

from numpy.testing import *
import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest

from AnyQt.QtCore import Qt, QEvent
from AnyQt.QtGui import QKeyEvent

from orangecontrib.educational.widgets.owgradientdescent import \
    OWGradientDescent


class TestOWGradientDescent(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWGradientDescent)  # type: OWGradientDescent
        self.iris = Table('iris')
        self.housing = Table('housing')

    @unittest.skip("Travis fails: TimeoutError")
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
        self.send_signal(w.Inputs.data, None)
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
        self.send_signal(w.Inputs.data, table_no_class)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Error.no_class.is_shown())

        # with only one class value
        table_one_class = Table(
            Domain([ContinuousVariable("x"), ContinuousVariable("y")],
                   DiscreteVariable("a", values=["k"])),
            [[1, 2], [2, 3]], [0, 0])
        self.send_signal(w.Inputs.data, table_one_class)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Error.to_few_values.is_shown())

        # not enough continuous variables
        table_no_enough_cont = Table(
            Domain(
                [ContinuousVariable("x"),
                 DiscreteVariable("y", values=["a", "b"])],
                DiscreteVariable("a", values=['a', 'b'])),
            [[1, 0], [2, 1]], [0, 1])
        self.send_signal(w.Inputs.data, table_no_enough_cont)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Error.to_few_features.is_shown())

        # init with ok data, discrete class - logistic regression
        num_continuous_attributes = sum(
            True for var in self.iris.domain.attributes
            if isinstance(var, ContinuousVariable))

        self.send_signal(w.Inputs.data, self.iris)
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
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)

        # not enough continuous variables when continuous class
        table_no_enough_cont = Table(
            Domain(
                [DiscreteVariable("y", values=["a", "b"])],
                ContinuousVariable("a")),
            [[1, 0], [2, 1]], [0, 1])
        self.send_signal(w.Inputs.data, table_no_enough_cont)
        self.assertIsNone(w.data)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertIsNone(w.learner)
        self.assertIsNone(w.cost_grid)
        self.assertTrue(w.Error.to_few_features.is_shown())

        # init with ok data, discrete class - linear regression
        num_continuous_attributes = sum(
            True for var in self.housing.domain.attributes
            if isinstance(var, ContinuousVariable))

        self.send_signal(w.Inputs.data, self.housing)
        self.assertEqual(w.cbx.count(), num_continuous_attributes)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertFalse(w.cby.isEnabled())
        self.assertFalse(w.target_class_combobox.isEnabled())
        self.assertEqual(w.cbx.currentText(), self.housing.domain[0].name)

        self.assertEqual(w.attr_x, self.housing.domain[0].name)

        # change showed attributes
        w.attr_x = self.housing.domain[1].name

        self.assertEqual(w.cbx.currentText(), self.housing.domain[1].name)

        self.assertEqual(w.attr_x, self.housing.domain[1].name)

    @unittest.skip("Travis fails: TimeoutError")
    def test_restart(self):
        """
        Test if restart works fine
        """
        w = self.widget

        # check if init is as expected
        self.assertIsNone(w.selected_data)
        self.assertIsNone(w.learner)

        # with logistic regression
        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual(len(w.selected_data), len(self.iris))
        assert_array_equal(w.learner.x, w.selected_data.X)
        assert_array_equal(w.learner.y, w.selected_data.Y)
        assert_array_equal(w.learner.domain, w.selected_data.domain)
        self.assertEqual(w.learner.alpha, w.alpha)
        self.assertEqual(w.learner.stochastic, False)
        self.assertEqual(w.learner.stochastic_step_size, w.step_size)

        # with linear regression
        self.send_signal(w.Inputs.data, self.housing)
        self.assertEqual(len(w.selected_data), len(self.housing))
        assert_array_equal(w.learner.x[:, 1][:, None], w.selected_data.X)
        # because of intercept
        assert_array_equal(w.learner.y, w.selected_data.Y)
        assert_array_equal(w.learner.domain, w.selected_data.domain)
        self.assertEqual(w.learner.alpha, w.alpha)
        self.assertEqual(w.learner.stochastic, False)
        self.assertEqual(w.learner.stochastic_step_size, w.step_size)

        # click on restart
        old_theta = np.copy(w.learner.theta)
        w.restart_button.click()
        assert_array_equal(w.learner.theta, old_theta)

        # again no data
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.selected_data)
        self.assertIsNone(w.learner)

    def test_change_alpha(self):
        """
        Function check if alpha is changing correctly
        """
        w = self.widget

        # to define learner
        self.send_signal(w.Inputs.data, self.iris)

        # check init alpha
        self.assertEqual(w.learner.alpha, 0.1)

        # change alpha
        w.alpha_spin.setValue(1)
        self.assertEqual(w.learner.alpha, 1)
        w.alpha_spin.setValue(0.3)
        self.assertEqual(w.learner.alpha, 0.3)

        # just check if nothing happens when no learner
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.learner)
        w.alpha_spin.setValue(5)

    def test_change_stochastic(self):
        """
        Test changing stochastic
        """
        w = self.widget

        # define learner
        self.send_signal(w.Inputs.data, self.iris)

        # check init
        self.assertFalse(w.learner.stochastic)

        # change stochastic
        w.stochastic_checkbox.click()
        self.assertTrue(w.learner.stochastic)
        w.stochastic_checkbox.click()
        self.assertFalse(w.learner.stochastic)

        # just check if nothing happens when no learner
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.learner)
        w.stochastic_checkbox.click()

    def test_change_step(self):
        """
        Function check if change step works correctly
        """
        w = self.widget

        # to define learner
        self.send_signal(w.Inputs.data, self.iris)

        # check init alpha
        self.assertEqual(w.learner.stochastic_step_size, 30)

        # change alpha
        w.step_size_spin.setValue(50)
        self.assertEqual(w.learner.stochastic_step_size, 50)
        w.step_size_spin.setValue(40)
        self.assertEqual(w.learner.stochastic_step_size, 40)

        # just check if nothing happens when no learner
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.learner)
        w.step_size_spin.setValue(30)

    def test_change_theta(self):
        """
        Test setting theta
        """
        w = self.widget

        # to define learner
        self.send_signal(w.Inputs.data, self.iris)

        # check init theta
        self.assertIsNotNone(w.learner.theta)

        # change theta
        w.change_theta(1, 1)
        assert_array_equal(w.learner.theta, [1, 1])
        w.scatter.bridge.chart_clicked(1, 2)
        assert_array_equal(w.learner.theta, [1, 2])

        # just check if nothing happens when no learner
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.learner)
        w.change_theta(1, 1)

    @unittest.skip("Travis fails: TimeoutError")
    def test_step(self):
        """
        Test step
        """
        w = self.widget

        # test function not crashes when no data and learner
        w.step()

        self.send_signal(w.Inputs.data, self.iris)

        # test theta set after step if not set yet
        w.step()
        self.assertIsNotNone(w.learner.theta)

        # check theta is changing when step
        old_theta = np.copy(w.learner.theta)
        w.step()
        self.assertNotEqual(sum(old_theta - w.learner.theta), 0)

        # with linear regression
        self.send_signal(w.Inputs.data, self.housing)

        # test theta set after step if not set yet
        w.step()
        self.assertIsNotNone(w.learner.theta)

        # check theta is changing when step
        old_theta = np.copy(w.learner.theta)
        w.step()
        self.assertNotEqual(sum(old_theta - w.learner.theta), 0)

        # check steps not allowed after 500
        for i in range(500):
            w.step()
        self.assertEqual(w.learner.step_no, 501)

    @unittest.skip("Travis fails: TimeoutError")
    def test_step_space(self):
        """
        Test step
        """
        w = self.widget

        event = QKeyEvent(
            QEvent.KeyPress, Qt.Key_Space, Qt.KeyboardModifiers(0))

        # test function not crashes when no data and learner
        w.keyPressEvent(event)

        self.send_signal(w.Inputs.data, self.iris)

        # test theta set after step if not set yet
        w.keyPressEvent(event)
        self.assertIsNotNone(w.learner.theta)

        # check theta is changing when step
        old_theta = np.copy(w.learner.theta)
        w.keyPressEvent(event)
        self.assertNotEqual(sum(old_theta - w.learner.theta), 0)

        # with linear regression
        self.send_signal(w.Inputs.data, self.housing)

        # test theta set after step if not set yet
        w.keyPressEvent(event)
        self.assertIsNotNone(w.learner.theta)

        # check theta is changing when step
        old_theta = np.copy(w.learner.theta)
        w.keyPressEvent(event)
        self.assertNotEqual(sum(old_theta - w.learner.theta), 0)

        old_theta = np.copy(w.learner.theta)
        # to cover else example and check not crashes
        event = QKeyEvent(
            QEvent.KeyPress, Qt.Key_Q, Qt.KeyboardModifiers(0))
        w.keyPressEvent(event)

        # check nothing changes
        assert_array_equal(old_theta, w.learner.theta)

    @unittest.skip("Travis fails: TimeoutError")
    def test_step_back(self):
        """
        Test stepping back
        """
        w = self.widget

        # test function not crashes when no data and learner
        w.step_back()

        self.send_signal(w.Inputs.data, self.iris)

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

        # with linear regression
        self.send_signal(w.Inputs.data, self.housing)

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

    @unittest.skip("Travis fails: TimeoutError")
    def test_replot(self):
        """
        Test replot function and all functions connected with it
        """
        w = self.widget
        # nothing happens when no data
        w.replot()

        self.assertIsNone(w.cost_grid)
        self.assertEqual(w.scatter.count_replots, 1)

        self.send_signal(w.Inputs.data, self.iris)
        self.assertTupleEqual(w.cost_grid.shape, (w.grid_size, w.grid_size))
        self.assertEqual(w.scatter.count_replots, 2)

        # when step no new re-plots
        w.step()
        self.assertEqual(w.scatter.count_replots, 2)

        # triggered new re-plot
        self.send_signal(w.Inputs.data, self.iris)
        self.assertTupleEqual(w.cost_grid.shape, (w.grid_size, w.grid_size))
        self.assertEqual(w.scatter.count_replots, 3)

        # with linear regression
        self.send_signal(w.Inputs.data, self.housing)
        self.assertTupleEqual(w.cost_grid.shape, (w.grid_size, w.grid_size))
        self.assertEqual(w.scatter.count_replots, 4)

        # when step no new re-plots
        w.step()
        self.assertEqual(w.scatter.count_replots, 4)

        # triggered new re-plot
        self.send_signal(w.Inputs.data, self.iris)
        self.assertTupleEqual(w.cost_grid.shape, (w.grid_size, w.grid_size))
        self.assertEqual(w.scatter.count_replots, 5)

    @unittest.skip("Travis fails: TimeoutError")
    def test_select_data(self):
        """
        Test select data function
        """
        w = self.widget

        # test for none data
        self.send_signal(w.Inputs.data, None)

        self.assertIsNone(w.select_data())  # result is none

        # test on iris
        self.send_signal(w.Inputs.data, self.iris)
        self.assertEqual(len(w.select_data()), len(self.iris))
        self.assertEqual(len(w.select_data().domain.attributes), 2)
        self.assertEqual(len(w.select_data().domain.class_var.values), 2)
        self.assertEqual(w.select_data().domain.class_var.values[1], 'Others')
        self.assertEqual(w.select_data().domain.attributes[0].name, w.attr_x)
        self.assertEqual(w.select_data().domain.attributes[1].name, w.attr_y)
        self.assertEqual(
            w.select_data().domain.class_var.values[0], w.target_class)

        # test on housing - continuous class
        self.send_signal(w.Inputs.data, self.housing)
        self.assertEqual(len(w.select_data()), len(self.housing))
        self.assertEqual(len(w.select_data().domain.attributes), 1)
        self.assertEqual(w.select_data().domain.attributes[0].name, w.attr_x)
        self.assertTrue(w.select_data().domain.class_var.is_continuous)

        # test with data set for logistic regression - class discrete
        # there no other class value is provided
        domain = Domain([ContinuousVariable('a'), ContinuousVariable('b')],
                        DiscreteVariable('c', values=['a', 'b']))
        data = Table(domain, [[1, 2], [1, 2]], [0, 1])

        self.send_signal(w.Inputs.data, data)
        self.assertEqual(len(w.select_data()), len(data))
        self.assertEqual(len(w.select_data().domain.attributes), 2)
        self.assertEqual(len(w.select_data().domain.class_var.values), 2)
        self.assertEqual(
            w.select_data().domain.class_var.values[1],
            data.domain.class_var.values[1])
        self.assertEqual(
            w.select_data().domain.class_var.values[0],
            data.domain.class_var.values[0])
        self.assertEqual(w.select_data().domain.attributes[0].name, w.attr_x)
        self.assertEqual(w.select_data().domain.attributes[1].name, w.attr_y)
        self.assertEqual(
            w.select_data().domain.class_var.values[0], w.target_class)

        # selected data none when one column only Nones
        data = Table(Domain([ContinuousVariable('a'), ContinuousVariable('b')],
                            DiscreteVariable('c', values=['a', 'b'])),
                     [[1, None], [1, None]], [0, 1])
        self.send_signal(w.Inputs.data, data)
        selected_data = w.select_data()
        self.assertIsNone(selected_data)

        data = Table(Domain([ContinuousVariable('a'), ContinuousVariable('b')],
                            DiscreteVariable('c', values=['a', 'b'])),
                     [[None, None], [None, None]], [0, 1])
        self.send_signal(w.Inputs.data, data)
        selected_data = w.select_data()
        self.assertIsNone(selected_data)

    def test_autoplay(self):
        """
        Test autoplay functionalities
        """
        w = self.widget

        # test if not chrashes when data is none
        w.auto_play()

        # set data
        self.send_signal(w.Inputs.data, self.iris)

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

    @unittest.skip("Travis fails: TimeoutError")
    def test_send_model(self):
        """
        Test sending model
        """
        w = self.widget

        # when no learner
        self.assertIsNone(self.get_output(w.Outputs.model))

        # when learner theta set automatically
        self.send_signal(w.Inputs.data, self.iris)
        self.assertIsNotNone(self.get_output(w.Outputs.model))

        # when everything fine
        w.change_theta(1., 1.)
        assert_array_equal(self.get_output(w.Outputs.model).theta, [1., 1.])

        # when data deleted
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.model))

        # when learner theta set automatically
        self.send_signal(w.Inputs.data, self.housing)
        self.assertIsNotNone(self.get_output(w.Outputs.model))

        # when everything fine
        w.change_theta(1., 1.)
        assert_array_equal(self.get_output(w.Outputs.model).theta, [1., 1.])

        # when data deleted
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.model))

    @unittest.skip("Travis fails: TimeoutError")
    def test_send_coefficients(self):
        w = self.widget

        # when no learner
        self.assertIsNone(self.get_output(w.Outputs.coefficients))

        # when learner but no theta
        self.send_signal(w.Inputs.data, self.iris)
        self.assertIsNotNone(self.get_output(w.Outputs.coefficients))

        # when everything fine
        w.change_theta(1., 1.)
        coef_out = self.get_output(w.Outputs.coefficients)
        self.assertEqual(len(coef_out), 2)
        self.assertEqual(len(coef_out.domain.attributes), 1)
        self.assertEqual(coef_out.domain.attributes[0].name, "Coefficients")
        self.assertEqual(len(coef_out.domain.metas), 1)
        self.assertEqual(coef_out.domain.metas[0].name, "Name")

        # when data deleted
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.coefficients))

        # for linear regression
        # when learner but no theta
        self.send_signal(w.Inputs.data, self.housing)
        self.assertIsNotNone(self.get_output(w.Outputs.coefficients))

        # when everything fine
        w.change_theta(1., 1.)
        coef_out = self.get_output(w.Outputs.coefficients)
        self.assertEqual(len(coef_out), 2)
        self.assertEqual(len(coef_out.domain.attributes), 1)
        self.assertEqual(coef_out.domain.attributes[0].name, "Coefficients")
        self.assertEqual(len(coef_out.domain.metas), 1)
        self.assertEqual(coef_out.domain.metas[0].name, "Name")

    def test_send_data(self):
        """
        Test sending selected data to output
        """
        w = self.widget

        # when no data
        self.assertIsNone(self.get_output(w.Outputs.data))

        # when everything fine
        self.send_signal(w.Inputs.data, self.iris)
        w.change_theta(1., 1.)
        assert_array_equal(self.get_output(w.Outputs.data), w.selected_data)

        # when data deleted
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.data))

    def test_change_attributes(self):
        """
        Check if reset is ok
        """
        w = self.widget

        # when everything fine
        self.send_signal(w.Inputs.data, self.iris)

        w.change_attributes()
        self.assertIsNotNone(w.learner)
        self.assertIsNotNone(w.learner.theta)

    def test_send_report(self):
        """
        Test if report does not chrashes
        """
        w = self.widget

        def _svg_ready():
            return getattr(w, w.graph_name).svg()

        # when everything fine
        self.send_signal(w.Inputs.data, self.iris)
        self.process_events(_svg_ready)
        w.send_report()

        # when no data
        self.send_signal(w.Inputs.data, None)
        self.process_events(_svg_ready)
        w.send_report()

        # for stochastic
        w.stochastic_checkbox.click()
        self.send_signal(w.Inputs.data, self.iris)
        self.process_events(_svg_ready)
        w.send_report()
        self.process_events()
