from functools import reduce

from numpy.testing import assert_array_equal

from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.classification import (
    LogisticRegressionLearner,
    TreeLearner,
    RandomForestLearner)
from Orange.preprocess.preprocess import Continuize, Discretize
from orangecontrib.educational.widgets.owpolynomialclassification import \
    OWPolynomialClassification
from orangecontrib.educational.widgets.utils.polynomialtransform import \
    PolynomialTransform


class TestOWPolynomialClassification(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPolynomialClassification)
        self.iris = Table("iris")

    def test_add_main_layout(self):
        """
        With this test we check if everything is ok on widget init
        """
        w = self.widget

        # add main layout is called when function is initialized

        # check just if it is initialize on widget start
        self.assertIsNotNone(w.options_box)
        self.assertIsNotNone(w.cbx)
        self.assertIsNotNone(w.cby)
        self.assertIsNotNone(w.target_class_combobox)
        self.assertIsNotNone(w.degree_spin)
        self.assertIsNotNone(w.plot_properties_box)
        self.assertIsNotNone(w.contours_enabled_checkbox)
        self.assertIsNotNone(w.contour_step_slider)
        self.assertIsNotNone(w.scatter)

        # default learner must be logistic regression
        self.assertEqual(w.LEARNER.name, LogisticRegressionLearner.name)

        # widget have to be resizable
        self.assertTrue(w.resizing_enabled)

        # learner should be Logistic regression
        self.assertTrue(isinstance(w.learner, LogisticRegressionLearner))

        # preprocessor should be PolynomialTransform
        self.assertEqual(
            type(w.default_preprocessor), type(PolynomialTransform))

        # check if there is learner on output
        self.assertEqual(self.get_output("Learner"), w.learner)

        # model and coefficients should be none because of no data
        self.assertIsNone(self.get_output("Classifier"))
        self.assertIsNone(self.get_output("Coefficients"))

        # this parameters are none because no plot should be called
        self.assertIsNone(w.xv)
        self.assertIsNone(w.yv)
        self.assertIsNone(w.probabilities_grid)

    def test_set_learner(self):
        """
        Test if learner is set correctly
        """
        w = self.widget

        learner = TreeLearner()

        # check if empty
        self.assertEqual(w.learner_other, None)
        self.assertTrue(isinstance(w.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(w.learner, w.LEARNER))
        self.assertEqual(
            type(self.get_output("Learner")), type(LogisticRegressionLearner()))

        self.send_signal("Learner", learner)

        # check if learners set correctly
        self.assertEqual(w.learner_other, learner)
        self.assertEqual(type(w.learner), type(learner))
        self.assertEqual(type(self.get_output("Learner")), type(learner))

        # after learner is removed there should be LEARNER used
        self.send_signal("Learner", None)
        self.assertEqual(w.learner_other, None)
        self.assertTrue(isinstance(w.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(w.learner, w.LEARNER))
        self.assertEqual(
            type(self.get_output("Learner")), type(LogisticRegressionLearner()))

        # set it again just in case something goes wrong
        learner = RandomForestLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(w.learner_other, learner)
        self.assertEqual(type(w.learner), type(learner))
        self.assertEqual(type(self.get_output("Learner")), type(learner))

        # change learner this time not from None
        learner = TreeLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(w.learner_other, learner)
        self.assertEqual(type(w.learner), type(learner))
        self.assertEqual(type(self.get_output("Learner")), type(learner))

    def test_set_preprocessor(self):
        """
        Test preprocessor set
        """
        w = self.widget

        preprocessor = Continuize()

        # check if empty
        self.assertIn(w.preprocessors, [[], None])

        self.send_signal("Preprocessor", preprocessor)

        # check preprocessor is set
        self.assertEqual(w.preprocessors, [preprocessor])
        self.assertIn(preprocessor, self.get_output("Learner").preprocessors)

        # remove preprocessor
        self.send_signal("Preprocessor", None)

        self.assertIn(w.preprocessors, [[], None])
        self.assertNotIn(preprocessor, self.get_output("Learner").preprocessors)

        # set it again
        preprocessor = Discretize()
        self.send_signal("Preprocessor", preprocessor)

        # check preprocessor is set
        self.assertEqual(w.preprocessors, [preprocessor])
        self.assertIn(preprocessor, self.get_output("Learner").preprocessors)

        # change preprocessor
        preprocessor = Continuize()
        self.send_signal("Preprocessor", preprocessor)

        # check preprocessor is set
        self.assertEqual(w.preprocessors, [preprocessor])
        self.assertIn(preprocessor, self.get_output("Learner").preprocessors)

    def test_set_data(self):
        """
        Test widget behavior when data set
        """
        w = self.widget

        num_continuous_attributes = sum(
            True for var in self.iris.domain.attributes
            if isinstance(var, ContinuousVariable))

        self.send_signal("Data", self.iris)

        # widget does not have any problems with that data set so
        # everything should be fine
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

        # remove data set
        self.send_signal("Data", None)
        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)

        # set data set again
        self.send_signal("Data", self.iris)

        # widget does not have any problems with that data set so
        # everything should be fine
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

        # set data set with no class
        table_no_class = Table(
            Domain([ContinuousVariable("x"), ContinuousVariable("y")]),
            [[1, 2], [2, 3]])
        self.send_signal("Data", table_no_class)

        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertTrue(w.Error.no_class.is_shown())

        # set data with one class variable
        table_one_class = Table(
            Domain([ContinuousVariable("x"), ContinuousVariable("y")],
                   DiscreteVariable("a", values=["k"])),
            [[1, 2], [2, 3]], [0, 0])
        self.send_signal("Data", table_one_class)

        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertTrue(w.Error.no_class.is_shown())

        # set data with not enough continuous variables
        table_no_enough_cont = Table(
            Domain(
                [ContinuousVariable("x"),
                 DiscreteVariable("y", values=["a", "b"])],
                ContinuousVariable("a")),
            [[1, 0], [2, 1]], [0, 0])
        self.send_signal("Data", table_no_enough_cont)

        self.assertEqual(w.cbx.count(), 0)
        self.assertEqual(w.cby.count(), 0)
        self.assertEqual(w.target_class_combobox.count(), 0)
        self.assertTrue(w.Error.to_few_features.is_shown())

    def test_init_learner(self):
        """
        Test init
        """
        w = self.widget

        learner = TreeLearner()

        # check if empty
        self.assertTrue(isinstance(w.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(w.learner, w.LEARNER))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, w.default_preprocessor),
                   w.learner.preprocessors, False))

        self.send_signal("Learner", learner)

        # check if learners set correctly
        self.assertEqual(type(w.learner), type(learner))

        # after learner is removed there should be LEARNER used
        self.send_signal("Learner", None)
        self.assertTrue(isinstance(w.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(w.learner, w.LEARNER))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, w.default_preprocessor),
                   w.learner.preprocessors, False))

        # set it again just in case something goes wrong
        learner = RandomForestLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(type(w.learner), type(learner))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, w.default_preprocessor),
                   w.learner.preprocessors, False))

        # change learner this time not from None
        learner = TreeLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(type(w.learner), type(learner))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, w.default_preprocessor),
                   w.learner.preprocessors, False))

        # set other preprocessor
        preprocessor = Discretize
        # selected this preprocessor because know that not exist in LogReg
        self.send_signal("Preprocessor", preprocessor())

        self.assertEqual(type(w.learner), type(learner))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, w.default_preprocessor),
                   w.learner.preprocessors, False))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, preprocessor),
                   w.learner.preprocessors, False))

        # remove preprocessor
        self.send_signal("Preprocessor", None)
        self.assertEqual(type(w.learner), type(learner))
        self.assertTrue(
            reduce(lambda x, y: x or isinstance(y, w.default_preprocessor),
                   w.learner.preprocessors, False))

        self.assertFalse(reduce(lambda x, y: x or isinstance(y, preprocessor),
                                w.learner.preprocessors, False))

    def test_replot(self):
        """
        Test everything that is possible to test in replot
        This function tests all replot functions
        """
        w = self.widget

        w.replot()

        # test nothing happens when no data
        self.assertIsNone(w.xv)
        self.assertIsNone(w.yv)
        self.assertIsNone(w.probabilities_grid)

        # when data available plot happens
        self.send_signal("Data", self.iris)
        self.assertIsNotNone(w.xv)
        self.assertIsNotNone(w.yv)
        self.assertIsNotNone(w.probabilities_grid)
        self.assertTupleEqual(
            (w.grid_size, w.grid_size), w.probabilities_grid.shape)
        self.assertTupleEqual((w.grid_size, w.grid_size), w.xv.shape)
        self.assertTupleEqual((w.grid_size, w.grid_size), w.yv.shape)

        # check that everything works fine when contours enabled/disabled
        w.contours_enabled_checkbox.click()

        self.assertIsNotNone(w.xv)
        self.assertIsNotNone(w.yv)
        self.assertIsNotNone(w.probabilities_grid)
        self.assertTupleEqual(
            (w.grid_size, w.grid_size), w.probabilities_grid.shape)
        self.assertTupleEqual((w.grid_size, w.grid_size), w.xv.shape)
        self.assertTupleEqual((w.grid_size, w.grid_size), w.yv.shape)

        w.contours_enabled_checkbox.click()

        self.assertIsNotNone(w.xv)
        self.assertIsNotNone(w.yv)
        self.assertIsNotNone(w.probabilities_grid)
        self.assertTupleEqual(
            (w.grid_size, w.grid_size), w.probabilities_grid.shape)
        self.assertTupleEqual((w.grid_size, w.grid_size), w.xv.shape)
        self.assertTupleEqual((w.grid_size, w.grid_size), w.yv.shape)

        # when remove data
        self.send_signal("Data", None)

        self.assertIsNone(w.xv)
        self.assertIsNone(w.yv)
        self.assertIsNone(w.probabilities_grid)

    def test_blur_grid(self):
        w = self.widget

        self.send_signal("Data", self.iris)
        # here we can check that 0.5 remains same
        assert_array_equal(w.probabilities_grid == 0.5,
                           w.blur_grid(w.probabilities_grid) == 0.5)

    def test_select_data(self):
        """
        Check if select data works properly
        """
        w = self.widget

        self.send_signal("Data", self.iris)

        selected_data = w.select_data()
        self.assertEqual(len(selected_data.domain.attributes), 2)
        self.assertIsNotNone(selected_data.domain.class_var)
        self.assertEqual(len(selected_data.domain.metas), 1)
        # meta with information about real cluster
        self.assertEqual(len(selected_data), len(self.iris))

        # selected data none when one column only Nones
        data = Table(Domain([ContinuousVariable('a'), ContinuousVariable('b')],
                            DiscreteVariable('c', values=['a', 'b'])),
                     [[1, None], [1, None]], [0, 1])
        self.send_signal("Data", data)
        selected_data = w.select_data()
        self.assertIsNone(selected_data)

        data = Table(Domain([ContinuousVariable('a'), ContinuousVariable('b')],
                            DiscreteVariable('c', values=['a', 'b'])),
                     [[None, None], [None, None]], [0, 1])
        self.send_signal("Data", data)
        selected_data = w.select_data()
        self.assertIsNone(selected_data)

    def test_send_learner(self):
        """
        Test if correct learner on output
        """
        w = self.widget

        self.assertEqual(self.get_output("Learner"), w.learner)
        self.assertTrue(isinstance(self.get_output("Learner"), w.LEARNER))

        # set new learner
        learner = TreeLearner
        self.send_signal("Learner", learner())
        self.assertEqual(self.get_output("Learner"), w.learner)
        self.assertTrue(isinstance(self.get_output("Learner"), learner))

        # back to default learner
        self.send_signal("Learner", None)
        self.assertEqual(self.get_output("Learner"), w.learner)
        self.assertTrue(isinstance(self.get_output("Learner"), w.LEARNER))

    def test_update_model(self):
        """
        Function check if correct model is on output
        """
        w = self.widget

        # when no data
        self.assertIsNone(w.model)
        self.assertIsNone(self.get_output("Classifier"))

        # set data
        self.send_signal("Data", self.iris)
        self.assertIsNotNone(w.model)
        self.assertEqual(w.model, self.get_output("Classifier"))

        # remove data
        self.send_signal("Data", None)
        self.assertIsNone(w.model)
        self.assertIsNone(self.get_output("Classifier"))

    def test_send_coefficients(self):
        """
        Coefficients are only available if Logistic regression is used
        """
        w = self.widget

        # none when no data (model not build)
        self.assertIsNone(self.get_output("Coefficients"))

        # by default LogisticRegression so coefficients exists
        self.send_signal("Data", self.iris)

        # to check correctness before degree is changed
        num_coefficients = sum(i + 1 for i in range(w.degree + 1))
        self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # change degree
        for j in range(1, 6):
            w.degree_spin.setValue(j)
            num_coefficients = sum(i + 1 for i in range(w.degree + 1))
            self.assertEqual(
                len(self.get_output("Coefficients")), num_coefficients)

        # change learner which does not have coefficients
        learner = TreeLearner
        self.send_signal("Learner", learner())
        self.assertIsNone(self.get_output("Coefficients"))

        # remove learner
        self.send_signal("Learner", None)

        # to check correctness before degree is changed
        num_coefficients = sum(i + 1 for i in range(w.degree + 1))
        self.assertEqual(
            len(self.get_output("Coefficients")), num_coefficients)

        # change degree
        for j in range(1, 6):
            w.degree_spin.setValue(j)
            num_coefficients = sum(i + 1 for i in range(w.degree + 1))
            self.assertEqual(
                len(self.get_output("Coefficients")), num_coefficients)

        # manulay set LogisticRegression
        self.send_signal("Learner", LogisticRegressionLearner())

        # to check correctness before degree is changed
        num_coefficients = sum(i + 1 for i in range(w.degree + 1))
        self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # change degree
        for j in range(1, 6):
            w.degree_spin.setValue(j)
            num_coefficients = sum(i + 1 for i in range(w.degree + 1))
            self.assertEqual(
                len(self.get_output("Coefficients")), num_coefficients)

    def test_send_data(self):
        """
        Check data output signal
        """
        w = self.widget

        self.assertIsNone(self.get_output("Data"))

        self.send_signal("Data", self.iris)

        # check correct number of attributes
        for j in range(1, 6):
            w.degree_spin.setValue(j)
            self.assertEqual(
                len(self.get_output("Data").domain.attributes), 2)

        self.assertEqual(len(self.get_output("Data").domain.metas), 1)
        self.assertIsNotNone(self.get_output("Data").domain.class_var)

        # check again none
        self.send_signal("Data", None)
        self.assertIsNone(self.get_output("Data"))

    def test_send_report(self):
        """
        Just test everything not crashes
        """
        w = self.widget

        self.send_signal("Data", self.iris)
        w.report_button.click()
