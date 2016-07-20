from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.classification import LogisticRegressionLearner, TreeLearner, RandomForestLearner
from Orange.preprocess.preprocess import Continuize, Discretize
from orangecontrib.educational.widgets.owpolynomialclassification import OWPolyinomialClassification
from orangecontrib.educational.widgets.utils.polynomialexpansion import PolynomialTransform
from functools import reduce
from numpy.testing import assert_array_equal


class TestOWPolynomialClassification(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWPolyinomialClassification)
        self.iris = Table("iris")

    def test_add_main_layout(self):
        """
        With this test we check if everything is ok on widget init
        """

        # add main layout is called when function is initialized

        # check just if it is initialize on widget start
        self.assertIsNotNone(self.widget.options_box)
        self.assertIsNotNone(self.widget.cbx)
        self.assertIsNotNone(self.widget.cby)
        self.assertIsNotNone(self.widget.target_class_combobox)
        self.assertIsNotNone(self.widget.degree_spin)
        self.assertIsNotNone(self.widget.plot_properties_box)
        self.assertIsNotNone(self.widget.contours_enabled_checkbox)
        self.assertIsNotNone(self.widget.contour_step_slider)
        self.assertIsNotNone(self.widget.scatter)

        # default learner must be logistic regression
        self.assertEqual(self.widget.LEARNER.name, LogisticRegressionLearner.name)

        # widget have to be resizable
        self.assertTrue(self.widget.resizing_enabled)

        # learner should be Logistic regression
        self.assertTrue(isinstance(self.widget.learner, LogisticRegressionLearner))

        # preprocessor should be PolynomialTransform
        self.assertEqual(type(self.widget.default_preprocessor), type(PolynomialTransform))

        # check if there is learner on output
        self.assertEqual(self.get_output("Learner"), self.widget.learner)

        # model and coefficients should be none because of no data
        self.assertIsNone(self.get_output("Classifier"))
        self.assertIsNone(self.get_output("Coefficients"))

        # this parameters are none because no plot should be called
        self.assertIsNone(self.widget.xv)
        self.assertIsNone(self.widget.yv)
        self.assertIsNone(self.widget.probabilities_grid)

    def test_set_learner(self):
        """
        Test if learner is set correctly
        """
        learner = TreeLearner()

        # check if empty
        self.assertEqual(self.widget.learner_other, None)
        self.assertTrue(isinstance(self.widget.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(self.widget.learner, self.widget.LEARNER))
        self.assertEqual(type(self.get_output("Learner")), type(LogisticRegressionLearner()))

        self.send_signal("Learner", learner)

        # check if learners set correctly
        self.assertEqual(self.widget.learner_other, learner)
        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertEqual(type(self.get_output("Learner")), type(learner))

        # after learner is removed there should be LEARNER used
        self.send_signal("Learner", None)
        self.assertEqual(self.widget.learner_other, None)
        self.assertTrue(isinstance(self.widget.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(self.widget.learner, self.widget.LEARNER))
        self.assertEqual(type(self.get_output("Learner")), type(LogisticRegressionLearner()))

        # set it again just in case something goes wrong
        learner = RandomForestLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(self.widget.learner_other, learner)
        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertEqual(type(self.get_output("Learner")), type(learner))

        # change learner this time not from None
        learner = TreeLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(self.widget.learner_other, learner)
        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertEqual(type(self.get_output("Learner")), type(learner))

    def test_set_preprocessor(self):
        """
        Test preprocessor set
        """
        preprocessor = Continuize()

        # check if empty
        self.assertIn(self.widget.preprocessors, [[], None])

        self.send_signal("Preprocessor", preprocessor)

        # check preprocessor is set
        self.assertEqual(self.widget.preprocessors, [preprocessor])
        self.assertIn(preprocessor, self.get_output("Learner").preprocessors)

        # remove preprocessor
        self.send_signal("Preprocessor", None)

        self.assertIn(self.widget.preprocessors, [[], None])
        self.assertNotIn(preprocessor, self.get_output("Learner").preprocessors)

        # set it again
        preprocessor = Discretize()
        self.send_signal("Preprocessor", preprocessor)

        # check preprocessor is set
        self.assertEqual(self.widget.preprocessors, [preprocessor])
        self.assertIn(preprocessor, self.get_output("Learner").preprocessors)

        # change preprocessor
        preprocessor = Continuize()
        self.send_signal("Preprocessor", preprocessor)

        # check preprocessor is set
        self.assertEqual(self.widget.preprocessors, [preprocessor])
        self.assertIn(preprocessor, self.get_output("Learner").preprocessors)

    def test_set_data(self):
        """
        Test widget behavior when data set
        """
        num_continuous_attributes = sum(True for var in self.iris.domain.attributes
                                        if isinstance(var, ContinuousVariable))

        self.send_signal("Data", self.iris)

        # widget does not have any problems with that data set so everything should be fine
        self.assertEqual(self.widget.cbx.count(), num_continuous_attributes)
        self.assertEqual(self.widget.cby.count(), num_continuous_attributes)
        self.assertEqual(self.widget.target_class_combobox.count(), len(self.iris.domain.class_var.values))
        self.assertEqual(self.widget.cbx.currentText(), self.iris.domain[0].name)
        self.assertEqual(self.widget.cby.currentText(), self.iris.domain[1].name)
        self.assertEqual(self.widget.target_class_combobox.currentText(), self.iris.domain.class_var.values[0])

        self.assertEqual(self.widget.attr_x, self.iris.domain[0].name)
        self.assertEqual(self.widget.attr_y, self.iris.domain[1].name)
        self.assertEqual(self.widget.target_class, self.iris.domain.class_var.values[0])

        # change showed attributes
        self.widget.attr_x = self.iris.domain[1].name
        self.widget.attr_y = self.iris.domain[2].name
        self.widget.target_class = self.iris.domain.class_var.values[1]

        self.assertEqual(self.widget.cbx.currentText(), self.iris.domain[1].name)
        self.assertEqual(self.widget.cby.currentText(), self.iris.domain[2].name)
        self.assertEqual(self.widget.target_class_combobox.currentText(), self.iris.domain.class_var.values[1])

        self.assertEqual(self.widget.attr_x, self.iris.domain[1].name)
        self.assertEqual(self.widget.attr_y, self.iris.domain[2].name)
        self.assertEqual(self.widget.target_class, self.iris.domain.class_var.values[1])

        # remove data set
        self.send_signal("Data", None)
        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)
        self.assertEqual(self.widget.target_class_combobox.count(), 0)

        # set data set again
        self.send_signal("Data", self.iris)

        # widget does not have any problems with that data set so everything should be fine
        self.assertEqual(self.widget.cbx.count(), num_continuous_attributes)
        self.assertEqual(self.widget.cby.count(), num_continuous_attributes)
        self.assertEqual(self.widget.target_class_combobox.count(), len(self.iris.domain.class_var.values))
        self.assertEqual(self.widget.cbx.currentText(), self.iris.domain[0].name)
        self.assertEqual(self.widget.cby.currentText(), self.iris.domain[1].name)
        self.assertEqual(self.widget.target_class_combobox.currentText(), self.iris.domain.class_var.values[0])

        self.assertEqual(self.widget.attr_x, self.iris.domain[0].name)
        self.assertEqual(self.widget.attr_y, self.iris.domain[1].name)
        self.assertEqual(self.widget.target_class, self.iris.domain.class_var.values[0])

        # set data set with no class
        table_no_class = Table(Domain([ContinuousVariable("x"), ContinuousVariable("y")]),
                               [[1, 2], [2, 3]])
        self.send_signal("Data", table_no_class)

        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)
        self.assertEqual(self.widget.target_class_combobox.count(), 0)
        self.assertIsNotNone(self.widget.widgetState["Warning"][1])

        # set data with one class variable
        table_one_class = Table(Domain([ContinuousVariable("x"), ContinuousVariable("y")],
                                       DiscreteVariable("a", values=["k"])),
                                [[1, 2], [2, 3]], [0, 0])
        self.send_signal("Data", table_one_class)

        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)
        self.assertEqual(self.widget.target_class_combobox.count(), 0)
        self.assertIsNotNone(self.widget.widgetState["Warning"][1])

        # set data with not enough continuous variables
        table_no_enough_cont = Table(Domain([ContinuousVariable("x"), DiscreteVariable("y", values=["a", "b"])],
                                            ContinuousVariable("a")),
                                     [[1, 0], [2, 1]], [0, 0])
        self.send_signal("Data", table_no_enough_cont)

        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)
        self.assertEqual(self.widget.target_class_combobox.count(), 0)
        self.assertIsNotNone(self.widget.widgetState["Warning"][1])

    def test_init_learner(self):
        """
        Test init
        """
        learner = TreeLearner()

        # check if empty
        self.assertTrue(isinstance(self.widget.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(self.widget.learner, self.widget.LEARNER))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, self.widget.default_preprocessor),
                               self.widget.learner.preprocessors, False))

        self.send_signal("Learner", learner)

        # check if learners set correctly
        self.assertEqual(type(self.widget.learner), type(learner))

        # after learner is removed there should be LEARNER used
        self.send_signal("Learner", None)
        self.assertTrue(isinstance(self.widget.learner, LogisticRegressionLearner))
        self.assertTrue(isinstance(self.widget.learner, self.widget.LEARNER))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, self.widget.default_preprocessor),
                               self.widget.learner.preprocessors, False))

        # set it again just in case something goes wrong
        learner = RandomForestLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, self.widget.default_preprocessor),
                               self.widget.learner.preprocessors, False))

        # change learner this time not from None
        learner = TreeLearner()
        self.send_signal("Learner", learner)

        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, self.widget.default_preprocessor),
                               self.widget.learner.preprocessors, False))

        # set other preprocessor
        preprocessor = Discretize  # we selected this preprocessor because we know that it does not exist in LogReg
        self.send_signal("Preprocessor", preprocessor())

        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, self.widget.default_preprocessor),
                               self.widget.learner.preprocessors, False))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, preprocessor),
                               self.widget.learner.preprocessors, False))

        # remove preprocessor
        self.send_signal("Preprocessor", None)
        self.assertEqual(type(self.widget.learner), type(learner))
        self.assertTrue(reduce(lambda x, y: x or isinstance(y, self.widget.default_preprocessor),
                               self.widget.learner.preprocessors, False))

        self.assertFalse(reduce(lambda x, y: x or isinstance(y, preprocessor),
                                self.widget.learner.preprocessors, False))

    def test_replot(self):
        """
        Test everything that is possible to test in replot
        This function tests all replot functions
        """

        # test nothing happens when no data
        self.assertIsNone(self.widget.xv)
        self.assertIsNone(self.widget.yv)
        self.assertIsNone(self.widget.probabilities_grid)

        # when data available plot happens
        self.send_signal("Data", self.iris)
        self.assertIsNotNone(self.widget.xv)
        self.assertIsNotNone(self.widget.yv)
        self.assertIsNotNone(self.widget.probabilities_grid)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.probabilities_grid.shape)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.xv.shape)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.yv.shape)

        # check that everything works fine when contours enabled/disabled
        self.widget.contours_enabled_checkbox.click()

        self.assertIsNotNone(self.widget.xv)
        self.assertIsNotNone(self.widget.yv)
        self.assertIsNotNone(self.widget.probabilities_grid)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.probabilities_grid.shape)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.xv.shape)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.yv.shape)

        self.widget.contours_enabled_checkbox.click()

        self.assertIsNotNone(self.widget.xv)
        self.assertIsNotNone(self.widget.yv)
        self.assertIsNotNone(self.widget.probabilities_grid)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.probabilities_grid.shape)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.xv.shape)
        self.assertTupleEqual((self.widget.grid_size, self.widget.grid_size), self.widget.yv.shape)

        # when remove data
        self.send_signal("Data", None)

        self.assertIsNone(self.widget.xv)
        self.assertIsNone(self.widget.yv)
        self.assertIsNone(self.widget.probabilities_grid)

    def test_blur_grid(self):
        self.send_signal("Data", self.iris)
        # here we can check that 0.5 remains same
        assert_array_equal(self.widget.probabilities_grid == 0.5,
                           self.widget.blur_grid(self.widget.probabilities_grid) == 0.5)

    def test_select_data(self):
        """
        Check if select data works properly
        """
        self.send_signal("Data", self.iris)

        selected_data = self.widget.select_data()
        self.assertEqual(len(selected_data.domain.attributes), 2)
        self.assertIsNotNone(selected_data.domain.class_var)
        self.assertEqual(len(selected_data.domain.metas), 1)  # meta with information about real cluster
        self.assertEqual(len(selected_data), len(self.iris))

    def test_send_learner(self):
        """
        Test if correct learner on output
        """
        self.assertEqual(self.get_output("Learner"), self.widget.learner)
        self.assertTrue(isinstance(self.get_output("Learner"), self.widget.LEARNER))

        # set new learner
        learner = TreeLearner
        self.send_signal("Learner", learner())
        self.assertEqual(self.get_output("Learner"), self.widget.learner)
        self.assertTrue(isinstance(self.get_output("Learner"), learner))

        # back to default learner
        self.send_signal("Learner", None)
        self.assertEqual(self.get_output("Learner"), self.widget.learner)
        self.assertTrue(isinstance(self.get_output("Learner"), self.widget.LEARNER))

    def test_update_model(self):
        """
        Function check if correct model is on output
        """

        # when no data
        self.assertIsNone(self.widget.model)
        self.assertIsNone(self.get_output("Classifier"))

        # set data
        self.send_signal("Data", self.iris)
        self.assertIsNotNone(self.widget.model)
        self.assertEqual(self.widget.model, self.get_output("Classifier"))

        # remove data
        self.send_signal("Data", None)
        self.assertIsNone(self.widget.model)
        self.assertIsNone(self.get_output("Classifier"))

    def test_send_coefficients(self):
        """
        Coefficients are only available if Logistic regression is used
        """

        # none when no data (model not build)
        self.assertIsNone(self.get_output("Coefficients"))

        # by default LogisticRegression so coefficients exists
        self.send_signal("Data", self.iris)

        # to check correctness before degree is changed
        num_coefficients = sum(i + 1 for i in range(self.widget.degree + 1))
        self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # change degree
        for j in range(1, 6):
            self.widget.degree_spin.setValue(j)
            num_coefficients = sum(i + 1 for i in range(self.widget.degree + 1))
            self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # change learner which does not have coefficients
        learner = TreeLearner
        self.send_signal("Learner", learner())
        self.assertIsNone(self.get_output("Coefficients"))

        # remove learner
        self.send_signal("Learner", None)

        # to check correctness before degree is changed
        num_coefficients = sum(i + 1 for i in range(self.widget.degree + 1))
        self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # change degree
        for j in range(1, 6):
            self.widget.degree_spin.setValue(j)
            num_coefficients = sum(i + 1 for i in range(self.widget.degree + 1))
            self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # manulay set LogisticRegression
        self.send_signal("Learner", LogisticRegressionLearner())

        # to check correctness before degree is changed
        num_coefficients = sum(i + 1 for i in range(self.widget.degree + 1))
        self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)

        # change degree
        for j in range(1, 6):
            self.widget.degree_spin.setValue(j)
            num_coefficients = sum(i + 1 for i in range(self.widget.degree + 1))
            self.assertEqual(len(self.get_output("Coefficients")), num_coefficients)
