# Sins committed by mock classes are as follows:
# pylint: disable=abstract-method, attribute-defined-outside-init

import unittest
from unittest.mock import Mock
from collections import Counter

from Orange.widgets.tests.base import WidgetTest, GuiTest
from Orange.data.domain import ContinuousVariable, DiscreteVariable
from orangecontrib.educational.widgets.owrandomdata import (
    OWRandomData, distributions,
    ParameterDef, pos_int, any_float, pos_float, prob_float,
    ParametersEditor, ParametersEditorContinuous, ParametersEditorDiscrete,
    Multinomial, HyperGeometric)


class MockEditor(ParametersEditor):
    name = "Mock Distribution"

    parameters = (
        ParameterDef("an int", "i", 42, pos_int),
        ParameterDef("a float", "f", 3.14, any_float),
        ParameterDef("pos float", "posf", 2.718, pos_float),
        ParameterDef("prob float", "prob", 1, prob_float),
        ParameterDef("something", "sth", "foo", str))


class MockEditorEmpty(ParametersEditor):
    name = "No parameters"


class TestParametersEditor(GuiTest):
    def test_initial_values(self):
        e = MockEditor()
        for par in MockEditor.parameters:
            line = e.edits[par.arg_name]
            self.assertEqual(line.text(), str(par.default))
            self.assertEqual(e.get(par.arg_name), par.default)

    @staticmethod
    def test_editor_without_parameters():
        MockEditorEmpty()

    def test_changed_values(self):
        e = MockEditor()
        edit = e.edits["posf"]
        edit.setText("6.28")
        self.assertEqual(e.get("posf"), 6.28)

    def test_fix_standard_parameters(self):
        e = MockEditor()
        e.fix_standard_parameters(42, "foo")
        self.assertEqual(e.nvars, 42)
        self.assertEqual(e.get_name_prefix({})[0], "foo")
        self.assertFalse(e.number_of_vars.isEnabled())
        self.assertFalse(e.name_prefix.isEnabled())

        e.fix_standard_parameters(42, None)
        self.assertEqual(e.nvars, 42)
        self.assertEqual(e.get_name_prefix({})[0], "foo")
        self.assertFalse(e.number_of_vars.isEnabled())
        self.assertTrue(e.name_prefix.isEnabled())

        e.fix_standard_parameters(None, "foo")
        self.assertEqual(e.nvars, 42)
        self.assertEqual(e.get_name_prefix({})[0], "foo")
        self.assertTrue(e.number_of_vars.isEnabled())
        self.assertFalse(e.name_prefix.isEnabled())

        e.fix_standard_parameters(None, None)
        self.assertEqual(e.nvars, 42)
        self.assertEqual(e.get_name_prefix({})[0], "foo")
        self.assertTrue(e.number_of_vars.isEnabled())
        self.assertTrue(e.name_prefix.isEnabled())

        e = MockEditorEmpty()
        e.fix_standard_parameters(42, "foo")
        self.assertEqual(e.nvars, 42)
        self.assertEqual(e.get_name_prefix({})[0], "foo")
        self.assertFalse(e.number_of_vars.isEnabled())
        self.assertFalse(e.name_prefix.isEnabled())

    def test_nvars(self):
        e = MockEditor()
        e.number_of_vars.setText("42")
        self.assertEqual(e.nvars, 42)

        e = MockEditorEmpty()
        e.number_of_vars.setText("42")
        self.assertEqual(e.nvars, 42)

    def test_get_parameters(self):
        e = MockEditor()
        parameters = {p.arg_name: p.default for p in MockEditor.parameters}
        self.assertEqual(e.get_parameters(), parameters)
        e.edits["f"].setText("6.28")
        parameters["f"] = 6.28
        self.assertEqual(e.get_parameters(), parameters)

        e = MockEditorEmpty()
        self.assertEqual(e.get_parameters(), {})

    def test_set_erorr(self):
        e = MockEditor()
        self.assertTrue(e.error.isHidden())
        e.set_error(None)
        self.assertTrue(e.error.isHidden())
        e.set_error("-1 is not a positive number")
        self.assertTrue("-1 is not a positive number" in e.error.text())
        self.assertFalse(e.error.isHidden())
        e.set_error(None)
        self.assertTrue(e.error.isHidden())

    def test_get_name_prefix(self):
        used = {"foo": 8, "bar": 5}
        e = MockEditor()

        e.name_prefix.setText("foo")
        self.assertEqual(e.get_name_prefix(used), ("foo", 9))

        e.name_prefix.setText("bar")
        self.assertEqual(e.get_name_prefix(used), ("bar", 6))

        e.name_prefix.setText("baz")
        self.assertEqual(e.get_name_prefix(used), ("baz", 1))

    def test_generate_data(self):
        e = MockEditor()
        e.number_of_vars.setText("42")
        pars = {"a": 12}
        e.get_parameters = Mock(return_value=pars)
        data = object()
        e.rvs = Mock(return_value=data)
        self.assertIs(e.generate_data(100), data)
        e.rvs.assert_called_with(size=(100, 42), **pars)

    def test_generate_data_checks_error(self):
        e = MockEditor()
        e.check = lambda **_: "not good"
        e.rvs = Mock()
        self.assertIsNone(e.generate_data(10))
        self.assertFalse(e.error.isHidden())
        self.assertTrue("not good" in e.error.text())
        e.rvs.assert_not_called()

        e.check = lambda **_: None
        e.generate_data(10)
        self.assertTrue(e.error.isHidden())
        e.rvs.assert_called_once()

    def test_generate_data_rvs_error(self):
        e = MockEditor()
        e.rvs = Mock(side_effect=ValueError("foo"))
        self.assertIsNone(e.generate_data(10))
        self.assertFalse(e.error.isHidden())

        e.rvs = Mock()
        e.generate_data(10)
        self.assertTrue(e.error.isHidden())

    def test_pack_settings(self):
        e = MockEditor()
        e.number_of_vars.setText("42")
        e.name_prefix.setText("bar")
        self.assertEqual(
            e.pack_settings(),
            dict(number_of_vars="42", name_prefix="bar",
                 i="42", f="3.14", posf="2.718", prob="1", sth="foo"))

    def unpack_settings(self):
        e = MockEditor()
        e.unpack_settings(
            dict(number_of_vars="2", name_prefix="bing",
                 i="43", f="6.28", posf="2.71828", prob="0.5", sth="baz"))
        self.assertEqual(e.nvars, 2)
        self.assertEqual(e.get_name_prefix({})[0], "bing")
        self.assertEqual(e.i, 42)
        self.assertEqual(e.f, 6.28)
        self.assertEqual(e.posf, "2.71828")
        self.assertEqual(e.prob, "0.5")
        self.assertEqual(e.sth, "baz")

    def test_prepare_variables_is_abstract(self):
        e = MockEditor()
        self.assertRaises(NotImplementedError, e.prepare_variables, {}, 3)

class TestParametersEditorContinuous(GuiTest):
    def test_prepare_variables(self):
        class MockEditorC(ParametersEditorContinuous):
            name = "mock"

        used = {"bar": 8, "baz": 5}
        e = MockEditorC()
        e.name_prefix.setText("baz")
        e.number_of_vars.setText("4")
        attrs = e.prepare_variables(used, 4)
        self.assertEqual(len(attrs), 4)
        self.assertTrue(
            all(isinstance(var, ContinuousVariable) for var in attrs))
        self.assertEqual({var.name for var in attrs},
                         {"baz0006", "baz0007", "baz0008", "baz0009"})
        self.assertEqual(used, {"bar": 8, "baz": 9})

        e.name_prefix.setText("x")
        e.number_of_vars.setText("3")
        attrs = e.prepare_variables(used, 1)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(
            all(isinstance(var, ContinuousVariable) for var in attrs))
        self.assertEqual({var.name for var in attrs}, {"x1", "x2", "x3"})
        self.assertEqual(used, {"bar": 8, "baz": 9, "x": 3})


class TestParametersEditorDiscrete(GuiTest):
    def test_prepare_variables(self):
        class MockEditorD(ParametersEditorDiscrete):
            name = "mock"

            @staticmethod
            def get_values(**_):
                return "abc", "def", "ghi"

        used = {"bar": 8, "baz": 5}
        e = MockEditorD()
        e.name_prefix.setText("baz")
        e.number_of_vars.setText("3")
        attrs = e.prepare_variables(used, 3)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(isinstance(var, DiscreteVariable) for var in attrs))
        self.assertTrue(
            all(tuple(var.values) == ("abc", "def", "ghi") for var in attrs))
        self.assertEqual({var.name for var in attrs},
                         {"baz006", "baz007", "baz008"})
        self.assertEqual(used, {"bar": 8, "baz": 8})

        e.name_prefix.setText("x")
        e.number_of_vars.setText("3")
        attrs = e.prepare_variables(used, 1)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(isinstance(var, DiscreteVariable) for var in attrs))
        self.assertTrue(
            all(tuple(var.values) == ("abc", "def", "ghi") for var in attrs))
        self.assertEqual({var.name for var in attrs}, {"x1", "x2", "x3"})
        self.assertEqual(used, {"bar": 8, "baz": 8, "x": 3})


class TestMultinomial(GuiTest):
    def test_ps(self):
        e = Multinomial()
        edit = e.edits["ps"]

        edit.setText("0.4 0.6")
        self.assertEqual(e.get("ps"), [0.4, 0.6])
        self.assertIsNone(e.check())
        self.assertEqual(e.nvars, 2)

        edit.setText("0.4, 0.6")
        self.assertEqual(e.get("ps"), [0.4, 0.6])
        self.assertIsNone(e.check())

        edit.setText("0.4; 0.6")
        self.assertEqual(e.get("ps"), [0.4, 0.6])
        self.assertIsNone(e.check())

        edit.setText("0.1, 0.2, 0.3, 0.4")
        self.assertEqual(e.get("ps"), [0.1, 0.2, 0.3, 0.4])
        self.assertIsNone(e.check())
        self.assertEqual(e.nvars, 4)

        edit.setText("0.1, 0.X, 0.3, 0.4")
        self.assertEqual(e.get("ps"), [])
        self.assertIsInstance(e.check(), str)
        self.assertEqual(e.nvars, 0)

        edit.setText("0.1, 0.4")
        self.assertEqual(e.get("ps"), [])
        self.assertIsInstance(e.check(), str)
        self.assertEqual(e.nvars, 0)

        self.assertIsNone(e.rvs([], 10, (5, 5)))


class TestHyperGeometric(GuiTest):
    def test_check(self):
        e = HyperGeometric()
        self.assertIsInstance(e.check(M=5, n=2, N=8), str)
        self.assertIsInstance(e.check(M=5, n=8, N=2), str)


class TestOWRandomData(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWRandomData)

    def test_all_distributions(self):
        settings = {
            "distributions":
            [('Normal distribution',
              {'number_of_vars': '1', 'name_prefix': 'a',
               'loc': '0', 'scale': '1'}),
             ('Bernoulli distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'p': '0.5'}),
             ('Binomial distribution',
              {'number_of_vars': '3', 'name_prefix': 'c',
               'n': '100', 'p': '0.5'}),
             ('Uniform distribution',
              {'number_of_vars': '4', 'name_prefix': 'd',
               'loc': '0', 'scale': '1'}),
             ('Discrete uniform distribution',
              {'number_of_vars': '5', 'name_prefix': 'e',
               'k': '6'}),
             ('Multinomial distribution',
              {'number_of_vars': '3', 'name_prefix': 'a',
               'ps': '0.5, 0.3, 0.2', 'n': '100'}),
             ('Hypergeometric distribution',
              {'number_of_vars': '6', 'name_prefix': 'f',
               'M': '100', 'n': '20', 'N': '20'}),
             ('Negative binomial distribution',
              {'number_of_vars': '7', 'name_prefix': 'g',
               'n': '10', 'p': '0.5'}),
             ('Poisson distribution',
              {'number_of_vars': '8', 'name_prefix': 'h',
               'mu': '5'}),
             ('Exponential distribution',
              {'number_of_vars': '9', 'name_prefix': 'i'}),
             ('Gamma distribution',
              {'number_of_vars': '10', 'name_prefix': 'j',
               'a': '2', 'scale': '2'}),
             ("Student's t distribution",
              {'number_of_vars': '11', 'name_prefix': 'k',
               'df': '1'}),
             ('Bivariate normal distribution',
              {'number_of_vars': '2', 'name_prefix': 'x, y',
               'mu1': '0', 'var1': '1', 'mu2': '0', 'var2': '1',
               'covar': '0.5'})
             ]}
        widget = self.create_widget(OWRandomData, settings)
        widget.n_instances = 50
        widget.generate()
        data = self.get_output(widget.Outputs.data, widget)
        self.assertEqual(
            data.X.shape,
            (50, sum(int(d[1]["number_of_vars"])
                     for d in settings["distributions"]))
        )
        var_counts = Counter(var.name[0] for var in data.domain.variables)
        self.assertEqual(var_counts["a"], 4)
        for i, c in enumerate("bcdefghijk"):
            self.assertEqual(var_counts[c], i + 2)

    def test_add_editor(self):
        settings = {
            "distributions":
            [('Normal distribution',
              {'number_of_vars': '1', 'name_prefix': '',
               'loc': '0', 'scale': '1'}),
             ('Bernoulli distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'p': '0.5'})]}
        widget = self.create_widget(OWRandomData, settings)
        widget.add_editor(distributions["Normal distribution"]())
        data = self.get_output(widget.Outputs.data, widget)
        self.assertEqual(data.X.shape[1], 1 + 2 + 10)
        self.assertEqual(
            {var.name for var in data.domain.variables},
            {"b01", "b02"} |
            {f"{ParametersEditor.default_prefix}{i:02}" for i in range(1, 12)}
        )

    def test_remove_editor(self):
        settings = {
            "distributions":
            [('Normal distribution',
              {'number_of_vars': '1', 'name_prefix': '',
               'loc': '0', 'scale': '1'}),
             ('Bernoulli distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'p': '0.5'})]}
        widget = self.create_widget(OWRandomData, settings)
        widget.sender = lambda: widget.editors[0]

        widget.remove_editor()
        data = self.get_output(widget.Outputs.data, widget)
        self.assertEqual(data.X.shape[1], 2)
        self.assertEqual(
            {var.name for var in data.domain.variables}, {"b1", "b2"})

        widget.remove_editor()
        data = self.get_output(widget.Outputs.data, widget)
        self.assertIsNone(data)

    def test_add_distribution(self):
        # pylint: disable=unsubscriptable-object
        widget = self.widget
        widget.add_editor = Mock()
        widget.add_combo.setCurrentText("Uniform distribution")
        widget.add_editor.assert_called_once()
        editor = widget.add_editor.call_args[0][0]
        self.assertIsInstance(editor, distributions["Uniform distribution"])
        self.assertEqual(widget.add_combo.currentIndex(), 0)

    def test_error_in_generate(self):
        settings = {
            "distributions":
            [('Multinomial distribution',
              {'number_of_vars': '1', 'name_prefix': '',
               'ps': '0.5, X', 'n': '10'}),  # error in prepare_variables
             ('Bernoulli distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'p': '0.5'}),
             ('Uniform distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'loc': '5', 'scale': '2'}),  # error in generate_data
             ]}
        widget = self.create_widget(OWRandomData, settings)
        data = self.get_output(widget.Outputs.data, widget)
        self.assertIsNone(data)
        self.assertTrue(widget.Error.sampling_error.is_shown())

        widget.sender = lambda: widget.editors[2]
        widget.remove_editor()
        data = self.get_output(widget.Outputs.data, widget)
        self.assertIsNone(data)
        self.assertTrue(widget.Error.sampling_error.is_shown())

        widget.sender = lambda: widget.editors[0]
        widget.remove_editor()
        data = self.get_output(widget.Outputs.data, widget)
        self.assertIsNotNone(data)
        self.assertFalse(widget.Error.sampling_error.is_shown())

    def test_pack_settings(self):
        settings = {
            "distributions":
            [('Multinomial distribution',
              {'number_of_vars': '1', 'name_prefix': '',
               'ps': '0.5, X', 'n': '10'}),
             ('Bernoulli distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'p': '0.5'}),
             ('Uniform distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'loc': '5', 'scale': '2'})
             ]}
        widget = self.create_widget(OWRandomData, settings)
        widget.sender = lambda: widget.editors[0]
        widget.remove_editor()
        widget.pack_editor_settings()
        self.assertEqual(
            widget.distributions,
            [('Bernoulli distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'p': '0.5'}),
             ('Uniform distribution',
              {'number_of_vars': '2', 'name_prefix': 'b',
               'loc': '5', 'scale': '2'})])
        widget.remove_editor()
        widget.remove_editor()
        widget.pack_editor_settings()
        self.assertEqual(widget.distributions, [])


if __name__ == "__main__":
    unittest.main()
