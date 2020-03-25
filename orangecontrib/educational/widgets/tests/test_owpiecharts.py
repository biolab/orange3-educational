from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from orangecontrib.educational.widgets.owpiecharts import OWPieChart


class TestOWPieChart(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWPieChart)
        self.iris = Table.from_file("iris")
        self.titanic = Table.from_file("titanic")

    def test_widget_load(self):
        self.assertIsNotNone(self.widget)

    def test_set_data(self):
        w = self.widget

        self.send_signal(w.Inputs.data, self.iris)
        self.assertIsNotNone(w.dataset)

        # send some none data
        self.send_signal(w.Inputs.data, None)
        self.assertIsNone(w.dataset)

        # send some empty data set
        data = Table.from_list(Domain([]), [])
        self.send_signal(w.Inputs.data, data)
        self.assertIsNone(w.dataset)

        # dataset with more discrete variables
        self.send_signal(w.Inputs.data, self.titanic)
        self.assertIsNotNone(w.dataset)
        self.assertEqual(len(w.attrs), 4)

        # dataset with no discrete variables
        data = Table.from_list(Domain([ContinuousVariable("a"),
                                       ContinuousVariable("b")]),
                               [[1, 2], [2, 2]])
        self.send_signal(w.Inputs.data, data)
        self.assertIsNotNone(w.dataset)
        self.assertEqual(len(w.attrs), 0)

        # dataset with only one value per variable
        data = Table.from_list(Domain([DiscreteVariable("a", ['1'])]),
                               [['1'], ['1'], ['1']])
        self.send_signal(w.Inputs.data, data)
        self.assertIsNotNone(w.dataset)
        self.assertEqual(len(w.attrs), 1)

    def test_report(self):
        w = self.widget
        w.send_report()

        self.send_signal(w.Inputs.data, self.iris)

        w.send_report()

        w.split_var = w.split_vars[1]
        w.update_scene()
        w.send_report()

    def test_explode_pies(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)

        # explode combo
        w.explode = True
        w.update_scene()

        # back to normal mode
        w.explode = False
        w.update_scene()

    def test_split(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)

        # split can be possible only over discrete variables. Iris has 1
        # (+ 1 for None)
        self.assertEqual(w.split_combobox.count(), 1 + 1)
        self.assertEqual(w.split_combobox.currentText(), "None")

        # change selection to "iris"
        w.split_var = w.split_vars[1]
        w.update_scene()
        self.assertEqual(w.split_combobox.currentText(), "iris")
        self.assertEqual(str(w.split_var), "iris")

        # remove everything from combo
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(w.split_combobox.count(), 1)  # only non in combo
