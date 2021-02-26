import unittest

import numpy as np
from Orange.data import (
    Table,
    DiscreteVariable,
    ContinuousVariable,
    TimeVariable,
)
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.educational.widgets.owcreatetable import OWCreateTable


class TestOWCreateTable(WidgetTest):
    def setUp(self):
        self.widget: OWCreateTable = self.create_widget(OWCreateTable)

    def set_data(self, data):
        self.widget.table_model.set_table(data)
        self.widget.data_changed()

    def test_output(self):
        output = self.get_output(self.widget.Outputs.data)
        # by default widget table have size 3x3 and is empty
        self.assertEqual(3, len(output))
        np.testing.assert_array_equal(np.zeros((3, 3)) * np.nan, output.X)
        self.assertListEqual(
            ["1", "2", "3"], [a.name for a in output.domain.attributes]
        )

        _input = [["1", "2", "3"], ["4", "5", "6"]]
        self.set_data(_input)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(2, len(output))
        np.testing.assert_array_equal(np.array(_input, dtype=float), output.X)

    def test_discrete_columns(self):
        """
        Test different combinations that would result in discrete columns
        """
        _input = [["a", "c", "a"], ["b", "d", "2020-01-01"], ["b", "5", "6"]]
        self.set_data(_input)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertEqual(3, len(output.domain.attributes))
        correct_values = [
            {"a", "b"},
            {"d", "c", "5"},
            {"a", "2020-01-01", "6"},
        ]
        for a, cv in zip(output.domain.attributes, correct_values):
            self.assertIsInstance(a, DiscreteVariable)
            self.assertSetEqual(cv, set(a.values))

    def test_continuous_columns(self):
        _input = [["1", "2.0", "3"], ["4", "5.0", "6"]]
        self.set_data(_input)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertEqual(3, len(output.domain.attributes))
        for a in output.domain.attributes:
            self.assertIsInstance(a, ContinuousVariable)

    def test_time_columns(self):
        _input = [
            [
                "2020-01-01",
                "2020-10-10T12:08:51",
                "2020-10-10 12:08:51",
                "2020-10-10 12:08",
            ],
            [
                "2020-01-02",
                "2020-10-10T12:09:51",
                "2020-10-10 12:09:51",
                "2020-10-10 12:09",
            ],
            [
                "2020-01-03",
                "2020-10-10T12:10:51",
                "2020-10-10 12:10:51",
                "2020-10-10 12:10",
            ],
        ]
        self.set_data(_input)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertEqual(4, len(output.domain.attributes))
        for a in output.domain.attributes:
            self.assertIsInstance(a, TimeVariable)

    def test_none_in_table(self):
        _input = [
            ["2020-01-01", "a", "1", None],
            ["2020-01-02", "b", "2", None],
            [None, None, None, None],
        ]
        self.set_data(_input)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        correct_types = [
            TimeVariable,
            DiscreteVariable,
            ContinuousVariable,
            ContinuousVariable,
        ]
        for a, typ in zip(output.domain.attributes, correct_types):
            self.assertIsInstance(a, typ)

    def test_num_col_row_change(self):
        _input = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
        self.set_data(_input)
        self.wait_until_finished()

        self.widget.r_spin.setValue(2)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output.domain.variables))
        self.assertEqual(2, len(output))
        self.assertEqual(2, self.widget.table_model.rowCount())
        self.assertEqual(3, self.widget.table_model.columnCount())

        self.widget.c_spin.setValue(2)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(2, len(output.domain.variables))
        self.assertEqual(2, len(output))
        self.assertEqual(2, self.widget.table_model.rowCount())
        self.assertEqual(2, self.widget.table_model.columnCount())
        np.testing.assert_array_equal(
            np.array(_input, dtype=float)[:2, :2], output.X
        )

        self.widget.r_spin.setValue(3)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(2, len(output.domain.variables))
        self.assertEqual(3, len(output))
        self.assertEqual(3, self.widget.table_model.rowCount())
        self.assertEqual(2, self.widget.table_model.columnCount())

        self.widget.c_spin.setValue(3)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(output.domain.variables))
        self.assertEqual(3, len(output))
        self.assertEqual(3, self.widget.table_model.rowCount())
        self.assertEqual(3, self.widget.table_model.columnCount())
        np.testing.assert_array_equal(
            np.hstack(
                (
                    np.vstack(
                        (
                            np.array(_input, dtype=float)[:2, :2],
                            np.zeros((1, 2)) * np.nan,
                        )
                    ),
                    np.zeros((3, 1)) * np.nan,
                )
            ),
            output.X,
        )

    def test_domain(self):
        iris = Table("iris")

        self.send_signal(self.widget.Inputs.data, iris)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertEqual(3, self.widget.table_model.rowCount())
        self.assertEqual(5, self.widget.table_model.columnCount())
        self.assertEqual(5, len(output.domain.variables))
        self.assertEqual(3, len(output))

        np.testing.assert_array_equal(np.zeros((3, 4)) * np.nan, output.X)
        np.testing.assert_array_equal(np.zeros((3,)) * np.nan, output.Y)

    def test_context(self):
        iris = Table("iris")

        input1_ = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]]
        self.set_data(input1_)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        np.testing.assert_array_equal(np.array(input1_, dtype=float), output.X)

        self.send_signal(self.widget.Inputs.data, iris)
        input2_ = [
            ["1", "2", "3", "4", "Iris-setosa"],
            ["4", "5", "6", "8", "Iris-setosa"],
            ["7", "8", "9", "10", "Iris-setosa"],
        ]
        self.set_data(input2_)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        np.testing.assert_array_equal(
            np.array(np.array(input2_)[:, :4], dtype=float), output.X
        )

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(np.array(input1_, dtype=float), output.X)

        self.send_signal(self.widget.Inputs.data, iris)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        np.testing.assert_array_equal(
            np.array(np.array(input2_)[:, :4], dtype=float), output.X
        )


if __name__ == "__main__":
    unittest.main()
