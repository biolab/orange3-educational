import unittest

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.educational.widgets.ow1ka import OW1ka


class TestOW1ka(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OW1ka)

    @unittest.skip("Travis has problems calling outside APIs")
    def test_output_data(self):
        self.widget.combo.addItem(
            'https://www.1ka.si/podatki/139234/A4228E24/')
        self.widget.load_url()

        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 5)

    def test_widget_load(self):
        self.assertIsNotNone(self.widget)