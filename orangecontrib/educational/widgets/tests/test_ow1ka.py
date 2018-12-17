from Orange.widgets.tests.base import WidgetTest
from orangecontrib.educational.widgets.ow1ka import OW1ka


class TestOW1ka(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OW1ka)
        self.widget.combo.addItem('https://www.1ka.si/podatki/139234/A4228E24/')
        self.widget.load_url()

    def test_output_data(self):
        self.widget.commit()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 5)