from Orange.widgets.tests.base import GuiTest
import Orange
from Orange.data.domain import ContinuousVariable
from Orange.widgets.classify.owadaboost import OWAdaBoostClassification
from PyQt4 import QtGui
from orangecontrib.educational.widgets.owkmeans import OWKmeans


class TestOWKmeans(GuiTest):

    def setUp(self):
        self.widget = OWKmeans()
        self.data = Orange.data.Table("iris")

    def test_button_labels(self):
        self.widget.set_data(self.data)
        self.widget.centroidNumbersSpinner.setValue(4)

        self.assertEqual(self.widget.stepButton.text(), self.widget.button_labels["step2"])

        # make step
        self.widget.stepButton.click()
        self.assertEqual(self.widget.stepButton.text(), self.widget.button_labels["step1"])

        # make next step
        self.widget.stepButton.click()
        self.assertEqual(self.widget.stepButton.text(), self.widget.button_labels["step2"])

        # make step to recompute centroids and then move one centroid (in graph)
        # automatic step have to ber preformed
        self.widget.stepButton.click()
        self.widget.centroid_dropped(0, 1, 1)
        self.assertEqual(self.widget.stepButton.text(), self.widget.button_labels["step2"])

        # make step to recompute centroids and then move one centroid (in graph)
        # automatic step have to ber preformed
        self.widget.stepButton.click()
        self.widget.graph_clicked(1, 1)
        self.assertEqual(self.widget.stepButton.text(), self.widget.button_labels["step2"])

    def test_boxes_disabling(self):
        """
        Check if disabling depending on input is correct
        """

        # none input
        self.widget.set_data(None)
        self.assertEqual(self.widget.optionsBox.isEnabled(), False)
        self.assertEqual(self.widget.centroidsBox.isEnabled(), False)
        self.assertEqual(self.widget.commandsBox.isEnabled(), False)

        # if data provided
        self.widget.set_data(self.data)
        self.assertEqual(self.widget.optionsBox.isEnabled(), True)
        self.assertEqual(self.widget.centroidsBox.isEnabled(), True)
        self.assertEqual(self.widget.commandsBox.isEnabled(), True)

        # if too les continuous attributes
        domain = Orange.data.Domain(self.data.domain.attributes[:1], self.data.domain.class_var)
        data1 = Orange.data.Table(domain, self.data)
        self.widget.set_data(data1)
        self.assertEqual(self.widget.optionsBox.isEnabled(), False)
        self.assertEqual(self.widget.centroidsBox.isEnabled(), False)
        self.assertEqual(self.widget.commandsBox.isEnabled(), False)

        # if too much clusters for data
        self.widget.numberOfClusters = 3
        self.widget.set_data(self.data[:2])
        self.assertEqual(self.widget.optionsBox.isEnabled(), True)
        self.assertEqual(self.widget.centroidsBox.isEnabled(), True)
        self.assertEqual(self.widget.commandsBox.isEnabled(), False)

    def test_no_data(self):
        """
        Check if everything ok when no data
        """
        self.widget.set_data(None)
        self.assertEqual(self.widget.k_means, None)
        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)

        # if too les continuous attributes
        domain = Orange.data.Domain(self.data.domain.attributes[:1], self.data.domain.class_var)
        data1 = Orange.data.Table(domain, self.data)
        self.widget.set_data(data1)
        self.assertEqual(self.widget.k_means, None)
        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)

    def test_combo_box(self):
        """
        Check if combo box contains proper number of attributes
        """
        num_continuous_attributes = sum(True for var in self.data.domain.attributes
                                        if isinstance(var, ContinuousVariable))
        self.widget.set_data(self.data)
        if num_continuous_attributes < 2:
            self.assertEqual(self.widget.cbx.count(), 0)
            self.assertEqual(self.widget.cby.count(), 0)
        else:
            self.assertEqual(self.widget.cbx.count(), num_continuous_attributes)
            self.assertEqual(self.widget.cby.count(), num_continuous_attributes)

    def test_autoplay(self):
        """
        Check if everything goes correct for autorun
        """
        self.widget.set_data(self.data)
        self.widget.centroidNumbersSpinner.setValue(4)

        # start autoplay
        self.widget.autoPlayButton.click()

        self.assertEqual(self.widget.optionsBox.isEnabled(), False)
        self.assertEqual(self.widget.centroidsBox.isEnabled(), False)
        self.assertEqual(self.widget.stepButton.isEnabled(), False)
        self.assertEqual(self.widget.stepBackButton.isEnabled(), False)
        self.assertEqual(self.widget.autoPlayButton.isEnabled(), True)
        self.assertEqual(self.widget.autoPlayButton.text(), self.widget.button_labels["autoplay_stop"])

        # stop autoplay
        self.widget.autoPlayButton.click()

        self.assertEqual(self.widget.optionsBox.isEnabled(), True)
        self.assertEqual(self.widget.centroidsBox.isEnabled(), True)
        self.assertEqual(self.widget.stepButton.isEnabled(), True)
        self.assertEqual(self.widget.stepBackButton.isEnabled(), True)
        self.assertEqual(self.widget.autoPlayButton.isEnabled(), True)
        self.assertEqual(self.widget.autoPlayButton.text(), self.widget.button_labels["autoplay_run"])

    def test_centroids_change(self):
        self.widget.set_data(self.data)
        self.widget.centroidNumbersSpinner.setValue(4)
        self.assertEqual(self.widget.k_means.k, 4)

        self.widget.centroidNumbersSpinner.setValue(5)
        self.assertEqual(self.widget.k_means.k, 5)

        self.widget.centroidNumbersSpinner.setValue(3)
        self.assertEqual(self.widget.k_means.k, 3)

        # make step to recompute centroids and then move one centroid (in graph)
        # automatic step have to ber preformed
        self.widget.centroid_dropped(0, 1, 1)
        self.assertEqual(self.widget.k_means.k, 3)

        # make step to recompute centroids and then move one centroid (in graph)
        # automatic step have to ber preformed
        self.widget.graph_clicked(1, 1)
        self.assertEqual(self.widget.k_means.k, 4)


