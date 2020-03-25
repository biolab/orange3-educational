from Orange.widgets.tests.base import WidgetTest
import Orange
from Orange.data import Domain, Table
from Orange.data.domain import ContinuousVariable
import numpy as np
from orangecontrib.educational.widgets.owkmeans import OWKmeans


class TestOWKmeans(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWKmeans)  # type: OWKmeans
        self.data = Orange.data.Table.from_file("iris")

    def test_button_labels(self):
        self.widget.set_data(self.data)
        self.widget.centroid_numbers_spinner.setValue(4)

        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[1])

        # make step
        self.widget.step_button.click()
        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[0])

        # make next step
        self.widget.step_button.click()
        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[1])

        # make step to recompute centroids and then move one centroid (in graph)
        # automatic step have to ber preformed
        self.widget.step_button.click()
        self.widget.centroid_dropped(0, 1, 1)
        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[1])

        # make step to recompute centroids and then move one centroid (in graph)
        # automatic step have to ber preformed
        self.widget.step_button.click()
        self.widget.graph_clicked(1, 1)
        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[1])

    def test_boxes_disabling(self):
        """
        Check if disabling depending on input is correct
        """

        # none input
        self.widget.set_data(None)
        self.assertEqual(self.widget.options_box.isEnabled(), False)
        self.assertEqual(self.widget.centroids_box.isEnabled(), False)
        self.assertEqual(self.widget.step_box.isEnabled(), False)
        self.assertEqual(self.widget.run_box.isEnabled(), False)

        # if data provided
        self.widget.set_data(self.data)
        self.assertEqual(self.widget.options_box.isEnabled(), True)
        self.assertEqual(self.widget.centroids_box.isEnabled(), True)
        self.assertEqual(self.widget.step_box.isEnabled(), True)
        self.assertEqual(self.widget.run_box.isEnabled(), True)

        # if too les continuous attributes
        domain = Orange.data.Domain(self.data.domain.attributes[:1],
                                    self.data.domain.class_var)
        data1 = Orange.data.Table.from_table(domain, self.data)
        self.widget.set_data(data1)
        self.assertEqual(self.widget.options_box.isEnabled(), False)
        self.assertEqual(self.widget.centroids_box.isEnabled(), False)
        self.assertEqual(self.widget.step_box.isEnabled(), False)
        self.assertEqual(self.widget.run_box.isEnabled(), False)

        # if too much clusters for data
        self.widget.number_of_clusters = 3
        self.widget.set_data(self.data[:2])
        self.assertEqual(self.widget.options_box.isEnabled(), True)
        self.assertEqual(self.widget.centroids_box.isEnabled(), True)
        self.assertEqual(self.widget.step_box.isEnabled(), False)
        self.assertEqual(self.widget.run_box.isEnabled(), False)

    def test_no_data(self):
        """
        Check if everything ok when no data
        """
        self.widget.set_data(None)
        self.assertEqual(self.widget.k_means, None)
        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)

        # if too les continuous attributes
        domain = Orange.data.Domain(self.data.domain.attributes[:1],
                                    self.data.domain.class_var)
        data1 = Orange.data.Table.from_table(domain, self.data)
        self.widget.set_data(data1)
        self.assertEqual(self.widget.k_means, None)
        self.assertEqual(self.widget.cbx.count(), 0)
        self.assertEqual(self.widget.cby.count(), 0)

    def test_combo_box(self):
        """
        Check if combo box contains proper number of attributes
        """
        num_continuous_attributes = sum(
            True for var in self.data.domain.attributes
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
        self.widget.centroid_numbers_spinner.setValue(4)

        # start autoplay
        self.widget.auto_play_button.click()

        self.assertEqual(self.widget.options_box.isEnabled(), False)
        self.assertEqual(self.widget.centroids_box.isEnabled(), False)
        self.assertEqual(self.widget.step_button.isEnabled(), False)
        self.assertEqual(self.widget.step_back_button.isEnabled(), False)
        self.assertEqual(self.widget.auto_play_button.isEnabled(), True)
        self.assertEqual(self.widget.auto_play_button.text(),
                         self.widget.AUTOPLAY_BUTTONS[1])

        # stop autoplay
        self.widget.auto_play_button.click()

        self.assertEqual(self.widget.options_box.isEnabled(), True)
        self.assertEqual(self.widget.centroids_box.isEnabled(), True)
        self.assertEqual(self.widget.step_button.isEnabled(), True)
        self.assertEqual(self.widget.auto_play_button.isEnabled(), True)
        self.assertEqual(self.widget.auto_play_button.text(),
                         self.widget.AUTOPLAY_BUTTONS[0])

    def test_centroids_change(self):
        """
        Test if number of centroid in k-means changes correctly when adding,
        deleting centroids
        """
        self.widget.set_data(self.data)
        self.widget.centroid_numbers_spinner.setValue(4)
        self.assertEqual(self.widget.k_means.k, 4)

        self.widget.centroid_numbers_spinner.setValue(5)
        self.assertEqual(self.widget.k_means.k, 5)

        self.widget.centroid_numbers_spinner.setValue(3)
        self.assertEqual(self.widget.k_means.k, 3)

        self.widget.centroid_dropped(0, 1, 1)
        self.assertEqual(self.widget.k_means.k, 3)

        self.widget.graph_clicked(1, 1)
        self.assertEqual(self.widget.k_means.k, 4)

    def test_step_back(self):
        """
        Test if step back restore same positions
        """
        self.widget.set_data(self.data)
        self.widget.centroid_numbers_spinner.setValue(4)
        before_centroids = np.copy(self.widget.k_means.centroids)
        before_clusters = np.copy(self.widget.k_means.clusters)
        self.widget.step_button.click()
        self.widget.step_back_button.click()
        after_centroids = np.copy(self.widget.k_means.centroids)
        after_clusters = np.copy(self.widget.k_means.clusters)

        np.testing.assert_equal(before_centroids, after_centroids)
        np.testing.assert_equal(before_clusters, after_clusters)

        # few click deeper
        self.widget.step_button.click()
        self.widget.step_button.click()
        self.widget.step_button.click()
        self.widget.step_button.click()
        self.widget.step_button.click()
        self.widget.step_back_button.click()
        self.widget.step_back_button.click()
        self.widget.step_back_button.click()
        self.widget.step_back_button.click()
        self.widget.step_back_button.click()
        after_centroids = np.copy(self.widget.k_means.centroids)
        after_clusters = np.copy(self.widget.k_means.clusters)

        np.testing.assert_equal(before_centroids, after_centroids)
        np.testing.assert_equal(before_clusters, after_clusters)

    def test_restart(self):
        self.widget.set_data(self.data)
        kmeans_before = self.widget.k_means
        self.widget.restart_button.click()

        # check if instantiated new k-means
        self.assertNotEqual(self.widget.k_means, kmeans_before)
        self.assertEqual(self.widget.k_means.step_no, 0)
        self.assertEqual(self.widget.k_means.k, self.widget.number_of_clusters)

    def test_button_text_change(self):
        self.widget.set_data(self.data)
        self.widget.centroid_numbers_spinner.setValue(4)
        self.widget.button_text_change()
        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[1])
        self.assertEqual(self.widget.step_back_button.isEnabled(), False)

        self.widget.step_button.click()
        self.widget.button_text_change()
        self.assertEqual(self.widget.step_button.text(),
                         self.widget.STEP_BUTTONS[0])
        self.assertEqual(self.widget.step_back_button.isEnabled(), True)

        self.widget.auto_play_button.click()
        self.widget.button_text_change()
        self.assertEqual(self.widget.step_back_button.isEnabled(), False)
        self.widget.auto_play_button.click()  # stop autoplay
        self.widget.button_text_change()
        self.assertEqual(self.widget.step_back_button.isEnabled(), True)

    def test_replot(self):
        self.widget.replot()
        # 1 because graph cleaned on the beginning
        self.assertEqual(self.widget.scatter.count_replots, 1)
        self.widget.set_data(self.data)
        self.assertEqual(self.widget.scatter.count_replots, 2)
        self.widget.step_button.click()
        self.assertEqual(self.widget.scatter.count_replots, 2)
        # still 2 because just centroids moved not complete replot

        self.widget.lines_checkbox.nextCheckState()
        self.assertEqual(self.widget.scatter.count_replots, 3)

        self.widget.step_button.click()
        self.assertEqual(self.widget.scatter.count_replots, 4)
        # complete replot because it is cluster change

        self.widget.step_button.click()
        self.assertEqual(self.widget.scatter.count_replots, 4)
        # just move centroids

    def test_number_of_clusters_change(self):
        # provide less data than clusters to check
        # if k-menas initiated after that
        self.widget.centroid_numbers_spinner.setValue(5)
        self.widget.set_data(self.data[:3])
        self.widget.centroid_numbers_spinner.setValue(1)
        self.assertNotEqual(self.widget.k_means, None)

        self.widget.set_data(self.data)
        self.assertEqual(self.widget.k_means.k, self.widget.number_of_clusters)
        self.widget.centroid_numbers_spinner.setValue(1)
        # ok if number of clusters unchanged
        self.assertEqual(self.widget.k_means.k, self.widget.number_of_clusters)
        self.widget.centroid_numbers_spinner.setValue(5)
        self.assertEqual(self.widget.k_means.k, self.widget.number_of_clusters)
        self.widget.centroid_numbers_spinner.setValue(3)
        self.assertEqual(self.widget.k_means.k, self.widget.number_of_clusters)

    def test_send_data(self):
        self.widget.send_data()
        # some test will be added after output check enabled in testGui

    def test_replot_series(self):
        self.widget.set_data(self.data)
        self.assertEqual(self.widget.scatter.count_replots, 2)
        self.widget.replot_series()
        self.assertEqual(self.widget.scatter.count_replots, 2)
        # still 2 because just centroids moved not complete replot

        self.widget.replot_series()
        self.assertEqual(self.widget.scatter.count_replots, 2)

        self.widget.lines_checkbox.nextCheckState()
        self.widget.replot_series()
        self.assertEqual(self.widget.scatter.count_replots, 3)
        # 3 because of nextState

        self.widget.replot_series()
        self.assertEqual(self.widget.scatter.count_replots, 3)

    def test_scatter_chart_clicked(self):
        self.widget.set_data(self.data)
        self.widget.centroid_numbers_spinner.setValue(1)
        self.widget.scatter.bridge.chart_clicked(1, 2)

        self.assertEqual(self.widget.k_means.k, 2)

        self.widget.set_data(None)
        self.widget.scatter.bridge.chart_clicked(1, 2)
        self.assertEqual(self.widget.k_means.k, 2)  # no changes when no data

    def test_scatter_point_dropped(self):
        self.widget.set_data(self.data)
        self.widget.centroid_numbers_spinner.setValue(1)
        self.widget.scatter.bridge.point_dropped(0, 1, 2)

        self.assertEqual(self.widget.k_means.k, 1)

        self.assertEqual(self.widget.k_means.centroids[0].tolist(), [1, 2])

    def test_send_report(self):
        """
        Just test report not crashes
        """
        w = self.widget

        w.set_data(self.data)
        w.centroid_numbers_spinner.setValue(3)
        self.process_events(lambda: w.scatter.svg())
        w.report_button.click()

        self.assertIn("Number of centroids", w.report_html)

        # if no data
        w.set_data(None)
        w.report_button.click()

        self.assertNotIn("Number of centroids", w.report_html)

    def test_data_nan_column(self):
        """
        Do not crash when a column has a nan value.
        GH-40
        """
        data = self.data
        domain = Domain(attributes=data.domain.attributes[:2], class_vars=data.domain.class_vars)
        data = data.transform(domain)
        data[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)


if __name__ == "__main__":
    import unittest
    unittest.main()
