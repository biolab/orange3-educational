import unittest
from unittest.mock import Mock

import numpy as np

from Orange.widgets.tests.base import WidgetTest
from Orange.data import Domain, ContinuousVariable, Table

from orangecontrib.educational.widgets import owkmeans
from orangecontrib.educational.widgets.owkmeans import OWKmeans

# No animations
owkmeans.AnimateNumpy.__call__ = lambda self: self.done(self.final)


class TestOWKmeans(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWKmeans)  # type: OWKmeans
        self.data = Table("iris")
        domain = Domain(attributes=self.data.domain.attributes[:2],
                        class_vars=self.data.domain.class_vars)
        self.data2 = self.data.transform(domain)
        domain = Domain([ContinuousVariable(n) for n in "xy"])
        self.dataxy = Table.from_numpy(domain, np.arange(20).reshape(10, 2))

    def test_simplify_widget(self):
        def on_iris(nattrs):
            self.assertIs(w.attr_x, self.data.domain[0])
            self.assertIs(w.attr_y, self.data.domain[1])
            self.assertIs(w.variables_box.isHidden(), nattrs <= 2)
            axes = ("bottom", "left")
            for axis, attr in zip(axes, (w.attr_x, w.attr_y)):
                axis = w.plot.getAxis(axis)
                self.assertTrue(axis.label.isVisible())
                self.assertTrue(axis.label.toPlainText(), attr.name)

        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        on_iris(4)

        self.send_signal(w.Inputs.data, self.data2)
        on_iris(2)

        self.send_signal(w.Inputs.data, self.data)
        on_iris(4)

        self.send_signal(w.Inputs.data, self.dataxy)
        self.assertIs(w.attr_x, self.dataxy.domain[0])
        self.assertIs(w.attr_y, self.dataxy.domain[1])
        self.assertTrue(w.variables_box.isHidden())
        axes = ("bottom", "left")
        for axis, attr in zip(axes, (w.attr_x, w.attr_y)):
            axis = w.plot.getAxis(axis)
            self.assertFalse(axis.label.isVisible())

        self.send_signal(w.Inputs.data, self.data)
        on_iris(4)

    def test_step(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        w.send_data = Mock()
        w._update_buttons = Mock()

        kmeans = w.k_means
        klass = type(w.k_means)
        w.step()
        w.send_data.assert_called()
        w._update_buttons.assert_called()
        w.send_data.reset_mock()
        w._update_buttons.reset_mock()
        self.assertIs(kmeans.history[-1].step, klass.move_centroids)

        w.step()
        w.send_data.assert_called()
        w._update_buttons.assert_called()
        w.send_data.reset_mock()
        w._update_buttons.reset_mock()
        self.assertIs(kmeans.history[-1].step, klass.assign_membership)

    def test_step_back(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.data)
        kmeans = w.k_means

        centroids0 = kmeans.centroids.copy()
        clusters0 = kmeans.clusters.copy()

        w.step()
        centroids1 = kmeans.centroids.copy()
        clusters1 = kmeans.clusters.copy()

        w.step()

        assert not np.all(centroids0 == centroids1)
        w.send_data = Mock()
        w._update_buttons = Mock()

        w.step_back()
        w.send_data.assert_called()
        w._update_buttons.assert_called()
        w.send_data.reset_mock()
        w._update_buttons.reset_mock()
        np.testing.assert_equal(kmeans.centroids, centroids1)
        np.testing.assert_equal(kmeans.clusters, clusters1)

        w.step_back()
        w.send_data.assert_called()
        w._update_buttons.assert_called()
        w.send_data.reset_mock()
        w._update_buttons.reset_mock()
        np.testing.assert_equal(kmeans.centroids, centroids0)
        np.testing.assert_equal(kmeans.clusters, clusters0)

    def test_autoplay_buttons(self):
        """
        Check if everything goes correct for autorun
        """
        self.widget.set_data(self.data)
        self.widget.number_of_clusters = 4

        # start autoplay
        self.widget.auto_play_button.click()

        self.assertEqual(self.widget.variables_box.isEnabled(), False)
        self.assertEqual(self.widget.restart_button.isEnabled(), False)
        self.assertEqual(self.widget.step_button.isEnabled(), False)
        self.assertEqual(self.widget.step_back_button.isEnabled(), False)
        self.assertEqual(self.widget.auto_play_button.isEnabled(), True)
        self.assertEqual(self.widget.auto_play_button.text(), "Stop")

        # stop autoplay
        self.widget.auto_play_button.click()

        self.assertEqual(self.widget.variables_box.isEnabled(), True)
        self.assertEqual(self.widget.restart_button.isEnabled(), True)
        self.assertEqual(self.widget.step_button.isEnabled(), True)
        self.assertEqual(self.widget.auto_play_button.isEnabled(), True)
        self.assertNotEqual(self.widget.auto_play_button.text(), "Stop")

    def test_centroids_change(self):
        self.widget.number_of_clusters = 4
        self.widget.set_data(self.data)
        self.assertEqual(self.widget.k_means.k, 4)
        self.assertEqual(self.widget.number_of_clusters, 4)
        np.testing.assert_equal(self.widget.color_map, [0, 1, 2, 3])

        self.widget.on_centroid_clicked(1)
        self.assertEqual(self.widget.number_of_clusters, 3)
        np.testing.assert_equal(self.widget.color_map, [0, 2, 3])

        self.widget.on_centroid_add(2, 3)
        self.assertEqual(self.widget.number_of_clusters, 4)
        np.testing.assert_equal(self.widget.color_map, [0, 2, 3, 1])

    def test_no_data(self):
        self.widget.set_data(None)
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.widget.reduced_data)
        self.assertIsNone(self.widget.k_means)
        self.assertEqual(self.widget.var_model.rowCount(), 0)

        # too few continuous attributes: don't create reduced data and k-means
        domain = Domain(self.data.domain.attributes[:1],
                        self.data.domain.class_var)
        data1 = Table.from_table(domain, self.data)
        self.widget.set_data(data1)
        self.assertIsNone(self.widget.reduced_data)
        self.assertIsNone(self.widget.k_means)
        self.assertTrue(self.widget.Error.num_features.is_shown())

    def test_data_no_non_nan_data(self):
        self.dataxy[:5, 0] = np.nan
        self.dataxy[5:, 1] = np.nan
        self.send_signal(self.widget.Inputs.data, self.dataxy)
        self.assertIsNone(self.widget.reduced_data)
        self.assertIsNone(self.widget.k_means)
        self.assertTrue(self.widget.Error.no_nonnan_data.is_shown())

    def test_restart(self):
        self.widget.set_data(self.data)
        prev_centroids = self.widget.k_means.centroids
        self.widget.restart_button.click()
        self.assertFalse(np.all(prev_centroids - self.widget.k_means.centroids == 0))

    def test_send_report(self):
        w = self.widget
        w.set_data(self.data)
        w.report_button.click()


if __name__ == "__main__":
    unittest.main()
