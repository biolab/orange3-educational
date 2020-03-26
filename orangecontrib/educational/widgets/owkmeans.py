import numpy as np
from os import path
from itertools import chain
import time

from AnyQt.QtCore import pyqtSlot, QThread, Qt, pyqtSignal, QObject

import Orange
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.data import DiscreteVariable, Table, Domain
from Orange.widgets import gui, settings
from orangewidget.report import report


from orangecontrib.educational.widgets.utils.kmeans import Kmeans
from orangecontrib.educational.widgets.utils.color_transform import \
    rgb_hash_brighter
from orangecontrib.educational.widgets.highcharts import Highchart


class Autoplay(QThread):
    """
    Class used for separated thread when using "Autoplay" for k-means

    Parameters
    ----------
    owkmeans : OWKmeans
        Instance of OWKmeans class
    """

    def __init__(self, owkmeans):
        QThread.__init__(self)
        self.owkmeans = owkmeans
        self.is_running = True

    def __del__(self):
        self.wait()

    def stop(self):
        self.is_running = False

    def run(self):
        """
        Stepping through the algorithm until converge or user interrupts
        """
        while (self.is_running and
               not self.owkmeans.k_means.converged and
               self.owkmeans.auto_play_enabled):
            try:
                self.owkmeans.step_trigger.emit()
            except RuntimeError:
                return
            time.sleep(2 - self.owkmeans.auto_play_speed)
        self.owkmeans.stop_auto_play_trigger.emit()


class Scatterplot(Highchart):
    """
    Scatterplot extends Highchart and just defines some sane defaults:
    * enables scroll-wheel zooming,
    * set callback functions for click (in empty chart), drag and drop
    * enables moving of centroids points
    * include drag_drop_js script by highchart
    """
    prew_time = 0

    # to make unit tesest
    count_replots = 0

    def __init__(self, click_callback, drop_callback, **kwargs):

        # read javascript for drag and drop
        with open(path.join(path.dirname(__file__), 'resources', 'draggable-points.js'),
                  encoding='utf-8') as f:
            drag_drop_js = f.read()

        class Bridge(QObject):
            @pyqtSlot(float, float)
            def chart_clicked(_, x, y):
                self.click_callback(x, y)

            @pyqtSlot(int, float, float)
            def point_dropped(_, index, x, y):
                self.drop_callback(index, x, y)

        super().__init__(
            enable_zoom=False,
            bridge=Bridge(),
            enable_select='',
            plotOptions_series_point_events_drop="""/**/(function(event) {
                var index = this.series.data.indexOf( this );
                window.pybridge.point_dropped(index, this.x, this.y);
                return false;
            })
            """,
            plotOptions_series_states_hover_enabled=False,
            plotOptions_series_showInLegend=False,
            javascript=drag_drop_js,
            **kwargs)

        self.click_callback = click_callback
        self.drop_callback = drop_callback

    def chart(self, *args, **kwargs):
        self.count_replots += 1
        super(Scatterplot, self).chart(*args, **kwargs)

    def update_series(self, series_no, data):
        for i, d in enumerate(data):
            self.evalJS("""chart.series[%d].points[%d].update({x: %f, y: %f},
                        %s,
                        {duration: 500, easing: 'linear'})"""
                        % (series_no, i, d[0], d[1],
                           ("true" if i == len(data) - 1 else "false")))

    def remove_last_series(self, no):
        self.evalJS("""for(var i = 0; i < %d; i++)
                    chart.series[1].remove(true);""" % no)

    def add_series(self, series):
        for i, s in enumerate(series):
            self.exposeObject('series%d' % i, series[i])
            self.evalJS("chart.addSeries('series%d', true);" % i)


class OWKmeans(OWWidget):
    """
    K-means widget
    """

    name = "Interactive k-Means"
    description = "Widget demonstrates working of k-means algorithm."
    keywords = ["kmeans", "clustering", "interactive"]
    icon = "icons/InteractiveKMeans.svg"
    want_main_area = False
    priority = 300

    # inputs and outputs
    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output("Annotated Data", Table, default=True)
        centroids = Output("Centroids", Table)

    class Warning(OWWidget.Warning):
        num_features = Msg("Widget requires at least two numeric features with valid values")
        cluster_points = Msg("The number of clusters can't exceed the number of points")

    # settings
    number_of_clusters = settings.Setting(3)
    auto_play_enabled = False
    auto_play_thread = None

    # data
    data = None
    selected_rows = None  # rows that are selected for kmeans (not nan rows)

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')

    # other settings
    k_means = None
    auto_play_speed = settings.Setting(1)
    lines_to_centroids = settings.Setting(True)
    graph_name = 'scatter'
    output_name = "cluster"
    STEP_BUTTONS = ["Reassign Membership", "Recompute Centroids"]
    AUTOPLAY_BUTTONS = ["Run", "Stop"]

    # colors taken from chart.options.colors in Highchart
    # (if more required check for more in chart.options.color)
    colors = ["#1F7ECA", "#D32525", "#28D825", "#D5861F", "#98257E",
              "#2227D5", "#D5D623", "#D31BD6", "#6A7CDB", "#78D5D4"]

    # signals
    step_trigger = pyqtSignal()
    stop_auto_play_trigger = pyqtSignal()

    def __init__(self):
        super().__init__()

        # options box
        self.options_box = gui.widgetBox(self.controlArea, "Data")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal,
            callback=self.restart, sendSelectedValue=True,
            )

        self.cbx = gui.comboBox(value='attr_x', label='X: ', **opts)
        self.cby = gui.comboBox(value='attr_y', label='Y: ', **opts)

        self.centroids_box = gui.widgetBox(self.controlArea, "Centroids")
        self.centroid_numbers_spinner = gui.spin(
            self.centroids_box, self, 'number_of_clusters',
            minv=1, maxv=10, step=1, label='Number of centroids:',
            alignment=Qt.AlignRight, callback=self.number_of_clusters_change)
        self.restart_button = gui.button(
            self.centroids_box, self, "Randomize Positions",
            callback=self.restart)
        gui.separator(self.centroids_box)
        self.lines_checkbox = gui.checkBox(
            self.centroids_box, self, 'lines_to_centroids',
            'Show membership lines', callback=self.complete_replot)

        # control box
        gui.separator(self.controlArea, 20, 20)
        self.step_box = gui.widgetBox(self.controlArea, "Manually step through")
        self.step_button = gui.button(
            self.step_box, self, self.STEP_BUTTONS[1], callback=self.step)
        self.step_back_button = gui.button(
            self.step_box, self, "Step Back", callback=self.step_back)

        self.run_box = gui.widgetBox(self.controlArea, "Run")

        self.auto_play_speed_spinner = gui.hSlider(
            self.run_box, self, 'auto_play_speed', label='Speed:',
            minValue=0, maxValue=1.91, step=0.1, intOnly=False,
            createLabel=False)
        self.auto_play_button = gui.button(
            self.run_box, self, self.AUTOPLAY_BUTTONS[0],
            callback=self.auto_play)

        gui.rubber(self.controlArea)

        # disable until data loaded
        self.set_disabled_all(True)

        # graph in mainArea
        self.scatter = Scatterplot(
            click_callback=self.graph_clicked,
            drop_callback=self.centroid_dropped,
            xAxis_gridLineWidth=0, yAxis_gridLineWidth=0,
            tooltip_enabled=False,
            debug=False)

        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

    def concat_x_y(self):
        """
        Function takes two selected columns from data table and merge them in
        new Orange.data.Table

        Returns
        -------
        Orange.data.Table
            table with selected columns
        """
        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.Y if subset.Y.size else subset.X)
        x = np.column_stack(cols)
        not_nan = ~np.isnan(x).any(axis=1)
        x = x[not_nan]  # remove rows with nan
        self.selected_rows = np.where(not_nan)
        domain = Domain([attr_x, attr_y])
        return Table.from_numpy(domain, x)

    def set_empty_plot(self):
        self.scatter.clear()

    def set_disabled_all(self, disabled):
        """
        Function disable all controls
        """
        self.options_box.setDisabled(disabled)
        self.centroids_box.setDisabled(disabled)
        self.step_box.setDisabled(disabled)
        self.run_box.setDisabled(disabled)

    @Inputs.data
    def set_data(self, data):
        """
        Function receives data from input and init part of widget if data are
        ok. Otherwise set empty plot and notice
        user about that

        Parameters
        ----------
        data : Orange.data.Table or None
            input data
        """
        self.data = data

        def get_valid_attributes(data):
            attrs = [var for var in data.domain.attributes if var.is_continuous]
            return [var for var in attrs if sum(~np.isnan(data[:, var])) > 0]

        def reset_combos():
            self.cbx.clear()
            self.cby.clear()

        def init_combos():
            """
            function initialize the combos with attributes
            """
            reset_combos()
            valid_class_vars = [var for var in data.domain.class_vars
                                if data is not None and var.is_continuous]
            for var in chain(valid_attributes, valid_class_vars):
                self.cbx.addItem(gui.attributeIconDict[var], var.name)
                self.cby.addItem(gui.attributeIconDict[var], var.name)

        # remove warnings about too less continuous attributes and not enough data
        self.Warning.clear()

        if self.auto_play_thread:
            self.auto_play_thread.stop()

        if data is None or len(data) == 0:
            reset_combos()
            self.set_empty_plot()
            self.set_disabled_all(True)
            return

        valid_attributes = get_valid_attributes(data)

        if len(valid_attributes) < 2:
            reset_combos()
            self.Warning.num_features()
            self.set_empty_plot()
            self.set_disabled_all(True)
        else:
            init_combos()
            self.set_disabled_all(False)
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            if self.k_means is None:
                self.k_means = Kmeans(self.concat_x_y())
            else:
                self.k_means.set_data(self.concat_x_y())
            self.number_of_clusters_change()

    def restart(self):
        """
        Function triggered on data change or restart button pressed
        """
        self.k_means = Kmeans(self.concat_x_y())
        self.number_of_clusters_change()

    def step(self):
        """
        Function called on every step
        """
        self.k_means.step()
        self.replot()
        self.button_text_change()
        self.send_data()

    def step_back(self):
        """
        Function called for step back
        """
        self.k_means.step_back()
        self.replot()
        self.button_text_change()
        self.send_data()
        self.number_of_clusters = self.k_means.k

    def button_text_change(self):
        """
        Function changes text on ste button and chanbe the button text
        """
        self.step_button.setText(self.STEP_BUTTONS[self.k_means.step_completed])
        if self.k_means.step_no <= 0:
            self.step_back_button.setDisabled(True)
        elif not self.auto_play_enabled:
            self.step_back_button.setDisabled(False)

    def auto_play(self):
        """
        Function called when autoplay button pressed
        """
        self.auto_play_enabled = not self.auto_play_enabled
        self.auto_play_button.setText(
            self.AUTOPLAY_BUTTONS[self.auto_play_enabled])
        if self.auto_play_enabled:
            self.options_box.setDisabled(True)
            self.centroids_box.setDisabled(True)
            self.step_box.setDisabled(True)
            self.auto_play_thread = Autoplay(self)
            self.step_trigger.connect(self.step)
            self.stop_auto_play_trigger.connect(self.stop_auto_play)
            self.auto_play_thread.start()
        else:
            self.stop_auto_play()

    def stop_auto_play(self):
        """
        Called when stop autoplay button pressed or in the end of autoplay
        """
        self.options_box.setDisabled(False)
        self.centroids_box.setDisabled(False)
        self.step_box.setDisabled(False)
        self.auto_play_enabled = False
        self.auto_play_button\
            .setText(self.AUTOPLAY_BUTTONS[self.auto_play_enabled])
        self.button_text_change()

    def replot(self):
        """
        Function refreshes the chart
        """
        if self.data is None or not self.attr_x or not self.attr_y:
            return

        km = self.k_means
        if not km.centroids_moved:
            self.complete_replot()
            return

        # when centroids moved during step
        self.scatter.update_series(0, self.k_means.centroids)

        if self.lines_to_centroids:
            for i, (c, pts) in enumerate(zip(
                    km.centroids, km.centroids_belonging_points)):
                self.scatter.update_series(1 + i, list(chain.from_iterable(
                    ([p[0], p[1]], [c[0], c[1]])
                    for p in pts)))

    def complete_replot(self):
        """
        This function performs complete replot of the graph without animation
        """
        try:
            attr_x = self.data.domain[self.attr_x]
            attr_y = self.data.domain[self.attr_y]
        except KeyError:
            return

        # plot centroids
        options = dict(series=[])
        n_colors = len(self.colors)
        km = self.k_means
        options['series'].append(
            dict(
                data=[{'x': p[0], 'y': p[1],
                       'marker':{'fillColor': self.colors[i % n_colors]}}
                      for i, p in enumerate(km.centroids)],
                type="scatter",
                draggableX=True,
                draggableY=True,
                cursor="move",
                zIndex=10,
                marker=dict(symbol='square', radius=8)))

        # plot lines between centroids and points
        if self.lines_to_centroids:
            for i, (c, pts) in enumerate(zip(
                    km.centroids, km.centroids_belonging_points)):
                options['series'].append(dict(
                    data=list(
                        chain.from_iterable(([p[0], p[1]], [c[0], c[1]])
                                            for p in pts)),
                    type="line",
                    lineWidth=0.2,
                    enableMouseTracking=False,
                    color="#ccc"))

        # plot data points
        for i, points in enumerate(km.centroids_belonging_points):
            options['series'].append(dict(
                data=points,
                type="scatter",
                color=rgb_hash_brighter(
                    self.colors[i % len(self.colors)], 0.3)))

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y))

        # plot
        self.scatter.chart(options, **kwargs)

    def replot_series(self):
        """
        This function replot just series connected with centroids and
        uses animation for that
        """
        km = self.k_means
        k = km.k

        series = []
        # plot lines between centroids and points
        if self.lines_to_centroids:
            for i, (c, pts) in enumerate(zip(
                    km.centroids, km.centroids_belonging_points)):
                series.append(dict(
                   data=list(
                       chain.from_iterable(([p[0], p[1]], [c[0], c[1]])
                                           for p in pts)),
                   type="line",
                   showInLegend=False,
                   lineWidth=0.2,
                   enableMouseTracking=False,
                   color="#ccc"))

        # plot data points
        for i, points in enumerate(km.centroids_belonging_points):
            series.append(dict(
                data=points,
                type="scatter",
                showInLegend=False,
                color=rgb_hash_brighter(
                    self.colors[i % len(self.colors)], 0.5)))

        self.scatter.add_series(series)

        self.scatter.remove_last_series(k * 2 if self.lines_to_centroids else k)

    def number_of_clusters_change(self):
        """
        Function that change number of clusters if required
        """
        if self.data is None:
            return
        if self.number_of_clusters > len(self.data):
            # if too less data for clusters number
            self.Warning.cluster_points()
            self.set_empty_plot()
            self.step_box.setDisabled(True)
            self.run_box.setDisabled(True)
        else:
            self.Warning.cluster_points.clear()
            self.step_box.setDisabled(False)
            self.run_box.setDisabled(False)
            if self.k_means is None:  # if before too less data k_means is None
                self.k_means = Kmeans(self.concat_x_y())
            if self.k_means.k < self.number_of_clusters:
                self.k_means.add_centroids(
                    self.number_of_clusters - self.k_means.k)
            elif not self.k_means.k == self.number_of_clusters:
                self.k_means.delete_centroids(
                    self.k_means.k - self.number_of_clusters)
            self.replot()
            self.send_data()
        self.button_text_change()

    def graph_clicked(self, x, y):
        """
        Function called when user click in graph. Centroid have to be added.
        """
        if self.k_means is not None and self.data is not None:
            self.k_means.add_centroids([x, y])
            self.number_of_clusters += 1
            self.replot()
            self.send_data()
            self.button_text_change()

    def centroid_dropped(self, _index, x, y):
        """
        Function called when centroid with _index moved.
        """
        self.k_means.move_centroid(_index, x, y)
        self.complete_replot()
        self.send_data()
        self.button_text_change()

    def send_data(self):
        """
        Function sends data with clusters column and data with centroids
        position to the output
        """
        km = self.k_means
        if km is None or km.clusters is None:
            self.Outputs.annotated_data.send(None)
            self.Outputs.centroids.send(None)
        else:
            clust_var = DiscreteVariable(
                self.output_name,
                values=["C%d" % (x + 1) for x in range(km.k)])
            attributes = self.data.domain.attributes
            classes = self.data.domain.class_vars
            meta_attrs = self.data.domain.metas
            if classes:
                meta_attrs += classes
            classes = [clust_var]
            domain = Domain(attributes, classes, meta_attrs)
            annotated_data = Table.from_table(domain, self.data)
            annotated_data.Y[self.selected_rows] = km.clusters

            centroids = Table.from_numpy(
                Domain(km.data.domain.attributes), km.centroids
            )
            self.Outputs.annotated_data.send(annotated_data)
            self.Outputs.centroids.send(centroids)

    def send_report(self):
        if self.data is None:
            return
        caption = report.render_items_vert((
             ("Number of centroids:", self.number_of_clusters),
        ))
        self.report_plot(self.scatter)
        self.report_caption(caption)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWKmeans).run(Table.from_file('iris'))
