import Orange
from Orange.widgets.widget import OWWidget
from Orange.data import DiscreteVariable, ContinuousVariable, Table, Domain
from Orange.widgets import gui, settings, highcharts, widget
import numpy as np
from orangecontrib.educational.widgets.utils.kmeans import Kmeans
from PyQt4.QtCore import pyqtSlot, QThread, SIGNAL, Qt
from PyQt4.QtGui import QSizePolicy
from os import path
from orangecontrib.educational.widgets.utils.color_transform import rgb_hash_brighter
from itertools import chain
import time


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

    def __del__(self):
        self.wait()

    def run(self):
        """
        Stepping through the algorithm until converge or user interrupts
        """
        while not self.owkmeans.k_means.converged and self.owkmeans.auto_play_enabled:
            self.emit(SIGNAL('step()'))
            time.sleep(2 - self.owkmeans.auto_play_speed)
        self.emit(SIGNAL('stop_auto_play()'))


class Scatterplot(highcharts.Highchart):
    """
    Scatterplot extends Highchart and just defines some sane defaults:
    * enables scroll-wheel zooming,
    * set callback functions for click (in empty chart), drag and drop
    * enables moving of centroids points
    * include drag_drop_js script by highchart
    """

    js_click_function = """/**/(function(event) {
                window.pybridge.chart_clicked(event.xAxis[0].value, event.yAxis[0].value);
            })
            """

    js_drop_function = """/**/(function(event) {
                var index = this.series.data.indexOf( this );
                window.pybridge.point_dropped(index, this.x, this.y);
            })
            """

    prew_time = 0

    def __init__(self, click_callback, drop_callback, **kwargs):

        # read javascript for drag and drop
        with open(path.join(path.dirname(__file__), 'resources', 'draggable-points.js'), 'r') as f:
            drag_drop_js = f.read()

        super().__init__(enable_zoom=True,
                         bridge=self,
                         enable_select='',
                         chart_events_click=self.js_click_function,
                         plotOptions_series_point_events_drop=self.js_drop_function,
                         plotOptions_series_states_hover_enabled=False,
                         plotOptions_series_cursor="move",
                         javascript=drag_drop_js,
                         **kwargs)

        self.click_callback = click_callback
        self.drop_callback = drop_callback

    @pyqtSlot(float, float)
    def chart_clicked(self, x, y):
        self.click_callback(x, y)

    @pyqtSlot(int, float, float)
    def point_dropped(self, index, x, y):
        self.drop_callback(index, x, y)

    def update_series(self, series_no, data):
        for i, d in enumerate(data):
            self.evalJS("""chart.series[%d].points[%d].update({x: %f, y: %f},
                        %s,
                        {duration: 500, easing: 'linear'})""" % (series_no, i, d[0], d[1],
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
    icon = "icons/InteractiveKMeans.svg"
    want_main_area = False

    # inputs and outputs
    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Annotated Data", Table, widget.Default),
               ("Centroids", Table)]

    # settings
    number_of_clusters = settings.Setting(1)
    auto_play_enabled = False

    # data
    data = None

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')

    # other settings
    k_means = None
    auto_play_speed = settings.Setting(1)
    lines_to_centroids = settings.Setting(0)
    graph_name = 'scatter'
    output_name = "cluster"
    button_labels = {"step1": "Reassign membership",
                     "step2": "Recompute centroids",
                     "step_back": "Step back",
                     "autoplay_run": "Run",
                     "autoplay_stop": "Stop",
                     "random_centroids": "Randomize"}

    # colors taken from chart.options.colors in Highchart (if more required check for more in chart.options.color)
    colors = ["#1F7ECA", "#D32525", "#28D825", "#D5861F", "#98257E",
              "#2227D5", "#D5D623", "#D31BD6", "#6A7CDB", "#78D5D4"]

    def __init__(self):
        super().__init__()

        # options box
        self.options_box = gui.widgetBox(self.controlArea)
        self.cbx = gui.comboBox(self.options_box, self, 'attr_x',
                                label='X:',
                                orientation=Qt.Horizontal,
                                callback=self.restart,
                                sendSelectedValue=True)
        self.cbx.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.cby = gui.comboBox(self.options_box, self, 'attr_y',
                                label='Y:',
                                orientation='horizontal',
                                callback=self.restart,
                                sendSelectedValue=True)
        self.cby.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        self.centroids_box = gui.widgetBox(self.controlArea, "Centroids")
        self.centroid_numbers_spinner = gui.spin(self.centroids_box,
                                                 self,
                                                 'number_of_clusters',
                                                 minv=1,
                                                 maxv=10,
                                                 step=1,
                                                 label='Number of centroids:',
                                                 callback=self.number_of_clusters_change)
        self.centroid_numbers_spinner.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.restart_button = gui.button(self.centroids_box, self, self.button_labels["random_centroids"],
                                         callback=self.restart)
        self.lines_checkbox = gui.checkBox(self.centroids_box,
                                           self,
                                           'lines_to_centroids',
                                           'Show membership lines',
                                           callback=self.complete_replot)

        # control box
        self.commands_box = gui.widgetBox(self.controlArea)
        self.step_button = gui.button(self.commands_box, self, self.button_labels["step2"],
                                      callback=self.step)
        self.step_back_button = gui.button(self.commands_box, self, self.button_labels["step_back"],
                                           callback=self.step_back)
        self.auto_play_button = gui.button(self.commands_box, self, self.button_labels["autoplay_run"],
                                           callback=self.auto_play)
        self.auto_play_speed_spinner = gui.hSlider(self.commands_box,
                                                   self,
                                                   'auto_play_speed',
                                                   minValue=0,
                                                   maxValue=1.91,
                                                   step=0.1,
                                                   intOnly=False,
                                                   createLabel=False,
                                                   label='Speed:')

        gui.rubber(self.controlArea)

        # disable until data loaded
        self.set_disabled_all(True)

        # graph in mainArea
        self.scatter = Scatterplot(click_callback=self.graph_clicked,
                                   drop_callback=self.centroid_dropped,
                                   xAxis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='',
                                   tooltip_shared=False,
                                   debug=True)  # TODO: set false when end of development
        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

    def concat_x_y(self):
        """
        Function takes two selected columns from data table and merge them in new Orange.data.Table

        Returns
        -------
        Orange.data.Table
            table with selected columns
        """
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.Y if subset.Y.size else subset.X)
        x = np.column_stack(cols)
        domain = Domain([attr_x, attr_y])
        return Table(domain, x)

    def set_empty_plot(self):
        self.scatter.clear()

    def set_disabled_all(self, disabled):
        """
        Function disable all controls
        """
        self.options_box.setDisabled(disabled)
        self.centroids_box.setDisabled(disabled)
        self.commands_box.setDisabled(disabled)

    def set_data(self, data):
        """
        Function receives data from input and init part of widget if data are ok. Otherwise set empty plot and notice
        user about that

        Parameters
        ----------
        data : Orange.data.Table or None
            input data
        """
        self.data = data

        def reset_combos():
            self.cbx.clear()
            self.cby.clear()

        def init_combos():
            """
            function initialize the combos with attributes
            """
            reset_combos()
            for var in data.domain if data is not None else []:
                if var.is_primitive() and var.is_continuous:
                    self.cbx.addItem(gui.attributeIconDict[var], var.name)
                    self.cby.addItem(gui.attributeIconDict[var], var.name)

        self.warning(1)  # remove warning about too less continuous attributes if exists
        self.warning(2)  # remove warning about not enough data

        if data is None or len(data) == 0:
            reset_combos()
            self.set_empty_plot()
            self.set_disabled_all(True)
        elif sum(True for var in data.domain.attributes if isinstance(var, ContinuousVariable)) < 2:
            reset_combos()
            self.warning(1, "Too few Continuous feature. Min 2 required")
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
        self.step_button.setText(self.button_labels["step2"]
                                 if self.k_means.step_completed
                                 else self.button_labels["step1"])
        if self.k_means.step_no <= 0:
            self.step_back_button.setDisabled(True)
        elif not self.auto_play_enabled:
            self.step_back_button.setDisabled(False)

    def auto_play(self):
        """
        Function called when autoplay button pressed
        """
        self.auto_play_enabled = not self.auto_play_enabled
        self.auto_play_button.setText(self.button_labels["autoplay_stop"]
                                      if self.auto_play_enabled
                                      else self.button_labels["autoplay_run"])
        if self.auto_play_enabled:
            self.options_box.setDisabled(True)
            self.centroids_box.setDisabled(True)
            self.step_button.setDisabled(True)
            self.step_back_button.setDisabled(True)
            self.autoPlayThread = Autoplay(self)
            self.connect(self.autoPlayThread, SIGNAL("step()"), self.step)
            self.connect(self.autoPlayThread, SIGNAL("stop_auto_play()"), self.stop_auto_play)
            self.autoPlayThread.start()
        else:
            self.stop_auto_play()

    def stop_auto_play(self):
        """
        Called when stop autoplay button pressed or in the end of autoplay
        """
        self.options_box.setDisabled(False)
        self.step_button.setDisabled(False)
        self.centroids_box.setDisabled(False)
        self.step_back_button.setDisabled(False)
        self.auto_play_enabled = False
        self.auto_play_button.setText(self.button_labelsp["autoplay_stop"]
                                      if self.auto_play_enabled
                                      else self.button_labels["autoplay_run"])

    def replot(self):
        """
        Function refreshes the chart
        """
        if self.data is None or not self.attr_x or not self.attr_y:
            return

        if self.k_means.centroids_moved:
            # when centroids moved during step
            self.scatter.update_series(0, self.k_means.centroids)

            if self.lines_to_centroids:
                for i, c in enumerate(self.k_means.centroids):
                    self.scatter.update_series(1 + i, list(chain.from_iterable(
                        ([p[0], p[1]], [c[0], c[1]])
                        for p in self.k_means.centroids_belonging_points[i])))
        else:
            self.complete_replot()

    def complete_replot(self):
        """
        This function performs complete replot of the graph without animation
        """
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]

        # plot centroids
        options = dict(series=[])
        options['series'].append(dict(
            data=[{'x': p[0],
                   'y': p[1],
                   'marker':{'fillColor': self.colors[i % len(self.colors)]}}
                  for i, p in enumerate(self.k_means.centroids)],
            type="scatter",
            draggableX=True if self.k_means.step_completed else False,
            draggableY=True if self.k_means.step_completed else False,
            showInLegend=False,
            zIndex=10,
            marker=dict(symbol='diamond',
                        radius=10)))

        # plot lines between centroids and points
        if self.lines_to_centroids:
            for i, c in enumerate(self.k_means.centroids):
                options['series'].append(dict(
                    data=list(
                        chain.from_iterable(
                            ([p[0], p[1]], [c[0], c[1]])
                            for p in self.k_means.centroids_belonging_points[i])),
                    type="line",
                    showInLegend=False,
                    lineWidth=0.2,
                    enableMouseTracking=False,
                    color="#ccc"))

        # plot data points
        for i, points in enumerate(self.k_means.centroids_belonging_points):
            options['series'].append(
                dict(data=points,
                     type="scatter",
                     showInLegend=False,
                     fillOpacity=0.5,
                     color=rgb_hash_brighter(self.colors[i % len(self.colors)], 0.3)))

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
        This function replot just series connected with centroids and uses animation for that
        """
        k = self.k_means.k

        series = []
        # plot lines between centroids and points
        if self.lines_to_centroids:
            for i, c in enumerate(self.k_means.centroids):
                series.append(dict(
                   data=list(chain.from_iterable(([p[0], p[1]], [c[0], c[1]])
                                                 for p in self.k_means.centroids_belonging_points[i])),
                   type="line",
                   showInLegend=False,
                   lineWidth=0.2,
                   enableMouseTracking=False,
                   color="#ccc"))

        # plot data points
        for i, points in enumerate(self.k_means.centroids_belonging_points):
            series.append(dict(
                data=points,
                type="scatter",
                showInLegend=False,
                color=rgb_hash_brighter(self.colors[i % len(self.colors)], 0.3)))

        self.scatter.add_series(series)

        self.scatter.remove_last_series(k * 2 if self.lines_to_centroids else k)

    def number_of_clusters_change(self):
        """
        Function that change number of clusters if required
        """
        if self.number_of_clusters > len(self.data):
            # if too less data for clusters number
            self.warning(2, "Please provide at least number of points equal to "
                            "number of clusters selected or decrease number of clusters")
            self.set_empty_plot()
            self.commands_box.setDisabled(True)
        else:
            self.warning(2)
            self.commands_box.setDisabled(False)
            if self.k_means is None:  # if before too less data k_means is None
                self.k_means = Kmeans(self.concat_x_y())
            if self.k_means.k < self.number_of_clusters:
                self.k_means.add_centroids(self.number_of_clusters - self.k_means.k)
            elif not self.k_means.k == self.number_of_clusters:
                self.k_means.delete_centroids(self.k_means.k - self.number_of_clusters)
            self.replot()
            self.send_data()
        self.button_text_change()

    def graph_clicked(self, x, y):
        """
        Function called when user click in graph. Centroid have to be added.
        """
        if self.k_means is not None:
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
        Function sends data with clusters column and data with centroids position to the output
        """
        if self.k_means is None or self.k_means.clusters is None:
            self.send("Annotated Data", None)
            self.send("Centroids", None)
        else:
            clust_var = DiscreteVariable(
                self.output_name, values=["C%d" % (x + 1) for x in range(self.k_means.k)])
            attributes, classes = self.data.domain.attributes, self.data.domain.class_vars
            meta_attrs = self.data.domain.metas
            if classes:
                meta_attrs += classes
            classes = [clust_var]
            domain = Domain(attributes, classes, meta_attrs)
            annotated_data = Table.from_table(domain, self.data)
            annotated_data.get_column_view(clust_var)[0][:] = self.k_means.clusters

            centroids = Table(Domain(self.k_means.data.domain.attributes), self.k_means.centroids)
            self.send("Annotated Data", annotated_data)
            self.send("Centroids", centroids)
