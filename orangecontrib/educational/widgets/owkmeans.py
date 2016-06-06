import Orange
from Orange.widgets.widget import OWWidget
from Orange.data import DiscreteVariable, ContinuousVariable, Table, Domain
from Orange.widgets import gui, settings, highcharts
import numpy as np
from .utils.kmeans import Kmeans
from PyQt4.QtCore import pyqtSlot
from os import path
from .utils.color_transform import rgb_hash_brighter
from itertools import chain


class Scatterplot(highcharts.Highchart):
    """
    Scatterplot extends Highchart and just defines some sane defaults:
    * enables scroll-wheel zooming,
    * enables rectangle (+ individual point) selection,
    * sets the chart type to 'scatter' (could also be 'bubble' or as
      appropriate; Se Highcharts JS docs)
    * sets the selection callback. The callback is passed a list (array)
      of indices of selected points for each data series the chart knows
      about.
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

    js_drag_function = """/**/(function(event) {
                var index = this.series.data.indexOf( this );
                console.log(event.x);
                console.log(event.y);
                // window.pybridge.point_dropped(index, event.x, event.y);
            })
            """

    def __init__(self, click_callback, drag_callback, **kwargs):

        # read javascript for drag and drop
        with open(path.join(path.dirname(__file__), 'resources', 'draggable-points.js'), 'r') as f:
            drag_drop_js = f.read()

        super().__init__(enable_zoom=True,
                         bridge=self,
                         enable_select='',
                         chart_events_click=self.js_click_function,
                         plotOptions_series_point_events_drag=self.js_drag_function,
                         plotOptions_series_point_events_drop=self.js_drop_function,
                         plotOptions_series_cursor="move",
                         javascript=drag_drop_js,
                         **kwargs)

        self.click_callback = click_callback
        self.drag_callback = drag_callback

    @pyqtSlot(float, float)
    def chart_clicked(self, x, y):
        self.click_callback(x, y)

    @pyqtSlot(int, float, float)
    def point_dragged(self, index, x, y):
        print(index, x, y)
        self.drag_callback(index, x, y)

    @pyqtSlot(int, float, float)
    def point_dropped(self, index, x, y):
        self.drag_callback(index, x, y)


class OWKmeans(OWWidget):

    name = "Educational k-Means"
    description = "Widget demonstrates working of k-means algorithm."
    icon = "icons/mywidget.svg"
    want_main_area = False

    # inputs and outputs
    inputs = [("Data", Orange.data.Table, "set_data")]

    # settings
    numberOfClusters = settings.Setting(0)
    stepNo = 0

    # data
    data = None

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    lines_to_centroids = settings.Setting(0)

    graph_name = 'scatter'

    def __init__(self):
        super().__init__()

        # info box
        box = gui.widgetBox(self.controlArea, "Info")
        self.info = gui.widgetLabel(box, 'No data on input yet, waiting to get something.')

        # options box
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        self.cbx = gui.comboBox(self.optionsBox, self, 'attr_x',
                                label='X:',
                                orientation='horizontal',
                                callback=self.restart,
                                sendSelectedValue=True)
        self.cby = gui.comboBox(self.optionsBox, self, 'attr_y',
                                label='Y:',
                                orientation='horizontal',
                                callback=self.restart,
                                sendSelectedValue=True)
        self.centroidNumbersSpinner = gui.spin(self.optionsBox, self, 'numberOfClusters',
                 minv=0, maxv=10, step=1, label='Number of centroids:',
                 callback=self.number_of_clusters_changed)
        gui.checkBox(self.optionsBox, self, 'lines_to_centroids',
                     'Membership lines', callback=self.replot)
        # step and restart buttons
        self.stepButton = gui.button(self.optionsBox, self, 'Step', callback=self.step)
        gui.button(self.optionsBox, self, 'Restart', callback=self.restart)

        # disable until data loaded
        self.optionsBox.setDisabled(True)
        gui.rubber(self.controlArea)

        # plot
        self.scatter = Scatterplot(click_callback=self.graph_clicked,
                                   drag_callback=self.centroid_dropped,
                                   xAxis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='',
                                   tooltip_shared=False,
                                   debug=True)  # TODO: set false when end of development
        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

        self.k_means = None

    def concat_x_y(self):
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.Y if subset.Y.size else subset.X)
        x = np.column_stack(cols)
        domain = Domain([attr_x, attr_y])
        return Table(domain, x)

    def set_data(self, data):
        self.data = data

        def init_combos():
            self.cbx.clear()
            self.cby.clear()
            for var in data.domain if data is not None else []:
                if var.is_primitive() and var.is_continuous:
                    self.cbx.addItem(gui.attributeIconDict[var], var.name)
                    self.cby.addItem(gui.attributeIconDict[var], var.name)

        init_combos()

        def set_empty_plot():
            self.scatter.clear()
            self.optionsBox.setDisabled(True)

        # if data contains at least two continuous attributes
        if data is None:
            self.info.setText("No data on input yet, waiting to get something.")
            set_empty_plot()
        elif sum(True for var in data.domain.attributes if isinstance(var, ContinuousVariable)) < 2:
            self.info.setText("Too few Continuous feature. Min 2 required")
            set_empty_plot()
        else:
            self.info.setText("")
            self.optionsBox.setDisabled(False)
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            self.restart()

    def restart(self):
        # if self.k_means is None:
        self.k_means = Kmeans(self.concat_x_y())
        # else:
        #     self.k_means.data = self.concat_x_y()
        self.stepNo = 0
        self.number_of_clusters_changed()
        self.replot()
        self.centroidNumbersSpinner.setDisabled(False)
        self.stepButton.setText("Move centroids")

    def step(self):
        self.stepNo += 1
        self.k_means.step()
        self.replot()
        self.centroidNumbersSpinner.setDisabled(False if self.k_means.stepNo % 2 == 0 else True)
        self.stepButton.setText("Move centroids" if self.k_means.stepNo % 2 == 0 else "Find new clusters")


    def replot(self):
        colors = ['#2f7ed8', '#0d233a', '#8bbc21', '#910000', '#1aadce',
                  '#492970', '#f28f43', '#77a1e5', '#c42525', '#a6c96a']

        if self.data is None or not self.attr_x or not self.attr_y:
            return

        data = self.data
        attr_x, attr_y = data.domain[self.attr_x], data.domain[self.attr_y]

        options = dict(series=[])

        if self.lines_to_centroids:
            for i, c in enumerate(self.k_means.centroids):
                options['series'].append(dict(data=list(chain.from_iterable(([p[0], p[1]], [c[0], c[1]])
                                                    for p in self.k_means.centroids_belonging_points[i])),
                                              type="line",
                                              showInLegend=False,
                                              lineWidth=0.2,
                                              color="#ccc"))

        # plot data points
        for i, points in enumerate(self.k_means.centroids_belonging_points):
            options['series'].append(dict(data=points,
                                          type="scatter",
                                          showInLegend=False,
                                          color=rgb_hash_brighter(colors[i % len(colors)], 30)))

        # plot centroids
        options['series'].append(dict(data=[{'x': p[0],
                                             'y':p[1],
                                             'marker':{'fillColor': colors[i % len(colors)]}}
                                            for i, p in enumerate(self.k_means.centroids)],
                                      type="scatter",
                                      draggableX=True if self.k_means.stepNo % 2 == 0 else False,
                                      draggableY=True if self.k_means.stepNo % 2 == 0 else False,
                                      showInLegend=False,
                                      marker=dict(symbol='diamond',
                                                  radius=10)))

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            tooltip_headerFormat=(
                '<span style="color:{point.color}">\u25CF</span> '
                '{series.name} <br/>'),
            tooltip_pointFormat=(
                '<b>{attr_x.name}:</b> {{point.x}}<br/>'
                '<b>{attr_y.name}:</b> {{point.y}}<br/>').format_map(locals()))
        # If any of selected attributes is discrete, we correctly scatter it
        # as a categorical
        if attr_x.is_discrete:
            kwargs['xAxis_categories'] = attr_x.values
        if attr_y.is_discrete:
            kwargs['yAxis_categories'] = attr_y.values

        # plot
        self.scatter.chart(options, **kwargs)

    def number_of_clusters_changed(self):
        if self.k_means.k < self.numberOfClusters:
            for _ in range(self.numberOfClusters - self.k_means.k):
                self.k_means.add_centroids()
        else:
            for _ in range(self.k_means.k - self.numberOfClusters):
                self.k_means.delete_centroids()
        self.replot()

    def graph_clicked(self, x, y):
        if self.k_means.stepNo % 2 == 0:
            self.k_means.add_centroids([x, y])
            self.numberOfClusters += 1
            self.replot()

    def centroid_dropped(self, _index, x, y):
        self.k_means.move_centroid(_index, x, y)
        self.replot()
