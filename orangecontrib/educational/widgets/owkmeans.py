import Orange
from Orange.widgets.widget import OWWidget
from Orange.data import DiscreteVariable, ContinuousVariable, Table, Domain
from Orange.widgets import gui, settings, highcharts
import numpy as np
from itertools import chain
from .utils.kmeans import Kmeans
from PyQt4.QtCore import pyqtSlot, QObject
from os import path


def rgb_hash_brighter(hash, percent_brighter):
    r, g, b = hex_to_rgb(hash)
    brightness_to_add = 255 * percent_brighter / 100
    r, g, b = r + brightness_to_add, g + brightness_to_add, b + brightness_to_add
    return rgb_to_hex(tuple(min(v, 255) for v in (r, g, b)))

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb


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

    def __init__(self, selection_callback, click_callback, drag_callback, **kwargs):

        # read javascript for drag and drop
        with open(path.join(path.dirname(__file__), 'resources', 'draggable-points.js'), 'r') as f:
            drag_drop_js = f.read()

        super().__init__(enable_zoom=True,
                         bridge=self,
                         enable_select='',
                         chart_type='scatter',
                         selection_callback=selection_callback,
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
        print('a')

    @pyqtSlot(int, float, float)
    def point_dropped(self, index, x, y):
        self.drag_callback(index, x, y)
        print('b')



class OWKmeans(OWWidget):

    name = "Educational k-Means"
    description = "Widget demonstrates working of k-mans algorithm."
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
        gui.spin(self.optionsBox, self, 'numberOfClusters',
                 minv=0, maxv=10, step=1, label='Number of centroids:', # how much to limit
                 callback=self.cluster_change)

        # step and restart buttons
        gui.button(self.optionsBox, self, 'Step', callback=self.step)
        gui.button(self.optionsBox, self, 'Restart', callback=self.restart)

        # disable until data loaded
        self.optionsBox.setDisabled(True)
        gui.rubber(self.controlArea)

        # plot
        self.scatter = Scatterplot(selection_callback=self.on_selection,
                                   click_callback=self.graph_clicked,
                                   drag_callback=self.centroid_dropped,
                                   xAxis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='',
                                   tooltip_shared=False,
                                   # In development, we can enable debug mode
                                   # and get right-click-inspect and related
                                   # console utils available:
                                   debug=True)
        # Just render an empty chart so it shows a nice 'No data to display'
        # warning
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)


    def merge_x_y(self):
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.Y if subset.Y.size else subset.X)
        X = np.column_stack(cols)
        domain = Domain([attr_x, attr_y])
        return Table(domain, X)

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

        def setEmptyPlot():
            self.scatter.clear()
            self.optionsBox.setDisabled(True)

        # if data contains at least two continuous attributes
        if sum(True for var in data.domain.attributes if isinstance(var, ContinuousVariable)) <= 2:
            self.info.setText("Too few Continuous feature. Min 2 required")
            setEmptyPlot()
        elif data is None:
            self.info.setText("No data on input yet, waiting to get something.")
            setEmptyPlot()
        else:
            self.info.setText("")
            self.optionsBox.setDisabled(False)
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            self.restart()

    def restart(self):
        self.k_means = Kmeans(self.merge_x_y())
        self.stepNo = 0
        self.cluster_change()
        self.replot()

    def replot(self):

        if self.data is None or not self.attr_x or not self.attr_y:
            return

        data = self.data
        attr_x, attr_y = data.domain[self.attr_x], data.domain[self.attr_y]

        options = dict(series=[])

        colors = ['#2f7ed8', '#0d233a', '#8bbc21', '#910000', '#1aadce',
            '#492970', '#f28f43', '#77a1e5', '#c42525', '#a6c96a'] # define some more colors

        # if self.stepNo == 0:
        #     X = self.merge_x_y().X
        #     options['series'].append(dict(data=X, showInLegend=False))
        # else:
        for i, points in enumerate(self.k_means.centroids_belonging_points):
            options['series'].append(dict(data=points,
                                          showInLegend=False,
                                          color=rgb_hash_brighter(colors[i % len(colors)], 30)))


        options['series'].append(dict(data=[{'x': p[0],
                                             'y':p[1],
                                             'marker':{'fillColor': colors[i % len(colors)]}}
            for i, p in enumerate(self.k_means.centroids_before
                                  if self.stepNo % 2 == 1
                                  else self.k_means.centroids)],
                                      draggableX=True if self.stepNo == 0 else False,
                                      draggableY=True if self.stepNo == 0 else False,
                                      showInLegend=False,
                                      marker=dict(symbol='diamond',
                                                  radius=7)))


        # Besides the options dict, Highcharts can also be passed keyword
        # parameters, where each parameter is split on underscores in
        # simulated object hierarchy. This works:
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

        # That's it, we can scatter our scatter by calling its chart method
        # with the parameters we'd constructed
        self.scatter.chart(options, **kwargs)

    def on_selection(self, indices):
        pass

    def cluster_change(self):
        if self.k_means.k < self.numberOfClusters:
            for _ in range(self.numberOfClusters - self.k_means.k):
                self.k_means.add_centroids()
        else:
            for _ in range(self.k_means.k - self.numberOfClusters):
                self.k_means.delete_centroids()
        self.replot()

    def step(self):
        self.stepNo += 1
        if self.stepNo % 2 == 1:
            self.k_means.step()
        self.replot()

    def graph_clicked(self, x, y):
        self.k_means.add_centroids([x, y])
        self.numberOfClusters += 1
        self.replot()

    def centroid_dropped(self, _index, x, y):
        self.k_means.move_centroid(_index, x, y)
        self.replot()

