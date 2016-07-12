from Orange.data import Table, ContinuousVariable, Table, Domain
from Orange.widgets import highcharts, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.classification import LogisticRegressionLearner, Learner
import numpy as np
from orangecontrib.educational.widgets.utils.polynomialexpansion import PolynomialTransform
from PyQt4.QtGui import QSizePolicy
from os import path
from orangecontrib.educational.widgets.utils.color_transform import rgb_hash_brighter
from orangecontrib.educational.widgets.utils.contour import Contour
from scipy.interpolate import splprep, splev

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

    paint_function = """
        paint_function = function() {
            console.log("a");
            $('#belowPath').remove()
            $('#abovePath').remove()

            var series = chart.series[0];
            var path = [];

            series.data.forEach(function(element) {
                path.push(element.plotX + chart.plotLeft);
                path.push(element.plotY + chart.plotTop);
            });

            var path_above = ['M', chart.plotLeft, chart.plotTop, 'L']
                .concat(path)
                .concat([chart.plotLeft + chart.plotWidth, chart.plotTop]);

            var path_below = ['M', chart.plotLeft, chart.plotTop + chart.plotHeight, 'L']
                .concat(path)
                .concat([chart.plotLeft + chart.plotWidth, chart.plotTop + chart.plotHeight]);

            chart.renderer.path(path_above)
                .attr({
                    stroke: "none",
                    fill: chart.series[1].color,
                    'fill-opacity': 0.2,
                    zIndex: 0.5,
                    id: "abovePath"
                }).add();

            chart.renderer.path(path_below)
                .attr({
                    stroke: "none",
                    fill: chart.series[2].color,
                    'fill-opacity': 0.2,
                    zIndex: 0.5,
                    id: "belowPath"
                }).add();
        }
        """

    def __init__(self, **kwargs):
        with open(path.join(path.dirname(__file__), 'resources', 'highcharts-contour.js'), 'r') as f:
            contour_js = f.read()

        super().__init__(enable_zoom=False,
                         bridge=self,
                         enable_select='',
                         plotOptions_series_cursor="move",
                         javascript=contour_js,
                         **kwargs)


class OWPolyinomialClassification(OWBaseLearner):
    name = "Polynomial Classification"
    description = "a"  #TODO: description
    icon = "icons/mywidget.svg"
    want_main_area = True
    resizing_enabled = True

    # inputs and outputs
    inputs = [("Data", Table, "set_data"),
              ("Learner", Learner, "set_learner")]

    data = None
    selected_data = None
    probabilities_grid = None

    LEARNER = LogisticRegressionLearner
    learner1 = None
    default_preprocessor = PolynomialTransform

    learner_name = settings.Setting("Polynomial Classification")

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    degree = settings.Setting(1)
    contours_enabled = settings.Setting(True)
    contour_step = settings.Setting(0.1)

    graph_name = 'scatter'

    # settings
    grid_size = 25
    colors = ['#2f7ed8', '#D32525']


    def add_main_layout(self):

        self.preprocessors = []

        # options box
        self.options_box = gui.widgetBox(self.controlArea, "Options")
        self.cbx = gui.comboBox(self.options_box, self, 'attr_x',
                                label='X:',
                                orientation='horizontal',
                                callback=self.refresh,
                                sendSelectedValue=True)
        self.cby = gui.comboBox(self.options_box, self, 'attr_y',
                                label='Y:',
                                orientation='horizontal',
                                callback=self.refresh,
                                sendSelectedValue=True)
        self.degree_spin = gui.spin(self.options_box, self, 'degree',
                 minv=1, maxv=5, step=1, label='Polynomial expansion:',  # how much to limit
                 callback=self.degree_changed)
        self.cbx.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.cby.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        self.plot_properties_box = gui.widgetBox(self.controlArea, "Plot Properties")
        self.contours_enabled_checkbox = gui.checkBox(self.plot_properties_box, self, 'contours_enabled',
                                            label="Show contours",
                                            callback=self.replot)
        self.contour_step_slider = gui.hSlider(self.plot_properties_box,
                                               self,
                                               'contour_step',
                                               minValue=0.05,
                                               maxValue=0.51,
                                               step=0.05,
                                               intOnly=False,
                                               createLabel=True,
                                               labelFormat="%.2f",
                                               label='Contour step:',
                                               callback=self.replot)

        gui.rubber(self.controlArea)

        # plot
        self.scatter = Scatterplot(Axis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='',
                                   tooltip_shared=False,
                                   debug=True)  # TODO: set false when end of development
        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

    def set_learner(self, learner):
        self.learner1 = learner if learner is not None else self.LEARNER(preprocessors=self.preprocessors)
        self.change_features()

    def set_data(self, data):
        """
        Function receives data from input and init part of widget if data are ok. Otherwise set empty plot and notice
        user about that
        :param data: input data
        :type data: Orange.data.Table or None
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
        elif sum(True for var in data.domain.attributes if isinstance(var, ContinuousVariable)) < 2:
            reset_combos()
            self.warning(1, "Too few Continuous feature. Min 2 required")
            self.set_empty_plot()
        elif data.domain.class_var is None:
            reset_combos()
            self.warning(1, "No class provided")
            self.set_empty_plot()
        elif len(data.domain.class_var.values) > 2:
            reset_combos()
            self.warning(1, "Too much classes. Max 2 required")
            self.set_empty_plot()
        else:
            init_combos()
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            self.change_features()

    def degree_changed(self):
        if self.learner1 is None:
            self.learner1 = self.LEARNER()
        self.learner1.preprocessors = self.preprocessors + [self.default_preprocessor(self.degree)]
        self.replot()

    def set_empty_plot(self):
        self.scatter.clear()

    def refresh(self):
        if self.data is not None:
            self.change_features()

    def change_features(self):

        self.selected_data = self.concat_x_y()
        self.replot()

    def replot(self):
        """
        This function performs complete replot of the graph without animation
        """
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        data_x = [v[0] for v in self.data[:, attr_x]]
        data_y = [v[0] for v in self.data[:, attr_y]]
        min_x = min(data_x)
        max_x = max(data_x)
        min_y = min(data_y)
        max_y = max(data_y)
        diff_x = max_x - min_x
        diff_y = max_y - min_y
        min_x, max_x = min_x - 0.03 * diff_x, max_x + 0.03 * diff_x
        min_y, max_y = min_y - 0.03 * diff_y, max_y + 0.03 * diff_y

       # plot centroids
        options = dict(series=[])

        line_series = self.plot_line(min_x, max_x, min_y, max_y)
        options['series'] += line_series

        classes = [0, 1]

        options['series'] += [dict(data=[list(p.attributes())
                                            for p in self.selected_data if int(p.get_class()) == _class],
                                   type="scatter",
                                   zIndex=10,
                                   color=self.colors[_class],
                                   showInLegend=False) for _class in [0, 1]]


        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            xAxis_min=min_x,
            xAxis_max=max_x,
            yAxis_min=min_y,
            yAxis_max=max_y,
            yAxis_startOnTick=False,
            yAxis_endOnTick= False,
            xAxis_startOnTick=False,
            xAxis_endOnTick= False,
            xAxis_lineWidth=0,
            yAxis_lineWidth=0,
            yAxis_tickWidth=1,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y),
            colorAxis=dict(
                stops=[
                [0, rgb_hash_brighter(self.colors[0], 30)],
                [0.5, '#ffffff'],
                [1, rgb_hash_brighter(self.colors[1], 30)]],
                tickInterval=0.2,
                min=0,
                max=1
            ))

    
        # plot
        self.scatter.chart(options, **kwargs)
        # self.scatter.evalJS("chart.redraw()")

    def plot_line(self, x_from, x_to, y_from, y_to):
        if self.learner1 is None:
            self.learner1 = self.LEARNER(preprocessors=self.preprocessors + [self.default_preprocessor(self.degree)])
        model = self.learner1(self.selected_data)

        x = np.linspace(x_from, x_to, self.grid_size)
        y = np.linspace(y_from, y_to, self.grid_size)

        xv, yv = np.meshgrid(x, y)
        attr = np.hstack((xv.reshape((-1, 1)), yv.reshape((-1, 1))))

        attr_data = Table(self.selected_data.domain, attr, np.array([[None]] * len(attr)))

        self.probabilities_grid = model(attr_data, 1)[:, 1].reshape(xv.shape)
            # take probabilities for second class (column 1), to have class 0 prob 0 and class 1 prob 1

        series = []
        if self.contours_enabled:

            contour = Contour(xv, yv, self.probabilities_grid)
            contour_lines = contour.contours(np.hstack(
                (np.arange(0.5, 0, - self.contour_step)[::-1],
                np.arange(0.5 + self.contour_step, 1, self.contour_step))))

            for key, value in contour_lines.items():
                for line in value:
                    if len(line) > 3:  # if less than 3 line can not be interpolated
                        tck, u = splprep([list(x) for x in zip(*reversed(line))], s=1)
                        new_int = np.arange(0, 1.01, 0.01)
                        interpolated_line = np.array(splev(new_int, tck)).T.tolist()
                    else:
                        interpolated_line = line

                    series.append(dict(data=self.labeled(interpolated_line),
                                       color="#aaaaaa",
                                       type="spline",
                                       lineWidth=0.5,
                                       showInLegend=False,
                                       marker=dict(enabled=False),
                                       name="%g" % round(key, 2),
                                       enableMouseTracking=False
                                       ))

        return [dict(data=[[xv[j, k], yv[j, k], self.probabilities_grid[j, k]] for j in range(len(xv))
                          for k in range(yv.shape[1])],
                        grid_width=self.grid_size,
                        type="contour")] + series

    @staticmethod
    def labeled(data):
        point = 1  # we will add this label on the first point
        data[point] = dict(
            x=data[point][0],
            y=data[point][1],
            dataLabels=dict(
                enabled=True,
                format="{series.name}",
                style=dict(
                    fontWeight="normal",
                    color="#aaaaaa",
                    textShadow=False
                )))
        return data

    def concat_x_y(self):
        """
        Function takes two selected columns from data table and merge them in new Orange.data.Table
        :return: table with selected columns
        :type: Orange.data.Table
        """
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.Y if subset.Y.size else subset.X)
        x = np.column_stack(cols)
        domain = Domain([attr_x, attr_y], self.data.domain.class_var)
        return Table(domain, x, self.data.Y)
