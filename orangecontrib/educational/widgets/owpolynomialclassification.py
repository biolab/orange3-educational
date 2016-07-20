from Orange.data import ContinuousVariable, Table, Domain, StringVariable, DiscreteVariable
from Orange.widgets import highcharts, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.classification import LogisticRegressionLearner, Learner
import numpy as np
from orangecontrib.educational.widgets.utils.polynomialexpansion import PolynomialTransform
from PyQt4.QtGui import QSizePolicy
from os import path
from orangecontrib.educational.widgets.utils.color_transform import rgb_hash_brighter
from orangecontrib.educational.widgets.utils.contour import Contour
from scipy.ndimage.filters import gaussian_filter
import copy


class Scatterplot(highcharts.Highchart):
    """
    Scatterplot extends Highchart and just defines some defaults:
    * disable scroll-wheel zooming,
    * disable all points selection
    * set cursor for series to move
    * adds javascript for contour
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

    def remove_contours(self):
        self.evalJS("""
            for(i=chart.series.length - 1; i >= 0; i--){
                if(chart.series[i].type == "spline")
                {
                    chart.series[i].remove(false);
                }
            }""")

    def add_series(self, series):
        for i, s in enumerate(series):
            self.exposeObject('series%d' % i, series[i])
            self.evalJS("chart.addSeries(series%d, false);" % i)

    def redraw_series(self):
        self.evalJS("chart.redraw();")


class OWPolyinomialClassification(OWBaseLearner):
    name = "Polynomial Classification"
    description = "Widget that demonstrates classification in two classes with polynomial expansion of attributes."
    icon = "icons/polyclassification.svg"
    want_main_area = True
    resizing_enabled = True
    send_report = True

    # inputs and outputs
    inputs = [("Data", Table, "set_data"),
              ("Learner", Learner, "set_learner")]
    outputs = [("Coefficients", Table)]

    # data attributes
    data = None
    selected_data = None
    probabilities_grid = None
    xv = None
    yv = None

    # learners
    LEARNER = LogisticRegressionLearner
    learner_other = None
    default_preprocessor = PolynomialTransform

    learner_name = settings.Setting("Polynomial Classification")

    # widget properties
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    target_class = settings.Setting('')
    degree = settings.Setting(1)
    contours_enabled = settings.Setting(True)
    contour_step = settings.Setting(0.1)

    graph_name = 'scatter'

    # settings
    grid_size = 30
    colors = ["#1F7ECA", "#D32525", "#28D825", "#D5861F", "#98257E",
              "#2227D5", "#D5D623", "#D31BD6", "#6A7CDB", "#78D5D4"]  # taken from highcharts.options.colors
    contour_color = "#1f1f1f"

    # layout elements
    options_box = None
    cbx = None
    cby = None
    degree_spin = None
    plot_properties_box = None
    contours_enabled_checkbox = None
    contour_step_slider = None
    scatter = None

    def add_main_layout(self):

        # options box
        self.options_box = gui.widgetBox(self.controlArea, "Options")
        self.cbx = gui.comboBox(self.options_box, self, 'attr_x',
                                label='X:',
                                orientation='horizontal',
                                callback=self.apply,
                                sendSelectedValue=True)
        self.cby = gui.comboBox(self.options_box, self, 'attr_y',
                                label='Y:',
                                orientation='horizontal',
                                callback=self.apply,
                                sendSelectedValue=True)
        self.target_class_combobox = gui.comboBox(self.options_box, self, 'target_class',
                                                  label='Target class:',
                                                  orientation='horizontal',
                                                  callback=self.apply,
                                                  sendSelectedValue=True)
        self.degree_spin = gui.spin(self.options_box, self, 'degree',
                                    minv=1,
                                    maxv=5,
                                    step=1,
                                    label='Polynomial expansion:',
                                    callback=self.init_learner)
        self.cbx.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.cby.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))
        self.target_class_combobox.setSizePolicy(QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed))

        # plot properties box
        self.plot_properties_box = gui.widgetBox(self.controlArea, "Plot Properties")
        self.contours_enabled_checkbox = gui.checkBox(self.plot_properties_box, self, 'contours_enabled',
                                                      label="Show contours",
                                                      callback=self.plot_contour)
        self.contour_step_slider = gui.spin(self.plot_properties_box,
                                            self,
                                            'contour_step',
                                            minv=0.10,
                                            maxv=0.50,
                                            step=0.05,

                                            label='Contour step:',
                                            decimals=2,
                                            spinType=float,
                                            callback=self.plot_contour)

        gui.rubber(self.controlArea)

        # chart
        self.scatter = Scatterplot(Axis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   yAxis_startOnTick=False,
                                   yAxis_endOnTick=False,
                                   xAxis_startOnTick=False,
                                   xAxis_endOnTick=False,
                                   xAxis_lineWidth=0,
                                   yAxis_lineWidth=0,
                                   yAxis_tickWidth=1,
                                   title_text='',
                                   tooltip_shared=False,
                                   colorAxis=dict(
                                       stops=[
                                           [0, rgb_hash_brighter(self.colors[0], 50)],
                                           [0.5, '#ffffff'],
                                           [1, rgb_hash_brighter(self.colors[1], 50)]],
                                       tickInterval=0.2,
                                       min=0,
                                       max=1
                                   ),
                                   legend=dict(enabled=False),
                                   debug=True)  # TODO: set false when end of development

        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

        self.init_learner()

    def set_learner(self, learner):
        """
        Function is sets learner when learner is changed on input
        """
        self.learner_other = learner
        self.init_learner()

    def set_preprocessor(self, preprocessor):
        """
        Function adds preprocessor when it changed on input
        """
        self.preprocessors = [preprocessor] if preprocessor else []
        self.init_learner()

    def set_data(self, data):
        """
        Function receives data from input and init part of widget if data are ok. Otherwise set empty plot and notice
        user about that

        Parameters
        ----------
        data : Table
            Input data
        """
        self.data = data

        def reset_combos():
            self.cbx.clear()
            self.cby.clear()
            self.target_class_combobox.clear()

        def init_combos():
            """
            function initialize the combos with attributes
            """
            reset_combos()
            for var in data.domain if data is not None else []:
                if var.is_primitive() and var.is_continuous:
                    self.cbx.addItem(gui.attributeIconDict[var], var.name)
                    self.cby.addItem(gui.attributeIconDict[var], var.name)

            for var in data.domain.class_var.values:
                self.target_class_combobox.addItem(var)

        self.warning(1)  # remove warning about too less continuous attributes if exists

        if data is None or len(data) == 0:
            reset_combos()
            self.set_empty_plot()
        elif sum(True for var in data.domain.attributes if isinstance(var, ContinuousVariable)) < 2:
            reset_combos()
            self.warning(1, "Too few Continuous feature. Min 2 required")
            self.set_empty_plot()
        elif data.domain.class_var is None or len(data.domain.class_var.values) < 2:
            reset_combos()
            self.warning(1, "No class provided or only one class variable")
            self.set_empty_plot()
        else:
            init_combos()
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            self.target_class = self.target_class_combobox.itemText(0)
            self.apply()

    def init_learner(self):
        """
        Function init learner and add preprocessors to learner
        """
        self.learner = copy.deepcopy(self.learner_other) or self.LEARNER(penalty='l2', C=1e10)
        self.learner.preprocessors = ((self.preprocessors or []) +
                                      (self.learner.preprocessors or []) +
                                      [self.default_preprocessor(self.degree)])
        self.apply()

    def set_empty_plot(self):
        """
        Function inits empty plot
        """
        self.scatter.clear()

    def replot(self):
        """
        This function performs complete replot of the graph
        """
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        data_x = [v[0] for v in self.data[:, attr_x]]
        data_y = [v[0] for v in self.data[:, attr_y]]
        min_x = min(data_x)
        max_x = max(data_x)
        min_y = min(data_y)
        max_y = max(data_y)
        diff_x = (max_x - min_x) if abs(max_x - min_x) > 0.001 else 0.1  # just in cas that diff is 0
        diff_y = (max_y - min_y) if abs(max_y - min_y) > 0.001 else 0.1
        min_x, max_x = min_x - 0.03 * diff_x, max_x + 0.03 * diff_x
        min_y, max_y = min_y - 0.03 * diff_y, max_y + 0.03 * diff_y

        options = dict(series=[])

        # gradient and contour
        options['series'] += self.plot_gradient_and_contour(min_x, max_x, min_y, max_y)

        # data points
        target_class_index = self.data.domain.class_var.values.index(self.target_class)
        classes = ([target_class_index] +
                   [i for i in range(len(self.data.domain.class_var.values)) if i != target_class_index])

        options['series'] += [dict(data=[list(p.attributes())
                                         for p in self.selected_data if int(p.metas[0]) == _class],
                                   type="scatter",
                                   zIndex=10,
                                   color=self.colors[i],
                                   showInLegend=False) for i, _class in enumerate(classes)]

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            xAxis_min=min_x,
            xAxis_max=max_x,
            yAxis_min=min_y,
            yAxis_max=max_y,

            plotOptions_contour_colsize=(max_y - min_y) / 1000,
            plotOptions_contour_rowsize=(max_x - min_x) / 1000,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y))

        self.scatter.chart(options, **kwargs)
        self.plot_contour()

    def plot_gradient_and_contour(self, x_from, x_to, y_from, y_to):
        """
        Function constructs series for gradient and contour

        Parameters
        ----------
        x_from : float
            Min grid x value
        x_to : float
            Max grid x value
        y_from : float
            Min grid y value
        y_to : float
            Max grid y value

        Returns
        -------
        list
            List containing series with background gradient and contour
        """

        # grid for gradient
        x = np.linspace(x_from, x_to, self.grid_size)
        y = np.linspace(y_from, y_to, self.grid_size)
        self.xv, self.yv = np.meshgrid(x, y)

        # parameters to predict from grid
        attr = np.hstack((self.xv.reshape((-1, 1)), self.yv.reshape((-1, 1))))
        attr_data = Table(self.selected_data.domain, attr,
                          np.array([[None]] * len(attr)),
                          np.array([[None]] * len(attr)))

        # results
        self.probabilities_grid = self.model(attr_data, 1)[:, 1].reshape(self.xv.shape)

        blurred = self.blur_grid(self.probabilities_grid)

        return self.plot_gradient(self.xv, self.yv, blurred)

    def plot_gradient(self, x, y, grid):
        """
        Function constructs background gradient
        """
        return [dict(data=[[x[j, k], y[j, k], grid[j, k]] for j in range(len(x))
                           for k in range(y.shape[1])],
                     grid_width=self.grid_size,
                     type="contour")]

    def plot_contour(self):
        """
        Function constructs contour lines
        """
        self.scatter.remove_contours()
        if self.contours_enabled:
            contour = Contour(self.xv, self.yv, self.probabilities_grid)
            contour_lines = contour.contours(
                np.hstack(
                    (np.arange(0.5, 0, - self.contour_step)[::-1],  # we want to have contour for 0.5
                     np.arange(0.5 + self.contour_step, 1, self.contour_step))))

            series = []
            count = 0
            for key, value in contour_lines.items():
                for line in value:

                    series.append(dict(data=self.labeled(line, count),
                                       color=self.contour_color,
                                       type="spline",
                                       lineWidth=0.5,
                                       showInLegend=False,
                                       marker=dict(enabled=False),
                                       name="%g" % round(key, 2),
                                       enableMouseTracking=False
                                       ))
                    count += 1
            self.scatter.add_series(series)
        self.scatter.redraw_series()

    @staticmethod
    def blur_grid(grid):
        filtered = gaussian_filter(grid, sigma=1)
        filtered[(grid > 0.45) & (grid < 0.55)] = grid[(grid > 0.45) & (grid < 0.55)]
        return filtered

    @staticmethod
    def labeled(data, no):
        """
        Function labels data with contour levels
        """
        point = (no * 5) % len(data)  # we will add this label on the first point
        point = point + (1 if point == 0 else 0)
        data[point] = dict(
            x=data[point][0],
            y=data[point][1],
            dataLabels=dict(
                enabled=True,
                format="{series.name}",
                verticalAlign='middle',
                style=dict(
                    fontWeight="normal",
                    color=OWPolyinomialClassification.contour_color,
                    textShadow=False
                )))
        return data

    def select_data(self):
        """
        Function takes two selected columns from data table and merge them in new Orange.data.Table

        Returns
        -------
        Table
            Table with selected columns
        """
        attr_x, attr_y = self.data.domain[self.attr_x], self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.X)
        x = np.column_stack(cols)
        domain = Domain([attr_x, attr_y],
                        [DiscreteVariable(name=self.data.domain.class_var.name, values=[self.target_class, 'Others'])],
                        [self.data.domain.class_var])
        y = [(0 if d.get_class().value == self.target_class else 1) for d in self.data]

        return Table(domain, x, y, self.data.Y[:, None])

    def apply(self):
        """
        Applies leaner and sends new model and coefficients
        """
        self.send_learner()
        self.update_model()
        self.send_coefficients()
        if self.data is not None:
            self.replot()

    def send_learner(self):
        """
        Function sends learner on widget's output
        """

        self.learner.name = self.learner_name
        self.send("Learner", self.learner)

    def update_model(self):
        """
        Function sends model on widget's output
        """
        if self.data is not None:
            self.selected_data = self.select_data()
            self.model = self.learner(self.selected_data)
            self.model.name = self.learner_name
            self.model.instances = self.selected_data

        self.send(self.OUTPUT_MODEL_NAME, self.model)

    def send_coefficients(self):
        """
        Function sends coefficients on widget's output if model has them
        """

        if self.model is not None and isinstance(self.learner, LogisticRegressionLearner):
            model = self.model.skl_model
            if model is not None and hasattr(model, "coef_"):
                domain = Domain([ContinuousVariable("coef", number_of_decimals=7)],
                                metas=[StringVariable("name")])
                coefficients = model.intercept_.tolist() + model.coef_[0].tolist()

                data = self.model.instances
                for preprocessor in self.learner.preprocessors:
                    data = preprocessor(data)
                names = [1] + [x.name for x in data.domain.attributes]

                coefficients_table = Table(domain, list(zip(coefficients, names)))
                self.send("Coefficients", coefficients_table)
            else:
                self.send("Coefficients", None)
        else:
            self.send("Coefficients", None)

    def add_bottom_buttons(self):
        pass
