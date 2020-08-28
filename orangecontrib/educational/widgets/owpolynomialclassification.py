from Orange.widgets.utils import itemmodels
from math import isnan
from os import path
import copy

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import splprep, splev
from scipy.ndimage.filters import gaussian_filter

from AnyQt.QtCore import Qt
from AnyQt.QtGui import QPixmap, QColor, QIcon

from Orange.base import Learner as InputLearner
from Orange.data import (
    ContinuousVariable, Table, Domain, StringVariable, DiscreteVariable)
from Orange.widgets import settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    TreeLearner
)
from Orange.widgets.widget import Msg, Input, Output
from orangewidget.report import report


from orangecontrib.educational.widgets.utils.polynomialtransform \
    import PolynomialTransform
from orangecontrib.educational.widgets.utils.color_transform \
    import rgb_hash_brighter, rgb_to_hex
from orangecontrib.educational.widgets.utils.contour import Contour
from orangecontrib.educational.widgets.highcharts import Highchart


class Scatterplot(Highchart):
    """
    Scatterplot extends Highchart and just defines some defaults:
    * disable scroll-wheel zooming,
    * disable all points selection
    * set cursor for series to move
    * adds javascript for contour
    """

    def __init__(self, **kwargs):
        with open(path.join(path.dirname(__file__), 'resources', 'highcharts-contour.js'),
                  encoding='utf-8') as f:
            contour_js = f.read()

        super().__init__(enable_zoom=False,
                         enable_select='',
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


class OWPolynomialClassification(OWBaseLearner):
    name = "Polynomial Classification"
    description = "Widget that demonstrates classification in two classes " \
                  "with polynomial expansion of attributes."
    keywords = ["polynomial classification", "classification", "class",
                "classification visualization", "polynomial features"]
    icon = "icons/polynomialclassification.svg"
    want_main_area = True
    resizing_enabled = True
    priority = 600

    # inputs and outputs
    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", InputLearner)

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, default=True)
        data = Output("Data", Table)

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
    legend_enabled = settings.Setting(True)
    contours_enabled = settings.Setting(False)
    contour_step = settings.Setting(0.1)

    graph_name = 'scatter'

    # settings
    grid_size = 25
    contour_color = "#1f1f1f"

    # layout elements
    options_box = None
    cbx = None
    cby = None
    degree_spin = None
    plot_properties_box = None
    contours_enabled_checkbox = None
    legend_enabled_checkbox = None
    contour_step_slider = None
    scatter = None
    target_class_combobox = None
    x_var_model = None
    y_var_model = None

    class Error(OWBaseLearner.Error):
        to_few_features = Msg(
            "Polynomial classification requires at least two numeric features")
        no_class = Msg("Data must have a single discrete class attribute")
        all_none_data = Msg("One of the features has no defined values")
        no_classifier = Msg("Learner must be a classifier")

    def add_main_layout(self):
        # var models
        self.x_var_model = itemmodels.VariableListModel()
        self.y_var_model = itemmodels.VariableListModel()

        # options box
        self.options_box = gui.widgetBox(self.controlArea, "Options")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal)
        opts_combo = dict(opts, **dict(sendSelectedValue=True))
        self.cbx = gui.comboBox(
            value='attr_x', label='X: ', callback=self.apply, **opts_combo)
        self.cby = gui.comboBox(
            value='attr_y', label='Y: ', callback=self.apply, **opts_combo)
        self.target_class_combobox = gui.comboBox(
            value='target_class', label='Target: ',
            callback=self.apply, **opts_combo)
        self.degree_spin = gui.spin(
            value='degree', label='Polynomial expansion:',
            minv=1, maxv=5, step=1, callback=self.init_learner,
            alignment=Qt.AlignRight, controlWidth=70, **opts)

        self.cbx.setModel(self.x_var_model)
        self.cby.setModel(self.y_var_model)

        # plot properties box
        self.plot_properties_box = gui.widgetBox(
            self.controlArea, "Plot Properties")
        self.legend_enabled_checkbox = gui.checkBox(
            self.plot_properties_box, self, 'legend_enabled',
            label="Show legend", callback=self.replot)
        self.contours_enabled_checkbox = gui.checkBox(
            self.plot_properties_box, self, 'contours_enabled',
            label="Show contours", callback=self.plot_contour)
        self.contour_step_slider = gui.spin(
            self.plot_properties_box, self, 'contour_step',
            minv=0.10, maxv=0.50, step=0.05, callback=self.plot_contour,
            label='Contour step:', decimals=2, spinType=float,
            alignment=Qt.AlignRight, controlWidth=70)

        gui.rubber(self.controlArea)

        # chart
        self.scatter = Scatterplot(
            xAxis_gridLineWidth=0, yAxis_gridLineWidth=0,
            xAxis_startOnTick=False, xAxis_endOnTick=False,
            yAxis_startOnTick=False, yAxis_endOnTick=False,
            xAxis_lineWidth=0, yAxis_lineWidth=0,
            yAxis_tickWidth=1, title_text='', tooltip_shared=False)

        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

        self.init_learner()

    @Inputs.learner
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
        Function receives data from input and init part of widget if data
        satisfy. Otherwise set empty plot and notice
        user about that

        Parameters
        ----------
        data : Table
            Input data
        """

        def reset_combos():
            self.x_var_model[:] = []
            self.y_var_model[:] = []
            self.target_class_combobox.clear()

        def init_combos():
            """
            function initialize the combos with attributes
            """
            reset_combos()

            c_vars = [var for var in data.domain.variables if var.is_continuous]

            self.x_var_model[:] = c_vars
            self.y_var_model[:] = c_vars

            for i, var in enumerate(data.domain.class_var.values):
                pix_map = QPixmap(60, 60)
                color = tuple(data.domain.class_var.colors[i].tolist())
                pix_map.fill(QColor(*color))
                self.target_class_combobox.addItem(QIcon(pix_map), var)

        self.Error.clear()

        # clear variables
        self.xv = None
        self.yv = None
        self.probabilities_grid = None

        if data is None or len(data) == 0:
            self.data = None
            reset_combos()
            self.set_empty_plot()
        elif sum(True for var in data.domain.attributes
                 if isinstance(var, ContinuousVariable)) < 2:
            self.data = None
            reset_combos()
            self.Error.to_few_features()
            self.set_empty_plot()
        elif (data.domain.class_var is None or
              data.domain.class_var.is_continuous or
              sum(line.get_class() == None for line in data) == len(data) or
              len(data.domain.class_var.values) < 2):
            self.data = None
            reset_combos()
            self.Error.no_class()
            self.set_empty_plot()
        else:
            self.data = data
            init_combos()
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            self.target_class = self.target_class_combobox.itemText(0)

        self.apply()

    def init_learner(self):
        """
        Function init learner and add preprocessors to learner
        """
        if self.learner_other is not None and \
            self.learner_other.__class__.__name__ == "LinearRegressionLearner":
            # in case that learner is a Linear Regression
            self.learner = None
            self.Error.no_classifier()
        else:
            self.learner = (copy.deepcopy(self.learner_other) or
                            self.LEARNER(penalty='l2', C=1e10))
            self.learner.preprocessors = (
                [self.default_preprocessor(self.degree)] +
                list(self.preprocessors or []) +
                list(self.learner.preprocessors or []))
            self.Error.no_classifier.clear()
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
        if self.data is None or self.selected_data is None:
            self.set_empty_plot()
            return

        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]
        data_x = [v[0] for v in self.data[:, attr_x] if not isnan(v[0])]
        data_y = [v[0] for v in self.data[:, attr_y] if not isnan(v[0])]
        min_x = min(data_x)
        max_x = max(data_x)
        min_y = min(data_y)
        max_y = max(data_y)
        # just in cas that diff is 0
        diff_x = (max_x - min_x) if abs(max_x - min_x) > 0.001 else 0.1
        diff_y = (max_y - min_y) if abs(max_y - min_y) > 0.001 else 0.1
        min_x, max_x = min_x - 0.03 * diff_x, max_x + 0.03 * diff_x
        min_y, max_y = min_y - 0.03 * diff_y, max_y + 0.03 * diff_y

        options = dict(series=[])

        # gradient and contour
        options['series'] += self.plot_gradient_and_contour(
            min_x, max_x, min_y, max_y)

        sd = self.selected_data
        # data points
        options['series'] += [
            dict(
                data=[list(p.attributes())
                      for p in sd
                      if (p.metas[0] == _class and
                          all(v is not None for v in p.attributes()))],
                type="scatter",
                zIndex=10,
                color=rgb_to_hex(tuple(
                    sd.domain.metas[0].colors[_class].tolist())),
                showInLegend=True,
                name=sd.domain.metas[0].values[_class])
            for _class in range(len(sd.domain.metas[0].values))]

        # add nan values as a gray dots
        options['series'] += [
            dict(
                data=[list(p.attributes())
                      for p in sd
                      if np.isnan(p.metas[0])],
                type="scatter",
                zIndex=10,
                color=rgb_to_hex((160, 160, 160)),
                showInLegend=False)]

        cls_domain = sd.domain.metas[0]

        target_idx = cls_domain.values.index(self.target_class)
        target_color = tuple(cls_domain.colors[target_idx].tolist())
        other_color = (tuple(cls_domain.colors[(target_idx + 1) % 2].tolist())
                       if len(cls_domain.values) == 2 else (170, 170, 170))

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            xAxis_min=min_x,
            xAxis_max=max_x,
            yAxis_min=min_y,
            yAxis_max=max_y,
            colorAxis=dict(
                labels=dict(enabled=False),
                stops=[
                    [0, rgb_hash_brighter(rgb_to_hex(other_color), 0.5)],
                    [0.5, '#ffffff'],
                    [1, rgb_hash_brighter(rgb_to_hex(target_color), 0.5)]],
                tickInterval=0.2, min=0, max=1),
            plotOptions_contour_colsize=(max_y - min_y) / 1000,
            plotOptions_contour_rowsize=(max_x - min_x) / 1000,
            legend=dict(
                enabled=self.legend_enabled,
                layout='vertical',
                align='right',
                verticalAlign='top',
                floating=True,
                backgroundColor='rgba(255, 255, 255, 0.3)',
                symbolWidth=0,
                symbolHeight=0),
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
        attr_data = Table.from_numpy(
            self.selected_data.domain, attr,
            np.array([[None]] * len(attr)),
            np.array([[None]] * len(attr))
        )

        # results
        self.probabilities_grid = self.model(attr_data, 1)[:, 0]\
            .reshape(self.xv.shape)

        blurred = self.blur_grid(self.probabilities_grid)

        is_tree = type(self.learner) in [RandomForestLearner, TreeLearner]
        return self.plot_gradient(self.xv, self.yv,
                                  self.probabilities_grid
                                  if is_tree else blurred)

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
        if not self.data:
            return
        if self.contours_enabled:
            is_tree = type(self.learner) in [RandomForestLearner, TreeLearner]
            # tree does not need smoothing
            contour = Contour(
                self.xv, self.yv, self.probabilities_grid
                if is_tree else self.blur_grid(self.probabilities_grid))
            contour_lines = contour.contours(
                np.hstack(
                    (np.arange(0.5, 0, - self.contour_step)[::-1],
                     np.arange(0.5 + self.contour_step, 1, self.contour_step))))
            # we want to have contour for 0.5

            series = []
            count = 0
            for key, value in contour_lines.items():
                for line in value:
                    if (len(line) > self.degree and
                                type(self.learner) not in
                                [RandomForestLearner, TreeLearner]):
                        # if less than degree interpolation fails
                        tck, u = splprep(
                            [list(x) for x in zip(*reversed(line))],
                            s=0.001, k=self.degree,
                            per=(len(line)
                                 if np.allclose(line[0], line[-1])
                                 else 0))
                        new_int = np.arange(0, 1.01, 0.01)
                        interpol_line = np.array(splev(new_int, tck)).T.tolist()
                    else:
                        interpol_line = line

                    series.append(dict(data=self.labeled(interpol_line, count),
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
        filtered[(grid > 0.45) & (grid < 0.55)] = grid[(grid > 0.45) &
                                                       (grid < 0.55)]
        return filtered

    @staticmethod
    def labeled(data, no):
        """
        Function labels data with contour levels
        """
        point = (no * 5) # to avoid points on same positions
        point += (1 if point == 0 else 0)
        point %= len(data)

        data[point] = dict(
            x=data[point][0],
            y=data[point][1],
            dataLabels=dict(
                enabled=True,
                format="{series.name}",
                verticalAlign='middle',
                style=dict(
                    fontWeight="normal",
                    color=OWPolynomialClassification.contour_color,
                    textShadow=False
                )))
        return data

    def select_data(self):
        """
        Function takes two selected columns from data table and merge them
        in new Orange.data.Table

        Returns
        -------
        Table
            Table with selected columns
        """
        self.Error.clear()

        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.X if not sp.issparse(subset.X) else subset.X.toarray())
        x = np.column_stack(cols)
        y_c = self.data.Y[:, None] if not sp.issparse(self.data.Y) else self.data.Y.toarray()

        if np.isnan(x).all(axis=0).any():
            self.Error.all_none_data()
            return None

        cls_domain = self.data.domain.class_var
        target_idx = cls_domain.values.index(self.target_class)
        other_value = cls_domain.values[(target_idx + 1) % 2]

        class_domain = [DiscreteVariable(
            name="Transformed " + self.data.domain.class_var.name,
            values=(self.target_class, 'Others'
            if len(cls_domain.values) > 2 else other_value))]

        domain = Domain(
            [attr_x, attr_y],
            class_domain,
            [self.data.domain.class_var])
        y = [(0 if d.get_class().value == self.target_class else 1)
             for d in self.data]

        return Table.from_numpy(domain, x, y, y_c)

    def apply(self):
        """
        Applies leaner and sends new model and coefficients
        """
        self.send_learner()
        self.update_model()
        self.send_coefficients()
        if any(a is None for a in (self.data, self.model)):
            self.set_empty_plot()
        else:
            self.replot()
        self.send_data()

    def send_learner(self):
        """
        Function sends learner on widget's output
        """
        if self.learner is not None:
            self.learner.name = self.learner_name
        self.Outputs.learner.send(self.learner)

    def update_model(self):
        """
        Function sends model on widget's output
        """
        self.Error.fitting_failed.clear()
        self.model = None
        if self.data is not None and self.learner is not None:
            self.selected_data = self.select_data()
            if self.selected_data is not None:
                try:
                    self.model = self.learner(self.selected_data)
                    self.model.name = self.learner_name
                    self.model.instances = self.selected_data
                except Exception as e:
                    self.Error.fitting_failed(str(e))

        self.Outputs.model.send(self.model)

    def send_coefficients(self):
        """
        Function sends coefficients on widget's output if model has them
        """

        if (self.model is not None and
                isinstance(self.learner, LogisticRegressionLearner) and
                hasattr(self.model, 'skl_model')):
            model = self.model.skl_model
            domain = Domain(
                [ContinuousVariable("coef")], metas=[StringVariable("name")])
            coefficients = (model.intercept_.tolist() +
                            model.coef_[0].tolist())

            data = self.model.instances
            for preprocessor in self.learner.preprocessors:
                data = preprocessor(data)
            names = ["Intercept"] + [x.name for x in data.domain.attributes]

            coefficients_table = Table.from_list(
                domain, list(zip(coefficients, names)))
            self.Outputs.coefficients.send(coefficients_table)
        else:
            self.Outputs.coefficients.send(None)

    def send_data(self):
        """
        Function sends data on widget's output
        """
        if self.data is not None:
            data = self.selected_data
            self.Outputs.data.send(data)
            return
        self.Outputs.data.send(None)

    def add_bottom_buttons(self):
        pass

    def send_report(self):
        if self.data is None:
            return
        caption = report.render_items_vert((
             ("Polynomial Expansion", self.degree),
        ))
        self.report_plot(self.scatter)
        if caption:
            self.report_caption(caption)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPolynomialClassification).run(Table.from_file('iris'))
