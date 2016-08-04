from os import path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
from PyQt4.QtCore import pyqtSlot, Qt, QThread, SIGNAL
from PyQt4.QtGui import QSizePolicy, QPixmap, QColor, QIcon

from Orange.widgets.utils import itemmodels
from Orange.classification import Model
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable, \
    StringVariable
from Orange.widgets import gui
from Orange.widgets import highcharts
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget, Msg
from Orange.preprocess.preprocess import Normalize

from orangecontrib.educational.widgets.utils.logistic_regression \
    import LogisticRegression
from orangecontrib.educational.widgets.utils.contour import Contour


class Scatterplot(highcharts.Highchart):
    """
    Scatterplot extends Highchart and just defines some sane defaults:
    * enables scroll-wheel zooming,
    * set callback functions for click (in empty chart), drag and drop
    * enables moving of centroids points
    * include drag_drop_js script by highcharts
    """

    js_click_function = """/**/(function(e) {
            window.pybridge.chart_clicked(e.xAxis[0].value, e.yAxis[0].value);
        })
        """

    # to make unit tesest
    count_replots = 0

    def __init__(self, click_callback, **kwargs):

        # read javascript for drag and drop
        with open(
                path.join(path.dirname(__file__), 'resources',
                          'highcharts-contour.js'), 'r') as f:
            contours_js = f.read()

        super().__init__(enable_zoom=True,
                         bridge=self,
                         enable_select='',
                         chart_events_click=self.js_click_function,
                         plotOptions_series_states_hover_enabled=False,
                         javascript=contours_js,
                         **kwargs)

        self.click_callback = click_callback

    def chart(self, *args, **kwargs):
        self.count_replots += 1
        super(Scatterplot, self).chart(*args, **kwargs)

    @pyqtSlot(float, float)
    def chart_clicked(self, x, y):
        """
        Function is called from javascript when click event happens
        """
        self.click_callback(x, y)

    def remove_series(self, idx):
        """
        Function remove series with id idx
        """
        self.evalJS("""
            series = chart.get('{id}');
            if (series != null)
                series.remove(true);
            """.format(id=idx))

    def remove_last_point(self, idx):
        """
        Function remove last point from series with id idx
        """
        self.evalJS("""
            series = chart.get('{id}');
            if (series != null)
                series.removePoint(series.data.length - 1, true);
            """.format(id=idx))

    def add_series(self, series):
        """
        Function add series to the chart
        """
        for i, s in enumerate(series):
            self.exposeObject('series%d' % i, series[i])
            self.evalJS("chart.addSeries(series%d, true);" % i)

    def add_point_to_series(self, idx, x, y):
        """
        Function add point to the series with id idx
        """
        self.evalJS("""
            series = chart.get('{id}');
            series.addPoint([{x}, {y}]);
        """.format(id=idx, x=x, y=y))


class Autoplay(QThread):
    """
    Class used for separated thread when using "Autoplay" for gradient descent

    Parameters
    ----------
    ow_gradient_descent : OWGradientDescent
        Instance of OWGradientDescent class
    """

    def __init__(self, ow_gradient_descent):
        QThread.__init__(self)
        self.ow_gradient_descent = ow_gradient_descent

    def __del__(self):
        self.wait()

    def run(self):
        """
        Stepping through the algorithm until converge or user interrupts
        """
        while (not self.ow_gradient_descent.learner.converged and
               self.ow_gradient_descent.auto_play_enabled):
            self.emit(SIGNAL('step()'))
            time.sleep(2 - self.ow_gradient_descent.auto_play_speed)
        self.emit(SIGNAL('stop_auto_play()'))


class OWGradientDescent(OWWidget):
    """
    Gradient descent widget algorithm
    """

    name = "Gradient Descent"
    description = "Widget shows the procedure of gradient descent."
    icon = "icons/GradientDescent.svg"
    want_main_area = True

    inputs = [("Data", Table, "set_data")]
    outputs = [("Classifier", Model),
               ("Coefficients", Table),
               ("Data", Table)]

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    target_class = settings.Setting('')
    alpha = settings.Setting(0.1)
    auto_play_speed = settings.Setting(1)
    stochastic = settings.Setting(False)

    # models
    x_var_model = None
    y_var_model = None

    # function used in gradient descent
    default_learner = LogisticRegression
    learner = None
    cost_grid = None
    grid_size = 15
    contour_color = "#aaaaaa"
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    # data
    data = None
    selected_data = None

    # autoplay
    auto_play_enabled = False
    auto_play_button_text = ["Run", "Stop"]
    auto_play_thread = None

    class Warning(OWWidget.Warning):
        """
        Class used fro widget warnings.
        """
        to_few_features = Msg("Too few Continuous feature. Min 2 required")
        no_class = Msg("No class provided or only one class variable")

    def __init__(self):
        super().__init__()

        # var models
        self.x_var_model = itemmodels.VariableListModel()
        self.y_var_model = itemmodels.VariableListModel()

        # options box
        policy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.options_box = gui.widgetBox(self.controlArea, "Data")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal,
            callback=self.restart, sendSelectedValue=True
        )
        self.cbx = gui.comboBox(value='attr_x', label='X:', **opts)
        self.cbx.setSizePolicy(policy)
        self.cby = gui.comboBox(value='attr_y', label='Y:', **opts)
        self.cby.setSizePolicy(policy)
        self.target_class_combobox = gui.comboBox(
            value='target_class', label='Target class: ', **opts)
        self.target_class_combobox.setSizePolicy(policy)

        self.cbx.setModel(self.x_var_model)
        self.cby.setModel(self.y_var_model)

        # properties box
        self.properties_box = gui.widgetBox(self.controlArea, "Properties")
        self.alpha_spin = gui.spin(
            widget=self.properties_box, master=self, callback=self.change_alpha,
            value="alpha", label="Learning rate: ",
            minv=0.01, maxv=1, step=0.01, spinType=float)
        self.stochastic_checkbox = gui.checkBox(
            widget=self.properties_box, master=self,
            callback=self.change_stochastic, value="stochastic",
            label="Stochastic: ")
        self.restart_button = gui.button(
            widget=self.properties_box, master=self,
            callback=self.restart, label="Restart")

        # step box
        self.step_box = gui.widgetBox(self.controlArea, "Manually step through")
        self.step_button = gui.button(
            widget=self.step_box, master=self, callback=self.step, label="Step")
        self.step_back_button = gui.button(
            widget=self.step_box, master=self, callback=self.step_back,
            label="Step back")

        # run box
        self.run_box = gui.widgetBox(self.controlArea, "Run")
        self.auto_play_button = gui.button(
            widget=self.run_box, master=self,
            label=self.auto_play_button_text[0], callback=self.auto_play)
        self.auto_play_speed_spinner = gui.hSlider(
            widget=self.run_box, master=self, value='auto_play_speed',
            minValue=0, maxValue=1.91, step=0.1,
            intOnly=False, createLabel=False, label='Speed:')

        # graph in mainArea
        self.scatter = Scatterplot(click_callback=self.change_theta,
                                   xAxis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='',
                                   tooltip_shared=False,
                                   debug=True)
        # TODO: set false when end of development
        gui.rubber(self.controlArea)

        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

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

        self.Warning.clear()

        # clear variables
        self.cost_grid = None
        self.learner = None

        d = data
        self.send_output()

        if data is None or len(data) == 0:
            self.data = None
            reset_combos()
            self.set_empty_plot()
        elif sum(True for var in d.domain.attributes
                 if isinstance(var, ContinuousVariable)) < 2:
            self.data = None
            reset_combos()
            self.Warning.to_few_features()
            self.set_empty_plot()
        elif d.domain.class_var is None or len(d.domain.class_var.values) < 2:
            self.data = None
            reset_combos()
            self.Warning.no_class()
            self.set_empty_plot()
        else:
            self.data = data
            init_combos()
            self.attr_x = self.cbx.itemText(0)
            self.attr_y = self.cbx.itemText(1)
            self.target_class = self.target_class_combobox.itemText(0)
            self.restart()

    def set_empty_plot(self):
        """
        Function render empty plot
        """
        self.scatter.clear()

    def restart(self):
        """
        Function restarts the algorithm
        """
        self.selected_data = self.select_data()
        self.learner = self.default_learner(
            data=self.selected_data,
            alpha=self.alpha, stochastic=self.stochastic)
        self.replot()
        self.send_output()

    def change_alpha(self):
        """
        Function changes alpha parameter of the alogrithm
        """
        if self.learner is not None:
            self.learner.set_alpha(self.alpha)

    def change_stochastic(self):
        """
        Function changes switches between stochastic or usual algorithm
        """
        if self.learner is not None:
            self.learner.stochastic = self.stochastic

    def change_theta(self, x, y):
        """
        Function set new theta
        """
        if self.learner is not None:
            self.learner.set_theta([x, y])
            self.scatter.remove_series("path")
            self.scatter.add_series([
                dict(id="path", data=[[x, y]], showInLegend=False,
                     type="scatter", lineWidth=1,
                     marker=dict(enabled=True, radius=2))],)
            self.send_output()

    def step(self):
        """
        Function performs one step of the algorithm
        """
        if self.data is None:
            return
        if self.learner.theta is None:
            self.change_theta(np.random.uniform(self.min_x, self.max_x),
                              np.random.uniform(self.min_y, self.max_y))
        self.learner.step()
        theta = self.learner.theta
        self.plot_point(theta[0], theta[1])
        self.send_output()

    def step_back(self):
        """
        Function performs step back
        """
        if self.learner.step_no > 0:
            self.learner.step_back()
            self.scatter.remove_last_point("path")
            self.send_output()

    def plot_point(self, x, y):
        """
        Function add point to the path
        """
        self.scatter.add_point_to_series("path", x, y)

    def replot(self):
        """
        This function performs complete replot of the graph
        """
        if self.data is None:
            return

        optimal_theta = self.learner.optimized()
        self.min_x = optimal_theta[0] - 5
        self.max_x = optimal_theta[0] + 5
        self.min_y = optimal_theta[1] - 5
        self.max_y = optimal_theta[1] + 5

        options = dict(series=[])

        # gradient and contour
        options['series'] += self.plot_gradient_and_contour(
            self.min_x, self.max_x, self.min_y, self.max_y)

        min_value = np.min(self.cost_grid)
        max_value = np.max(self.cost_grid)

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text="theta 0",
            yAxis_title_text="theta 1",
            xAxis_min=self.min_x,
            xAxis_max=self.max_x,
            yAxis_min=self.min_y,
            yAxis_max=self.max_y,
            xAxis_startOnTick=False,
            xAxis_endOnTick=False,
            yAxis_startOnTick=False,
            yAxis_endOnTick=False,
            # colorAxis=dict(
            #     stops=[
            #         [min_value, "#ffffff"],
            #         [max_value, "#ff0000"]],
            #     tickInterval=1, max=max_value, min=min_value),
            plotOptions_contour_colsize=(self.max_y - self.min_y) / 10000,
            plotOptions_contour_rowsize=(self.max_x - self.min_x) / 10000,
            tooltip_enabled=False,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y))

        self.scatter.chart(options, **kwargs)

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
        xv, yv = np.meshgrid(x, y)
        thetas = np.column_stack((xv.flatten(), yv.flatten()))

        # cost_values = np.vstack([self.learner.j(theta) for theta in thetas])
        cost_values = self.learner.j(thetas)

        # results
        self.cost_grid = cost_values.reshape(xv.shape)

        blurred = self.blur_grid(self.cost_grid)

        # return self.plot_gradient(self.xv, self.yv, blurred) + \
        return self.plot_contour(xv, yv, blurred)

    def plot_gradient(self, x, y, grid):
        """
        Function constructs background gradient
        """
        return [dict(data=[[x[j, k], y[j, k], grid[j, k]] for j in range(len(x))
                           for k in range(y.shape[1])],
                     grid_width=self.grid_size,
                     type="contour")]

    def select_data(self):
        """
        Function takes two selected columns from data table and merge them
        in new Orange.data.Table

        Returns
        -------
        Table
            Table with selected columns
        """
        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]
        cols = []
        for attr in (attr_x, attr_y):
            subset = self.data[:, attr]
            cols.append(subset.X)
        x = np.column_stack(cols)
        domain = Domain(
            [attr_x, attr_y],
            [DiscreteVariable(name=self.data.domain.class_var.name,
                              values=[self.target_class, 'Others'])],
            [self.data.domain.class_var])
        y = [(0 if d.get_class().value == self.target_class else 1)
             for d in self.data]

        return Normalize(Table(domain, x, y, self.data.Y[:, None]))

    def plot_contour(self, xv, yv, cost_grid):
        """
        Function constructs contour lines
        """

        contour = Contour(
            xv, yv, cost_grid)
        contour_lines = contour.contours(
            np.linspace(np.min(cost_grid), np.max(cost_grid), 10))

        series = []
        count = 0
        for key, value in contour_lines.items():
            for line in value:
                interpol_line = line

                series.append(dict(data=interpol_line,
                                   color=self.contour_color,
                                   type="spline",
                                   lineWidth=0.5,
                                   showInLegend=False,
                                   marker=dict(enabled=False),
                                   name="%g" % round(key, 2),
                                   enableMouseTracking=False
                                   ))
                count += 1
        return series

    @staticmethod
    def blur_grid(grid):
        """
        Function blur the grid, to make crossings smoother
        """
        filtered = gaussian_filter(grid, sigma=1)
        return filtered

    def auto_play(self):
        """
        Function called when autoplay button pressed
        """
        self.auto_play_enabled = not self.auto_play_enabled
        self.auto_play_button.setText(
            self.auto_play_button_text[self.auto_play_enabled])
        if self.auto_play_enabled:
            self.disable_controls(self.auto_play_enabled)
            self.auto_play_thread = Autoplay(self)
            self.connect(self.auto_play_thread, SIGNAL("step()"), self.step)
            self.connect(
                self.auto_play_thread, SIGNAL("stop_auto_play()"),
                self.stop_auto_play)
            self.auto_play_thread.start()
        else:
            self.stop_auto_play()

    def stop_auto_play(self):
        """
        Called when stop autoplay button pressed or in the end of autoplay
        """
        self.auto_play_enabled = False
        self.disable_controls(self.auto_play_enabled)
        self.auto_play_button.setText(
            self.auto_play_button_text[self.auto_play_enabled])

    def disable_controls(self, disabled):
        """
        Function disable or enable all controls except those from run part
        """
        self.step_box.setDisabled(disabled)
        self.options_box.setDisabled(disabled)
        self.properties_box.setDisabled(disabled)

    def send_output(self):
        self.send_model()
        self.send_coefficients()
        self.send_data()

    def send_model(self):
        if self.learner is not None and self.learner.theta is not None:
            self.send("Classifier", self.learner.model)
        else:
            self.send("Classifier", None)

    def send_coefficients(self):
        if self.learner is not None and self.learner.theta is not None:
            domain = Domain(
                    [ContinuousVariable("coef", number_of_decimals=7)],
                    metas=[StringVariable("name")])
            names = ["theta 0", "theta 1"]

            coefficients_table = Table(
                    domain, list(zip(list(self.learner.theta), names)))
            self.send("Coefficients", coefficients_table)
        else:
            self.send("Coefficients", None)

    def send_data(self):
        if self.selected_data is not None:
            self.send("Data", self.selected_data)
        else:
            self.send("Data", None)