from os import path
import time

import numpy as np
from Orange.widgets.utils import itemmodels
from PyQt4.QtCore import pyqtSlot, Qt, QThread, SIGNAL
from PyQt4.QtGui import QSizePolicy, QPixmap, QColor, QIcon
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter

from Orange.classification import Model
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
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
    * include drag_drop_js script by highchart
    """

    js_click_function = """/**/(function(event) {
            window.pybridge.chart_clicked(event.xAxis[0].value, event.yAxis[0].value);
        })
        """

    # to make unit tesest
    count_replots = 0

    def __init__(self, click_callback, **kwargs):

        # read javascript for drag and drop
        with open(path.join(path.dirname(__file__), 'resources', 'highcharts-contour.js'), 'r') as f:
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
        self.click_callback(x, y)

    def remove_series(self, id):
        self.evalJS("""
            series = chart.get('{id}');
            if (series != null)
                series.remove(true);
            """.format(id=id))

    def remove_last_point(self, id):
        self.evalJS("""
            series = chart.get('{id}');
            if (series != null)
                series.removePoint(series.data.length - 1, true);
            """.format(id=id))

    def add_series(self, series):
        for i, s in enumerate(series):
            self.exposeObject('series%d' % i, series[i])
            self.evalJS("chart.addSeries(series%d, true);" % i)

    def add_point_to_series(self, id, x, y):
        self.evalJS("""
            series = chart.get('{id}');
            series.addPoint([{x}, {y}]);
        """.format(id=id, x=x, y=y))


class Autoplay(QThread):
    """
    Class used for separated thread when using "Autoplay" for k-means
    Parameters
    ----------
    owkmeans : OWKmeans
        Instance of OWKmeans class
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

    name = "Gradient Descent"
    description = "Widget demonstrates shows the procedure of gradient descent."
    icon = "icons/InteractiveKMeans.svg"
    want_main_area = True

    inputs = [("Data", Table, "set_data")]
    outputs = [("Model", Model),
               ("Coefficients", Table)]

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    target_class = settings.Setting('')
    alpha = settings.Setting(0.1)
    auto_play_speed = settings.Setting(1)

    # models
    x_var_model = None
    y_var_model = None

    # function used in gradient descent
    default_learner = LogisticRegression
    learner = None
    cost_grid = None
    grid_size = 15
    contour_color = "#aaaaaa"

    # data
    data = None
    selected_data = None

    # autoplay
    auto_play_enabled = False
    autoplay_button_text = ["Run", "Stop"]

    class Warning(OWWidget.Warning):
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

        self.properties_box = gui.widgetBox(self.controlArea, "Properties")
        self.alpha_spin = gui.spin(widget=self.properties_box,
                                   master=self,
                                   callback=self.change_alpha,
                                   value="alpha",
                                   label="Alpha: ",
                                   minv=0.01,
                                   maxv=1,
                                   step=0.01,
                                   spinType=float)
        self.restart_button = gui.button(widget=self.properties_box,
                                       master=self,
                                       callback=self.restart,
                                       label="Restart")

        self.step_box = gui.widgetBox(self.controlArea, "Manually step through")

        self.step_button = gui.button(widget=self.step_box,
                                       master=self,
                                       callback=self.step,
                                       label="Step")
        self.step_back_button = gui.button(widget=self.step_box,
                                       master=self,
                                       callback=self.step_back,
                                       label="Step back")

        self.run_box = gui.widgetBox(self.controlArea, "Run")
        self.auto_play_button = gui.button(
            self.run_box, self, self.autoplay_button_text[0],
            callback=self.auto_play)
        self.auto_play_speed_spinner = gui.hSlider(self.run_box,
                                                   self,
                                                   'auto_play_speed',
                                                   minValue=0,
                                                   maxValue=1.91,
                                                   step=0.1,
                                                   intOnly=False,
                                                   createLabel=False,
                                                   label='Speed:')

        # graph in mainArea
        self.scatter = Scatterplot(click_callback=self.set_theta,
                                   xAxis_gridLineWidth=0,
                                   yAxis_gridLineWidth=0,
                                   title_text='',
                                   tooltip_shared=False,
                                   debug=True)

        gui.rubber(self.controlArea)

        # TODO: set false when end of development
        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

        # set random learner

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
        self.xv = None
        self.yv = None
        self.cost_grid = None

        if data is None or len(data) == 0:
            self.data = None
            reset_combos()
            self.set_empty_plot()
        elif sum(True for var in data.domain.attributes
                 if isinstance(var, ContinuousVariable)) < 2:
            self.data = None
            reset_combos()
            self.Warning.to_few_features()
            self.set_empty_plot()
        elif (data.domain.class_var is None or
                      len(data.domain.class_var.values) < 2):
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
        self.scatter.clear()

    def restart(self):
        self.selected_data = self.select_data()
        self.learner = self.default_learner(data=Normalize(self.selected_data),
                                            alpha=self.alpha)
        self.replot()

    def change_alpha(self):
        if self.learner is not None:
            self.learner.set_alpha(self.alpha)

    def step(self):
        if self.data is None:
            return
        if self.learner.theta is None:
            self.set_theta(np.random.uniform(self.min_x, self.max_x),
                           np.random.uniform(self.min_y, self.max_y))
        self.learner.step()
        theta = self.learner.theta
        self.plot_point(theta[0], theta[1])

    def step_back(self):
        if self.learner.step_no > 0:
            self.learner.step_back()
            self.scatter.remove_last_point("path")

    def plot_point(self, x, y):
        self.scatter.add_point_to_series("path", x, y)

    def replot(self):
        """
        This function performs complete replot of the graph
        """
        if self.data is None:
            return

        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]

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
        self.xv, self.yv = np.meshgrid(x, y)
        thetas = np.column_stack((self.xv.flatten(), self.yv.flatten()))

        cost_values = np.vstack([self.learner.j(theta) for theta in thetas])

        # results
        self.cost_grid = cost_values.reshape(self.xv.shape)

        blurred = self.blur_grid(self.cost_grid)

        # return self.plot_gradient(self.xv, self.yv, blurred) + \
        return self.plot_contour()

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

        return Table(domain, x, y, self.data.Y[:, None])

    def plot_contour(self):
        """
        Function constructs contour lines
        """

        contour = Contour(
            self.xv, self.yv, self.blur_grid(self.cost_grid))
        contour_lines = contour.contours(
            np.linspace(np.min(self.cost_grid), np.max(self.cost_grid), 10))

        series = []
        count = 0
        for key, value in contour_lines.items():
            for line in value:
                # if len(line) > 3:
                #     # if less than degree interpolation fails
                #     tck, u = splprep(
                #         [list(x) for x in zip(*reversed(line))],
                #         s=0.001, k=3,
                #         per=(len(line)
                #              if np.allclose(line[0], line[-1])
                #              else 0))
                #     new_int = np.arange(0, 1.01, 0.01)
                #     interpol_line = np.array(splev(new_int, tck)).T.tolist()
                # else:
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
        filtered = gaussian_filter(grid, sigma=1)
        return filtered

    def set_theta(self, x, y):
        if self.learner is not None:
            self.learner.set_theta([x, y])
            self.scatter.remove_series("path")
            self.scatter.add_series([
                dict(id="path", data=[[x, y]], showInLegend=False,
                     type="scatter", lineWidth=1,
                     marker=dict(enabled=True, radius=2))],)

    def auto_play(self):
        """
        Function called when autoplay button pressed
        """
        self.auto_play_enabled = not self.auto_play_enabled
        self.auto_play_button.setText(
            self.autoplay_button_text[self.auto_play_enabled])
        if self.auto_play_enabled:
            self.disable_controls(self.auto_play_enabled)
            self.autoPlayThread = Autoplay(self)
            self.connect(self.autoPlayThread, SIGNAL("step()"), self.step)
            self.connect(
                self.autoPlayThread, SIGNAL("stop_auto_play()"),
                self.stop_auto_play)
            self.autoPlayThread.start()
        else:
            self.stop_auto_play()

    def stop_auto_play(self):
        """
        Called when stop autoplay button pressed or in the end of autoplay
        """
        self.auto_play_enabled = False
        self.disable_controls(self.auto_play_enabled)
        self.auto_play_button.setText(
            self.autoplay_button_text[self.auto_play_enabled])

    def disable_controls(self, disabled):
        self.step_box.setDisabled(disabled)
        self.options_box.setDisabled(disabled)
        self.properties_box.setDisabled(disabled)


