import operator

from os import path
import time

import numpy as np
import scipy.sparse as sp

from AnyQt.QtCore import pyqtSlot, Qt, QThread, pyqtSignal, QObject
from AnyQt.QtGui import QPixmap, QColor, QIcon
from AnyQt.QtWidgets import QSizePolicy

from Orange.widgets.utils import itemmodels
from Orange.classification import Model
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable, \
    StringVariable
from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.preprocess.preprocess import Normalize
from scipy.interpolate import splprep, splev
from orangewidget.report import report

from orangecontrib.educational.widgets.utils.color_transform import (
    rgb_to_hex, hex_to_rgb)
from orangecontrib.educational.widgets.utils.linear_regression import \
    LinearRegression
from orangecontrib.educational.widgets.utils.logistic_regression \
    import LogisticRegression
from orangecontrib.educational.widgets.utils.contour import Contour
from orangecontrib.educational.widgets.highcharts import Highchart


class Scatterplot(Highchart):
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
        with open(path.join(path.dirname(__file__), 'resources', 'highcharts-contour.js'),
                  encoding='utf-8') as f:
            contours_js = f.read()

        class Bridge(QObject):
            @pyqtSlot(float, float)
            def chart_clicked(self, x, y):
                """
                Function is called from javascript when click event happens
                """
                click_callback(x, y)

        super().__init__(enable_zoom=True,
                         bridge=Bridge(),
                         enable_select='',
                         chart_events_click=self.js_click_function,
                         plotOptions_series_states_hover_enabled=False,
                         chart_panning=False,
                         javascript=contours_js,
                         **kwargs)

        self.click_callback = click_callback

    def chart(self, *args, **kwargs):
        self.count_replots += 1
        super(Scatterplot, self).chart(*args, **kwargs)

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

    def add_point_to_series(self, idx, point):
        """
        Function add point to the series with id idx
        """
        self.exposeObject('point', point)
        self.evalJS("""
            series = chart.get('{id}');
            series.addPoint(point);
        """.format(id=idx))


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
        while (self.ow_gradient_descent.learner and
               not self.ow_gradient_descent.learner.converged and
               self.ow_gradient_descent.auto_play_enabled and
               self.ow_gradient_descent.learner.step_no <= 500):
            try:
                self.ow_gradient_descent.step_trigger.emit()
            except RuntimeError:
                return
            time.sleep(2 - self.ow_gradient_descent.auto_play_speed)
        self.ow_gradient_descent.stop_auto_play_trigger.emit()


class OWGradientDescent(OWWidget):
    """
    Gradient descent widget algorithm
    """

    name = "Gradient Descent"
    description = "Widget shows the procedure of gradient descent " \
                  "on logistic regression."
    keywords = ["gradient descent", "optimization", "gradient"]
    icon = "icons/GradientDescent.svg"
    want_main_area = True
    priority = 400

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", Model)
        coefficients = Output("Coefficients", Table)
        data = Output("Data", Table)

    graph_name = "scatter"

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    target_class = settings.Setting('')
    alpha = settings.Setting(0.1)
    step_size = settings.Setting(30)  # step size for stochastic gds
    auto_play_speed = settings.Setting(1)
    stochastic = settings.Setting(False)

    # models
    x_var_model = None
    y_var_model = None

    # function used in gradient descent
    learner_name = ""
    learner = None
    cost_grid = None
    grid_size = 10
    contour_color = "#aaaaaa"
    default_background_color = "#00BFFF"
    line_colors = ["#00BFFF", "#ff0000", "#33cc33"]
    min_x = None
    max_x = None
    min_y = None
    max_y = None
    current_gradient_color = None

    # data
    data = None
    selected_data = None

    # autoplay
    auto_play_enabled = False
    auto_play_button_text = ["Run", "Stop"]
    auto_play_thread = None

    # signals
    step_trigger = pyqtSignal()
    stop_auto_play_trigger = pyqtSignal()

    class Error(OWWidget.Error):
        """
        Class used fro widget warnings.
        """
        to_few_features = Msg("Too few numeric features.")
        no_class = Msg("Data must have a single class attribute")
        to_few_values = Msg("Class attribute must have at least two values.")
        all_none = Msg("One of the features has no defined values")

    def __init__(self):
        super().__init__()

        # var models
        self.x_var_model = itemmodels.VariableListModel()
        self.y_var_model = itemmodels.VariableListModel()

        # info box
        self.info_box = gui.widgetBox(self.controlArea, "Info")
        self.learner_label = gui.label(
            widget=self.info_box, master=self, label="")

        # options box
        policy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.options_box = gui.widgetBox(self.controlArea, "Data")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal,
            callback=self.change_attributes, sendSelectedValue=True,
        )
        self.cbx = gui.comboBox(value='attr_x', label='X:', **opts)
        self.cby = gui.comboBox(value='attr_y', label='Y:', **opts)
        self.target_class_combobox = gui.comboBox(
            value='target_class', label='Target class: ', **opts)

        self.cbx.setModel(self.x_var_model)
        self.cby.setModel(self.y_var_model)

        gui.separator(self.controlArea, 20, 20)

        # properties box
        self.properties_box = gui.widgetBox(self.controlArea, "Properties")
        self.alpha_spin = gui.spin(
            widget=self.properties_box, master=self, callback=self.change_alpha,
            value="alpha", label="Learning rate: ",
            minv=0.001, maxv=100, step=0.001, spinType=float, decimals=3,
            alignment=Qt.AlignRight, controlWidth=80)
        self.stochastic_checkbox = gui.checkBox(
            widget=self.properties_box, master=self,
            callback=self.change_stochastic, value="stochastic",
            label="Stochastic ")
        self.step_size_spin = gui.spin(
            widget=self.properties_box, master=self, callback=self.change_step,
            value="step_size", label="Step size: ",
            minv=1, maxv=100, step=1, alignment=Qt.AlignRight, controlWidth=80)
        self.restart_button = gui.button(
            widget=self.properties_box, master=self,
            callback=self.restart, label="Restart")

        self.alpha_spin.setSizePolicy(policy)
        self.step_size_spin.setSizePolicy(policy)

        gui.separator(self.controlArea, 20, 20)

        # step box
        self.step_box = gui.widgetBox(self.controlArea, "Manually step through")
        self.step_button = gui.button(
            widget=self.step_box, master=self, callback=self.step, label="Step",
            default=True)
        self.step_back_button = gui.button(
            widget=self.step_box, master=self, callback=self.step_back,
            label="Step back")

        gui.separator(self.controlArea, 20, 20)

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
                                   legend=dict(enabled=False),)

        gui.rubber(self.controlArea)

        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

        self.step_size_lock()
        self.step_back_button_lock()

    @Inputs.data
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
        d = data

        def reset_combos():
            self.x_var_model[:] = []
            self.y_var_model[:] = []
            self.target_class_combobox.clear()

        def init_combos():
            """
            function initialize the combos with attributes
            """
            reset_combos()

            c_vars = [var for var in d.domain.attributes if var.is_continuous]

            self.x_var_model[:] = c_vars
            self.y_var_model[:] = c_vars if self.is_logistic else []

            for i, var in (enumerate(d.domain.class_var.values)
                           if d.domain.class_var.is_discrete else []):
                pix_map = QPixmap(60, 60)
                color = tuple(d.domain.class_var.colors[i].tolist())
                pix_map.fill(QColor(*color))
                self.target_class_combobox.addItem(QIcon(pix_map), var)

            self.cby.setDisabled(not self.is_logistic)
            self.target_class_combobox.setDisabled(not self.is_logistic)

        self.Error.clear()

        # clear variables
        self.cost_grid = None
        self.learner = None
        self.selected_data = None
        self.data = None
        self.set_empty_plot()

        self.send_output()

        self.cby.setDisabled(False)
        self.target_class_combobox.setDisabled(False)
        self.learner_name = ""

        if data is None or len(data) == 0:
            reset_combos()
        elif d.domain.class_var is None:
            reset_combos()
            self.Error.no_class()
        elif d.domain.class_var.is_continuous:
            if sum(True for var in d.domain.attributes
                   if isinstance(var, ContinuousVariable)) < 1:
                # not enough (2) continuous variable
                reset_combos()
                self.Error.to_few_features()
            else:
                self.data = data
                self.learner_name = "Linear regression"
                init_combos()
                self.attr_x = self.cbx.itemText(0)
                self.step_size_spin.setMaximum(len(d))
                self.restart()
        else:  # is discrete, if discrete logistic regression is used
            if sum(True for var in d.domain.attributes
                   if isinstance(var, ContinuousVariable)) < 2:
                # not enough (2) continuous variable
                reset_combos()
                self.Error.to_few_features()
            elif len(d.domain.class_var.values) < 2:
                reset_combos()
                self.Error.to_few_values()
                self.set_empty_plot()
            else:
                self.data = data
                self.learner_name = "Logistic regression"
                init_combos()
                self.attr_x = self.cbx.itemText(0)
                self.attr_y = self.cbx.itemText(1)
                self.target_class = self.target_class_combobox.itemText(0)
                self.step_size_spin.setMaximum(len(d))
                self.restart()

        self.learner_label.setText("Learner: " + self.learner_name)

    def set_empty_plot(self):
        """
        Function render empty plot
        """
        self.scatter.clear()

    def change_attributes(self):
        """
        Function changes when user changes attribute or target
        """
        self.learner = None  # that theta does not same equal
        self.restart()

    def restart(self):
        """
        Function restarts the algorithm
        """
        self.selected_data = self.select_data()
        if self.selected_data is None:
            self.set_empty_plot()
            return

        theta = self.learner.history[0][0] if self.learner is not None else None
        selected_learner = (LogisticRegression
                            if self.learner_name == "Logistic regression"
                            else LinearRegression)
        self.learner = selected_learner(
            data=self.selected_data,
            alpha=self.alpha, stochastic=self.stochastic,
            theta=theta, step_size=self.step_size,
            intercept=(self.learner_name == "Linear regression"))
        self.replot()
        if theta is None:  # no previous theta exist
            self.change_theta(np.random.uniform(self.min_x, self.max_x),
                              np.random.uniform(self.min_y, self.max_y))
        else:
            self.change_theta(theta[0], theta[1])
        self.send_output()
        self.step_back_button_lock()

    def change_alpha(self):
        """
        Function changes alpha parameter of the algorithm
        """
        if self.learner is not None:
            self.learner.set_alpha(self.alpha)

    def change_stochastic(self):
        """
        Function changes switches between stochastic or usual algorithm
        """
        if self.learner is not None:
            self.learner.stochastic = self.stochastic
        self.step_size_lock()

    def change_step(self):
        if self.learner is not None:
            self.learner.stochastic_step_size = self.step_size

    def change_theta(self, x, y):
        """
        Function set new theta
        """
        if self.learner is not None:
            self.learner.set_theta([x, y])
            self.scatter.remove_series("path")
            self.scatter.remove_series("last_point")
            self.scatter.add_series([
                dict(id="last_point",
                     data=[dict(
                         x=x, y=y, dataLabels=dict(
                             enabled=True,
                             format='{0:.2f}'.format(
                                 self.learner.j(np.array([x, y]))),
                             verticalAlign='middle',
                             align="right",
                             style=dict(
                                 fontWeight="normal",
                                 textShadow=False
                             ))
                     )],
                     type="scatter", enableMouseTracking=False,
                     color="#ffcc00", marker=dict(radius=4)),
                dict(id="path", data=[dict(
                    x=x, y=y, h='{0:.2f}'.format(
                        self.learner.j(np.array([x, y]))))],
                     type="scatter", lineWidth=1,
                     color=self.line_color(),
                     marker=dict(
                         enabled=True, radius=2),
                     tooltip=dict(
                         pointFormat="Cost: {point.h}",
                         shared=False,
                         valueDecimals=2
                     ))])
            self.send_output()

    def step(self):
        """
        Function performs one step of the algorithm
        """
        if self.data is None:
            return
        if self.learner.step_no > 500:  # limit step no to avoid freezes
            return
        self.learner.step()
        theta = self.learner.theta
        self.plot_point(theta[0], theta[1])
        self.send_output()
        self.step_back_button_lock()

    def step_back(self):
        """
        Function performs step back
        """
        if self.data is None:
            return
        if self.learner.step_no > 0:
            self.learner.step_back()
            self.scatter.remove_last_point("path")
            theta = self.learner.theta
            self.plot_last_point(theta[0], theta[1])
            self.send_output()
        self.step_back_button_lock()

    def step_back_button_lock(self):
        """
        Function lock or unlock step back button.
        """
        self.step_back_button.setDisabled(
            self.learner is None or self.learner.step_no == 0)

    def step_size_lock(self):
        self.step_size_spin.setDisabled(not self.stochastic)

    def plot_point(self, x, y):
        """
        Function add point to the path
        """
        self.scatter.add_point_to_series("path", dict(
            x=x, y=y, h='{0:.2f}'.format(self.learner.j(np.array([x, y])))
        ))
        self.plot_last_point(x, y)

    def plot_last_point(self, x, y):
        self.scatter.remove_last_point("last_point")
        self.scatter.add_point_to_series(
            "last_point",
            dict(
                x=x, y=y, dataLabels=dict(
                    enabled=True,
                    format='{0:.2f}'.format(
                        self.learner.j(np.array([x, y]))),
                    verticalAlign='middle',
                    align="left" if self.label_right() else "right",
                    style=dict(
                        fontWeight="normal",
                        textShadow=False
                    ))
            ))

    def label_right(self):
        l = self.learner
        return l.step_no == 0 or l.history[l.step_no - 1][0][0] < l.theta[0]

    def gradient_color(self):
        if not self.is_logistic:
            return self.default_background_color
        else:
            target_class_idx = self.data.domain.class_var.values.\
                index(self.target_class)
            color = self.data.domain.class_var.colors[target_class_idx]
            return rgb_to_hex(tuple(color))

    def line_color(self):
        rgb_tuple = hex_to_rgb(self.current_gradient_color)
        max_index, _ = max(enumerate(rgb_tuple), key=operator.itemgetter(1))
        return self.line_colors[max_index]

    def replot(self):
        """
        This function performs complete replot of the graph
        """
        if self.data is None or self.selected_data is None:
            self.set_empty_plot()
            return

        optimal_theta = self.learner.optimized()
        self.min_x = optimal_theta[0] - 10
        self.max_x = optimal_theta[0] + 10
        self.min_y = optimal_theta[1] - 10
        self.max_y = optimal_theta[1] + 10

        options = dict(series=[])

        # gradient and contour
        options['series'] += self.plot_gradient_and_contour(
            self.min_x, self.max_x, self.min_y, self.max_y)

        # select gradient color
        self.current_gradient_color = self.gradient_color()

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text="Θ {attr}".format(
                attr=self.attr_x if self.is_logistic else 0),
            yAxis_title_text="Θ {attr}".format(
                attr=self.attr_y if self.is_logistic else self.attr_x),
            xAxis_min=self.min_x,
            xAxis_max=self.max_x,
            yAxis_min=self.min_y,
            yAxis_max=self.max_y,
            xAxis_startOnTick=False,
            xAxis_endOnTick=False,
            yAxis_startOnTick=False,
            yAxis_endOnTick=False,
            colorAxis=dict(
                labels=dict(enabled=False),
                minColor="#ffffff", maxColor=self.current_gradient_color,
                endOnTick=False, startOnTick=False),
            plotOptions_contour_colsize=(self.max_y - self.min_y) / 1000,
            plotOptions_contour_rowsize=(self.max_x - self.min_x) / 1000,
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

        return (self.plot_gradient(xv, yv, self.cost_grid) +
                self.plot_contour(xv, yv, self.cost_grid))

    def plot_gradient(self, x, y, grid):
        """
        Function constructs background gradient
        """
        return [dict(data=[[x[j, k], y[j, k], grid[j, k]] for j in range(len(x))
                           for k in range(y.shape[1])],
                     grid_width=self.grid_size,
                     type="contour")]

    def plot_contour(self, xv, yv, cost_grid):
        """
        Function constructs contour lines
        """
        contour = Contour(xv, yv, cost_grid)
        contour_lines = contour.contours(
            np.linspace(np.min(cost_grid), np.max(cost_grid), 20))

        series = []
        count = 0
        for key, value in contour_lines.items():
            for line in value:
                if len(line) > 3:
                    tck, u = splprep(np.array(line).T, u=None, s=0.0, per=0)
                    u_new = np.linspace(u.min(), u.max(), 100)
                    x_new, y_new = splev(u_new, tck, der=0)
                    interpol_line = np.c_[x_new, y_new]
                else:
                    interpol_line = line

                series.append(dict(data=interpol_line,
                                   color=self.contour_color,
                                   type="spline",
                                   lineWidth=0.5,
                                   marker=dict(enabled=False),
                                   name="%g" % round(key, 2),
                                   enableMouseTracking=False
                                   ))
                count += 1
        return series

    def select_data(self):
        """
        Function takes two selected columns from data table and merge them
        in new Orange.data.Table

        Returns
        -------
        Table
            Table with selected columns
        """
        if self.data is None:
            return

        self.Error.clear()

        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y] if self.is_logistic else None
        cols = []
        for attr in (attr_x, attr_y) if attr_y is not None else (attr_x, ):
            subset = self.data[:, attr]
            cols.append(subset.X if not sp.issparse(subset.X) else subset.X.toarray())
        x = np.column_stack(cols)
        y_c = self.data.Y if not sp.issparse(self.data.Y) else self.data.Y.toarray()
        if y_c.ndim == 2 and y_c.shape[1] == 1:
            y_c = y_c.flatten()
        # remove nans
        indices = ~np.isnan(x).any(axis=1) & ~np.isnan(y_c)
        x = x[indices]
        y_c = y_c[indices]

        if len(x) == 0:
            self.Error.all_none()
            return None

        if self.is_logistic:
            two_classes = len(self.data.domain.class_var.values) == 2
            if two_classes:
                domain = Domain([attr_x, attr_y], [self.data.domain.class_var])
            else:
                domain = Domain(
                    [attr_x, attr_y],
                    [DiscreteVariable(
                        name=self.data.domain.class_var.name + "-bin",
                        values=(self.target_class, 'Others'))],
                    [self.data.domain.class_var])

            y = [(0 if self.data.domain.class_var.values[int(d)] ==
                       self.target_class else 1)
                 for d in y_c]

            return Normalize()(Table.from_numpy(domain, x, y_c)
                               if two_classes else
                               Table.from_numpy(domain, x, y, y_c[:, None]))
        else:
            domain = Domain([attr_x], self.data.domain.class_var)
            return Normalize(transform_class=True)(
                Table.from_numpy(domain, x, y_c)
            )

    def auto_play(self):
        """
        Function called when autoplay button pressed
        """
        if self.data is not None:
            self.auto_play_enabled = not self.auto_play_enabled
            self.auto_play_button.setText(
                self.auto_play_button_text[self.auto_play_enabled])
            if self.auto_play_enabled:
                self.disable_controls(self.auto_play_enabled)
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
        """
        Function sends output
        """
        self.send_model()
        self.send_coefficients()
        self.send_data()

    def send_model(self):
        """
        Function sends model on output.
        """
        if self.learner is not None and self.learner.theta is not None:
            self.Outputs.model.send(self.learner.model)
        else:
            self.Outputs.model.send(None)

    def send_coefficients(self):
        """
        Function sends logistic regression coefficients on output.
        """
        if self.learner is not None and self.learner.theta is not None:
            domain = Domain(
                    [ContinuousVariable("Coefficients")],
                    metas=[StringVariable("Name")])
            names = ["theta 0", "theta 1"]

            coefficients_table = Table.from_list(
                    domain, list(zip(list(self.learner.theta), names)))
            self.Outputs.coefficients.send(coefficients_table)
        else:
            self.Outputs.coefficients.send(None)

    def send_data(self):
        """
        Function sends data on output.
        """
        if self.selected_data is not None:
            self.Outputs.data.send(self.selected_data)
        else:
            self.Outputs.data.send(None)

    key_actions = {(0, Qt.Key_Space): step}  # space button for step

    def keyPressEvent(self, e):
        """
        Handle default key actions in this widget
        """
        if (int(e.modifiers()), e.key()) in self.key_actions:
            fun = self.key_actions[(int(e.modifiers()), e.key())]
            fun(self)
        else:
            super(OWGradientDescent, self).keyPressEvent(e)

    @property
    def is_logistic(self):
        return self.learner_name == "Logistic regression"

    def send_report(self):
        if self.data is None:
            return
        caption_items = (
            ("Target class", self.target_class),
            ("Learning rate", self.alpha),
            ("Stochastic", str(self.stochastic))
        )
        if self.stochastic:
            caption_items += (("Stochastic step size", self.step_size),)
        caption = report.render_items_vert(caption_items)
        self.report_plot(self.scatter)
        self.report_caption(caption)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWGradientDescent).run(Table.from_file('iris'))
