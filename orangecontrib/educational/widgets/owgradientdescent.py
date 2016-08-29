from Orange.canvas import report
from os import path
import time

import numpy as np
from PyQt4.QtCore import pyqtSlot, Qt, QThread, pyqtSignal
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
from scipy.interpolate import splprep, splev

from orangecontrib.educational.widgets.utils.linear_regression import \
    LinearRegression
from orangecontrib.educational.widgets.utils.logistic_regression \
    import LogisticRegression
from orangecontrib.educational.widgets.utils.contour import Contour
import orangecontrib.educational.optimizers as opt


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
                         chart_panning=False,
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
        while (not self.ow_gradient_descent.learner.converged and
               self.ow_gradient_descent.auto_play_enabled and
               self.ow_gradient_descent.learner.step_no <= 500):
            self.ow_gradient_descent.step_trigger.emit()
            time.sleep(2 - self.ow_gradient_descent.auto_play_speed)
        self.ow_gradient_descent.stop_auto_play_trigger.emit()


class OWGradientDescent(OWWidget):
    """
    Gradient descent widget algorithm
    """

    name = "Gradient Descent"
    description = "Widget shows the procedure of gradient descent " \
                  "on logistic regression."
    icon = "icons/GradientDescent.svg"
    want_main_area = True

    inputs = [("Data", Table, "set_data")]
    outputs = [("Model", Model),
               ("Coefficients", Table),
               ("Data", Table)]

    graph_name = "Gradient descent graph"

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')
    target_class = settings.Setting('')
    alpha = settings.Setting(0.1)
    step_size = settings.Setting(30)  # step size for stochastic gds
    auto_play_speed = settings.Setting(1)
    stochastic = settings.Setting(False)

    # SGD optimizers
    class _Optimizer:
        SGD, MOMENTUM, NAG, ADAGRAD, RMSPROP, ADADELTA, ADAM, ADAMAX = range(8)
        names = ['Vanilla SGD', 'Momentum', "Nesterov momentum", 'AdaGrad',
                 'RMSprop', 'AdaDelta', 'Adam', 'Adamax']
    opt_type = settings.Setting(_Optimizer.SGD)
    momentum = settings.Setting(0.9)
    rho = settings.Setting(0.9)
    beta1 = settings.Setting(0.9)
    beta2 = settings.Setting(0.999)

    # models
    x_var_model = None
    y_var_model = None

    # function used in gradient descent
    learner_name = ""
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
        self.learner_label = gui.label(widget=self.info_box, master=self, label="")

        # options box
        policy = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)

        self.options_box = gui.widgetBox(self.controlArea, "Data")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal,
            callback=self.change_attributes, sendSelectedValue=True,
            maximumContentsLength=15
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
            minv=1e-5, maxv=1e5, step=0.001, spinType=float, decimals=3,
            alignment=Qt.AlignRight, controlWidth=80)
        self.step_size_spin = gui.spin(
            widget=self.properties_box, master=self, callback=self.change_step,
            value="step_size", label="Num. samples: ",
            minv=1, maxv=100, step=1, alignment=Qt.AlignRight, controlWidth=80)
        self.stochastic_checkbox = gui.checkBox(
            widget=self.properties_box, master=self,
            callback=self.change_stochastic, value="stochastic",
            label="Stochastic ")

        self.combobox_opt = gui.comboBox(
            widget=self.properties_box, master=self, value="opt_type",
            label="SGD optimizer: ", items=self._Optimizer.names,
            orientation=Qt.Horizontal, addSpace=4, callback=self._opt_changed)

        paramtip = "Controls the 'inertia' of the update.\nHigher momentum " \
                   "results in smoothing over more update steps"
        _m_comp = gui.doubleSpin(
            widget=self.properties_box, master=self, value="momentum",
            minv=1e-4, maxv=1e+4, step=1e-4, label="Momentum:", decimals=4,
            alignment=Qt.AlignRight, controlWidth=90,
            callback=self._opt_changed, tooltip=paramtip)

        paramtip = "Decay rate of the gradient moving average"
        _r_comp = gui.doubleSpin(
            widget=self.properties_box, master=self, value="rho", minv=1e-4,
            maxv=1e+4, step=1e-4, label="Rho:", decimals=4,
            alignment=Qt.AlignRight, controlWidth=90,
            callback=self._opt_changed, tooltip=paramtip)

        paramtip = "Exponential decay rate for the 1st moment estimates " \
                   "(the mean)"
        _b1_comp = gui.doubleSpin(
            widget=self.properties_box, master=self, value="beta1", minv=1e-5,
            maxv=1e+5, step=1e-4, label="Beta 1:", decimals=5,
            alignment=Qt.AlignRight, controlWidth=90,
            callback=self._opt_changed, tooltip=paramtip)

        paramtip = "Exponential decay rate for the 2nd moment estimates (the " \
                   "uncentered variance)"
        _b2_comp = gui.doubleSpin(
            widget=self.properties_box, master=self, value="beta2", minv=1e-5,
            maxv=1e+5, step=1e-4, label="Beta 2:", decimals=5,
            alignment=Qt.AlignRight, controlWidth=90,
            callback=self._opt_changed, tooltip=paramtip)

        self._opt_params = [_m_comp, _r_comp, _b1_comp, _b2_comp]
        self._show_right_optimizer()

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
                                   debug=True,
                                   legend_symbolWidth=0,
                                   legend_symbolHeight=0)
        # TODO: set false when end of development
        gui.rubber(self.controlArea)

        # Just render an empty chart so it shows a nice 'No data to display'
        self.scatter.chart()
        self.mainArea.layout().addWidget(self.scatter)

        self.step_size_lock()
        self.step_back_button_lock()

    def _opt_changed(self):
        self._show_right_optimizer()
        self.select_optimizer()

    def _show_right_optimizer(self):
        enabled = [[False, False, False, False],  # SGD
                   [True, False, False, False],  # Momentum
                   [True, False, False, False],  # NAG
                   [False, False, False, False],  # AdaGrad
                   [False, True, False, False],  # RMSprop
                   [False, True, False, False],  # AdaDelta
                   [False, False, True, True],  # Adam
                   [False, False, True, True],  # Adamax
                ]

        mask = [False, False, False, False]
        self.combobox_opt.box.setVisible(self.stochastic)
        if self.stochastic:
            mask = enabled[self.opt_type]

        # All hidden
        for spin, enabled in zip(self._opt_params, mask):
            [spin.box.hide, spin.box.show][enabled]()

    def select_optimizer(self):
        if self.learner is not None:
            if self.opt_type == self._Optimizer.MOMENTUM:
                self.learner.sgd_optimizer = opt.Momentum(
                    momentum=self.momentum
                )
            elif self.opt_type == self._Optimizer.NAG:
                self.learner.sgd_optimizer = opt.NesterovMomentum(
                    momentum=self.momentum
                )
            elif self.opt_type == self._Optimizer.ADAGRAD:
                self.learner.sgd_optimizer = opt.AdaGrad()

            elif self.opt_type == self._Optimizer.RMSPROP:
                self.learner.sgd_optimizer = opt.RMSProp(rho=self.rho)

            elif self.opt_type == self._Optimizer.ADADELTA:
                self.learner.sgd_optimizer = opt.AdaDelta(rho=self.rho)

            elif self.opt_type == self._Optimizer.ADAM:
                self.learner.sgd_optimizer = opt.Adam(beta1=self.beta1,
                                                      beta2=self.beta2)

            elif self.opt_type == self._Optimizer.ADAMAX:
                self.learner.sgd_optimizer = opt.Adamax(beta1=self.beta1,
                                                        beta2=self.beta2)
            else:
                self.learner.sgd_optimizer = opt.SGD()

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
        self.select_optimizer()
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
        self._opt_changed()

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
                             format='&nbsp;{0:.2f}&nbsp;'.format(
                                 self.learner.j(np.array([x, y]))),
                             verticalAlign='middle',
                             useHTML=True,
                             align="right",
                             style=dict(
                                 fontWeight="normal",
                                 textShadow=False
                             ))
                     )],
                     showInLegend=False,
                     type="scatter", enableMouseTracking=False,
                     color="#ffcc00", marker=dict(radius=4)),
                dict(id="path", data=[dict(
                    x=x, y=y, h='{0:.2f}'.format(
                        self.learner.j(np.array([x, y]))))],
                     showInLegend=False,
                     type="scatter", lineWidth=1,
                     color="#ff0000",
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
                    format='&nbsp;{0:.2f}&nbsp;'.format(self.learner.j(np.array([x, y]))),
                    useHTML=True,
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

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text="<p>&theta;<sub>{attr}</sub></p>"
                .format(attr=self.attr_x if self.is_logistic else 0),
            xAxis_title_useHTML=True,
            yAxis_title_text="&theta;<sub>{attr}</sub>".
                format(attr=self.attr_y if self.is_logistic else self.attr_x),
            yAxis_title_useHTML=True,
            xAxis_min=self.min_x,
            xAxis_max=self.max_x,
            yAxis_min=self.min_y,
            yAxis_max=self.max_y,
            xAxis_startOnTick=False,
            xAxis_endOnTick=False,
            yAxis_startOnTick=False,
            yAxis_endOnTick=False,
            colorAxis=dict(
                minColor="#ffffff", maxColor="#00BFFF",
                endOnTick=False, startOnTick=False),
            plotOptions_contour_colsize=(self.max_y - self.min_y) / 1000,
            plotOptions_contour_rowsize=(self.max_x - self.min_x) / 1000,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y))

        self.scatter.chart(options, **kwargs)
        # to remove the colorAxis legend
        self.scatter.evalJS("chart.colorAxis[0].axisParent.destroy();")

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
                                   showInLegend=False,
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
            cols.append(subset.X)
        x = np.column_stack(cols)
        y_c = self.data.Y

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
                        values=[self.target_class, 'Others'])],
                    [self.data.domain.class_var])

            y = [(0 if self.data.domain.class_var.values[int(d)] ==
                       self.target_class else 1)
                 for d in y_c]

            return Normalize(Table(domain, x, y_c) if two_classes
                             else Table(domain, x, y, y_c[:, None]))
        else:
            domain = Domain([attr_x], self.data.domain.class_var)
            return Normalize(
                Table(domain, x, y_c), transform_class=True)

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
            self.send("Model", self.learner.model)
        else:
            self.send("Model", None)

    def send_coefficients(self):
        """
        Function sends logistic regression coefficients on output.
        """
        if self.learner is not None and self.learner.theta is not None:
            domain = Domain(
                    [ContinuousVariable("Coefficients", number_of_decimals=7)],
                    metas=[StringVariable("Name")])
            names = ["theta 0", "theta 1"]

            coefficients_table = Table(
                    domain, list(zip(list(self.learner.theta), names)))
            self.send("Coefficients", coefficients_table)
        else:
            self.send("Coefficients", None)

    def send_data(self):
        """
        Function sends data on output.
        """
        if self.selected_data is not None:
            self.send("Data", self.selected_data)
        else:
            self.send("Data", None)

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
        caption = report.render_items_vert((
             ("Stochastic", str(self.stochastic)),
        ))
        self.report_plot(self.scatter)
        self.report_caption(caption)