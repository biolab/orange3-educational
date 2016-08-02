from math import isnan
from os import path

import numpy as np
from Orange.widgets.utils import itemmodels
from PyQt4.QtCore import pyqtSlot, Qt
from PyQt4.QtGui import QSizePolicy, QPixmap, QColor, QIcon

from Orange.classification import Model
from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets import highcharts
from Orange.widgets import settings
from Orange.widgets.widget import OWWidget, Msg
from scipy.ndimage import gaussian_filter

from orangecontrib.educational.widgets.utils.logistic_regression \
    import LogisticRegression


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
                         plotOptions_series_cursor="move",
                         javascript=contours_js,
                         **kwargs)

        self.click_callback = click_callback

    def chart(self, *args, **kwargs):
        self.count_replots += 1
        super(Scatterplot, self).chart(*args, **kwargs)

    @pyqtSlot(float, float)
    def chart_clicked(self, x, y):
        self.click_callback(x, y)


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

    # models
    x_var_model = None
    y_var_model = None

    # function used in gradient descent
    default_learner = LogisticRegression
    learner = None
    cost_grid = None
    grid_size = 20

    # data
    data = None
    selected_data = None

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

        self.options_box = gui.widgetBox(self.controlArea)
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

        # graph in mainArea
        self.scatter = Scatterplot(click_callback=self.graph_clicked,
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

    def restart(self):
        self.selected_data = self.select_data()
        self.learner = self.default_learner(data=self.selected_data)
        self.replot()

    def replot(self):
        """
        This function performs complete replot of the graph
        """
        if self.data is None:
            return

        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]

        optimal_theta = self.learner.optimized()
        min_x = optimal_theta[0] - 3
        max_x = optimal_theta[0] + 3
        min_y = optimal_theta[1] - 3
        max_y = optimal_theta[1] + 3

        options = dict(series=[])

        # gradient and contour
        options['series'] += self.plot_gradient_and_contour(
            min_x, max_x, min_y, max_y)

        data = options['series'][0]['data']
        data = [d[2] for d in data]
        min_value = np.min(data)
        max_value = np.max(data)

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            xAxis_min=min_x,
            xAxis_max=max_x,
            yAxis_min=min_y,
            yAxis_max=max_y,
            colorAxis=dict(
                stops=[
                    [min_value, "#ffffff"],
                    [max_value, "#ff0000"]],
                tickInterval=1, max=max_value, min=min_value),
            plotOptions_contour_colsize=(max_y - min_y) / 1000,
            plotOptions_contour_rowsize=(max_x - min_x) / 1000,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y))

        self.scatter.chart(options, **kwargs)
            # hack to destroy the legend for coloraxis

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

        return self.plot_gradient(self.xv, self.yv, blurred)

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

    # def plot_contour(self):
    #     """
    #     Function constructs contour lines
    #     """
    #     self.scatter.remove_contours()
    #     if self.contours_enabled:
    #         contour = Contour(
    #             self.xv, self.yv, self.blur_grid(self.probabilities_grid))
    #         contour_lines = contour.contours(
    #             np.hstack(
    #                 (np.arange(0.5, 0, - self.contour_step)[::-1],
    #                  np.arange(0.5 + self.contour_step, 1, self.contour_step))))
    #         # we want to have contour for 0.5
    #
    #         series = []
    #         count = 0
    #         for key, value in contour_lines.items():
    #             for line in value:
    #                 if len(line) > self.degree:
    #                     # if less than degree interpolation fails
    #                     tck, u = splprep(
    #                         [list(x) for x in zip(*reversed(line))],
    #                         s=0.001, k=self.degree,
    #                         per=(len(line)
    #                              if np.allclose(line[0], line[-1])
    #                              else 0))
    #                     new_int = np.arange(0, 1.01, 0.01)
    #                     interpol_line = np.array(splev(new_int, tck)).T.tolist()
    #                 else:
    #                     interpol_line = line
    #
    #                 series.append(dict(data=self.labeled(interpol_line, count),
    #                                    color=self.contour_color,
    #                                    type="spline",
    #                                    lineWidth=0.5,
    #                                    showInLegend=False,
    #                                    marker=dict(enabled=False),
    #                                    name="%g" % round(key, 2),
    #                                    enableMouseTracking=False
    #                                    ))
    #                 count += 1
    #         self.scatter.add_series(series)
    #     self.scatter.redraw_series()

    @staticmethod
    def blur_grid(grid):
        filtered = gaussian_filter(grid, sigma=1)
        filtered[(grid > 0.45) & (grid < 0.55)] = grid[(grid > 0.45) &
                                                       (grid < 0.55)]
        return filtered

    def graph_clicked(self, x, y):
        pass

