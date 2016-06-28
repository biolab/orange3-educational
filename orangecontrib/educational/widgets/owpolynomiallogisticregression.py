from Orange.data import Table, ContinuousVariable, Table, Domain
from Orange.widgets import highcharts, settings, gui
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.classification import LogisticRegressionLearner
import numpy as np


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

    def __init__(self, **kwargs):
        super().__init__(enable_zoom=True,
                         bridge=self,
                         enable_select='',
                         chart_type='scatter',
                         plotOptions_series_cursor="move",
                         **kwargs)


class OWPolyinomialLogisticRegression(OWBaseLearner):
    name = "Polynomial logistic regression"
    description = "a"  #TODO: description
    icon = "icons/mywidget.svg"
    want_main_area = True

    # inputs and outputs
    inputs = [("Data", Table, "set_data")]

    data = None
    selected_data = None
    learner = None

    LEARNER = LogisticRegressionLearner
    learner_name = settings.Setting("Univariate Classification")

    # selected attributes in chart
    attr_x = settings.Setting('')
    attr_y = settings.Setting('')

    graph_name = 'scatter'

    def add_main_layout(self):
        # options box
        self.optionsBox = gui.widgetBox(self.controlArea, "Options")
        self.cbx = gui.comboBox(self.optionsBox, self, 'attr_x',
                                label='X:',
                                orientation='horizontal',
                                callback=self.refresh(),
                                sendSelectedValue=True)
        self.cby = gui.comboBox(self.optionsBox, self, 'attr_y',
                                label='Y:',
                                orientation='horizontal',
                                callback=self.refresh(),
                                sendSelectedValue=True)

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
        # plot centroids
        options = dict(series=[])
        classes = list(set(self.data.Y))

        options['series'] += [dict(data=[list(p.attributes())
                                            for p in self.selected_data if int(p.get_class()) == _class],
                                      type="scatter",
                                      showInLegend=False) for _class in classes]

        # highcharts parameters
        kwargs = dict(
            xAxis_title_text=attr_x.name,
            yAxis_title_text=attr_y.name,
            tooltip_headerFormat="",
            tooltip_pointFormat="<strong>%s:</strong> {point.x:.2f} <br/>"
                                "<strong>%s:</strong> {point.y:.2f}" %
                                (self.attr_x, self.attr_y))
        # plot
        self.scatter.chart(options, **kwargs)

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
