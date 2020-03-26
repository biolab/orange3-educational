import math

from Orange.evaluation import RMSE, TestOnTrainingData, MAE
from AnyQt.QtCore import Qt, QRectF
from AnyQt.QtGui import QColor, QPalette, QPen, QFont

import sklearn.preprocessing as skl_preprocessing
import pyqtgraph as pg
import numpy as np

from Orange.data import Table, Domain
from Orange.data.variable import ContinuousVariable, StringVariable
from Orange.regression.linear import (RidgeRegressionLearner, PolynomialLearner,
                                      LinearRegressionLearner)
from Orange.regression import Learner
from Orange.widgets import settings, gui
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Msg, Input, Output
from orangewidget.report import report



class OWUnivariateRegression(OWBaseLearner):
    name = "Polynomial Regression"
    description = "Univariate regression with polynomial expansion."
    keywords = ["polynomial regression", "regression",
                "regression visualization", "polynomial features"]
    icon = "icons/UnivariateRegression.svg"
    priority = 500

    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, default=True)
        data = Output("Data", Table)

    replaces = [
        "Orange.widgets.regression.owunivariateregression."
        "OWUnivariateRegression",
        "orangecontrib.prototypes.widgets.owpolynomialregression."
        "OWPolynomialRegression"
    ]

    LEARNER = PolynomialLearner

    learner_name = settings.Setting("Univariate Regression")

    polynomialexpansion = settings.Setting(1)

    x_var_index = settings.ContextSetting(0)
    y_var_index = settings.ContextSetting(1)
    error_bars_enabled = settings.Setting(False)

    default_learner_name = "Linear Regression"
    error_plot_items = []

    rmse = ""
    mae = ""
    regressor_name = ""

    want_main_area = True
    graph_name = 'plot'

    class Error(OWBaseLearner.Error):
        all_none = Msg("One of the features has no defined values")
        no_cont_variables = Msg("Polynomial Regression requires at least one numeric feature.")

    def add_main_layout(self):

        self.data = None
        self.learner = None

        self.scatterplot_item = None
        self.plot_item = None

        self.x_label = 'x'
        self.y_label = 'y'

        self.rmse = ""
        self.mae = ""
        self.regressor_name = self.default_learner_name

        # info box
        info_box = gui.vBox(self.controlArea, "Info")
        self.regressor_label = gui.label(
            widget=info_box, master=self,
            label="Regressor: %(regressor_name).30s")
        gui.label(widget=info_box, master=self,
            label="Mean absolute error: %(mae).6s")
        gui.label(widget=info_box, master=self,
                  label="Root mean square error: %(rmse).6s")

        box = gui.vBox(self.controlArea, "Variables")

        self.x_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesX = gui.comboBox(
            box, self, value='x_var_index', label="Input: ",
            orientation=Qt.Horizontal, callback=self.apply)
        self.comboBoxAttributesX.setModel(self.x_var_model)
        self.expansion_spin = gui.doubleSpin(
            gui.indentedBox(box),
            self, "polynomialexpansion", 0, 10,
            label="Polynomial expansion:", callback=self.apply)

        gui.separator(box, height=8)
        self.y_var_model = itemmodels.VariableListModel()
        self.comboBoxAttributesY = gui.comboBox(
            box, self, value="y_var_index", label="Target: ",
            orientation=Qt.Horizontal, callback=self.apply)
        self.comboBoxAttributesY.setModel(self.y_var_model)

        properties_box = gui.vBox(self.controlArea, "Properties")
        self.error_bars_checkbox = gui.checkBox(
            widget=properties_box, master=self, value='error_bars_enabled',
            label="Show error bars", callback=self.apply)

        gui.rubber(self.controlArea)

        # main area GUI
        self.plotview = pg.PlotWidget(background="w")
        self.plot = self.plotview.getPlotItem()

        axis_color = self.palette().color(QPalette.Text)
        axis_pen = QPen(axis_color)

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        axis = self.plot.getAxis("bottom")
        axis.setLabel(self.x_label)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        axis = self.plot.getAxis("left")
        axis.setLabel(self.y_label)
        axis.setPen(axis_pen)
        axis.setTickFont(tickfont)

        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0),
                           disableAutoRange=True)

        self.mainArea.layout().addWidget(self.plotview)

    def send_report(self):
        if self.data is None:
            return
        caption = report.render_items_vert((
             ("Polynomial Expansion: ", self.polynomialexpansion),
        ))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    def clear(self):
        self.data = None
        self.rmse = ""
        self.mae = ""
        self.clear_plot()

    def clear_plot(self):
        if self.plot_item is not None:
            self.plot_item.setParentItem(None)
            self.plotview.removeItem(self.plot_item)
            self.plot_item = None

        if self.scatterplot_item is not None:
            self.scatterplot_item.setParentItem(None)
            self.plotview.removeItem(self.scatterplot_item)
            self.scatterplot_item = None

        self.remove_error_items()

        self.plotview.clear()

    @check_sql_input
    def set_data(self, data):
        self.clear()
        self.Error.no_cont_variables.clear()
        if data is not None:
            cvars = [var for var in data.domain.variables if var.is_continuous]
            class_cvars = [var for var in data.domain.class_vars
                           if var.is_continuous]

            nvars = len(cvars)
            nclass = len(class_cvars)
            self.x_var_model[:] = cvars
            self.y_var_model[:] = cvars
            if nvars == 0:
                self.data = None
                self.Error.no_cont_variables()
                return

            self.x_var_index = min(max(0, self.x_var_index), nvars - 1)
            if nclass > 0:
                self.y_var_index = min(max(0, nvars-nclass), nvars - 1)
            else:
                self.y_var_index = min(max(0, nvars-1), nvars - 1)
        self.data = data

    @Inputs.learner
    def set_learner(self, learner):
        self.learner = learner
        self.regressor_name = (learner.name if learner is not None else self.default_learner_name)

    def handleNewSignals(self):
        self.apply()

    def plot_scatter_points(self, x_data, y_data):
        if self.scatterplot_item:
            self.plotview.removeItem(self.scatterplot_item)
        self.n_points = len(x_data)
        self.scatterplot_item = pg.ScatterPlotItem(
            x=x_data, y=y_data, data=np.arange(self.n_points),
            symbol="o", size=10, pen=pg.mkPen(0.2), brush=pg.mkBrush(0.7),
            antialias=True)
        self.scatterplot_item.opts["useCache"] = False
        self.plotview.addItem(self.scatterplot_item)
        self.plotview.replot()

    def set_range(self, x_data, y_data):
        min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
        min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
        self.plotview.setRange(
            QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
            padding=0.025)
        self.plotview.replot()

    def plot_regression_line(self, x_data, y_data):
        if self.plot_item:
            self.plotview.removeItem(self.plot_item)
        self.plot_item = pg.PlotCurveItem(
            x=x_data, y=y_data,
            pen=pg.mkPen(QColor(255, 0, 0), width=3),
            antialias=True
        )
        self.plotview.addItem(self.plot_item)
        self.plotview.replot()

    def remove_error_items(self):
        for it in self.error_plot_items:
            self.plotview.removeItem(it)
        self.error_plot_items = []

    def plot_error_bars(self, x,  actual, predicted):
        self.remove_error_items()
        if self.error_bars_enabled:
            for x, a, p in zip(x, actual, predicted):
                line = pg.PlotCurveItem(
                    x=[x, x], y=[a, p],
                    pen=pg.mkPen(QColor(150, 150, 150), width=1),
                    antialias=True)
                self.plotview.addItem(line)
                self.error_plot_items.append(line)
        self.plotview.replot()

    def apply(self):
        degree = int(self.polynomialexpansion)
        learner = self.LEARNER(
            preprocessors=self.preprocessors, degree=degree,
            learner=LinearRegressionLearner() if self.learner is None
            else self.learner)
        learner.name = self.learner_name
        predictor = None

        self.Error.all_none.clear()

        if self.data is not None:
            attributes = self.x_var_model[self.x_var_index]
            class_var = self.y_var_model[self.y_var_index]
            data_table = Table.from_table(
                Domain([attributes], class_vars=[class_var]), self.data
            )

            # all lines has nan
            if sum(math.isnan(line[0]) or math.isnan(line.get_class())
                   for line in data_table) == len(data_table):
                self.Error.all_none()
                self.clear_plot()
                return

            predictor = learner(data_table)

            preprocessed_data = data_table
            for preprocessor in learner.active_preprocessors:
                preprocessed_data = preprocessor(preprocessed_data)

            x = preprocessed_data.X.ravel()
            y = preprocessed_data.Y.ravel()

            linspace = np.linspace(
                np.nanmin(x), np.nanmax(x), 1000).reshape(-1,1)
            values = predictor(linspace, predictor.Value)

            # calculate prediction for x from data
            validation = TestOnTrainingData()
            predicted = validation(preprocessed_data, [learner])
            self.rmse = round(RMSE(predicted)[0], 6)
            self.mae = round(MAE(predicted)[0], 6)

            # plot error bars
            self.plot_error_bars(
                x, predicted.actual, predicted.predicted.ravel())

            # plot data points
            self.plot_scatter_points(x, y)

            # plot regression line
            self.plot_regression_line(linspace.ravel(), values.ravel())

            x_label = self.x_var_model[self.x_var_index]
            axis = self.plot.getAxis("bottom")
            axis.setLabel(x_label)

            y_label = self.y_var_model[self.y_var_index]
            axis = self.plot.getAxis("left")
            axis.setLabel(y_label)

            self.set_range(x, y)

        self.Outputs.learner.send(learner)
        self.Outputs.model.send(predictor)

        # Send model coefficents
        model = None
        if predictor is not None:
            model = predictor.model
            if hasattr(model, "model"):
                model = model.model
            elif hasattr(model, "skl_model"):
                model = model.skl_model
        if model is not None and hasattr(model, "coef_"):
            domain = Domain([ContinuousVariable("coef")],
                            metas=[StringVariable("name")])
            coefs = [model.intercept_ + model.coef_[0]] + list(model.coef_[1:])
            names = ["1", x_label] + \
                    ["{}^{}".format(x_label, i) for i in range(2, degree + 1)]
            coef_table = Table.from_list(domain, list(zip(coefs, names)))
            self.Outputs.coefficients.send(coef_table)
        else:
            self.Outputs.coefficients.send(None)

        self.send_data()

    def send_data(self):
        if self.data is not None:
            attributes = self.x_var_model[self.x_var_index]
            class_var = self.y_var_model[self.y_var_index]

            data_table = Table.from_table(
                Domain([attributes], class_vars=[class_var]), self.data)
            polyfeatures = skl_preprocessing.PolynomialFeatures(
                int(self.polynomialexpansion))

            valid_mask = ~np.isnan(data_table.X).any(axis=1)
            x = data_table.X[valid_mask]
            x = polyfeatures.fit_transform(x)
            x_label = data_table.domain.attributes[0].name

            out_array = np.concatenate((x, data_table.Y[np.newaxis].T[valid_mask]), axis=1)

            out_domain = Domain(
                [ContinuousVariable("1")] + ([data_table.domain.attributes[0]]
                                             if self.polynomialexpansion > 0
                                             else []) +
                [ContinuousVariable("{}^{}".format(x_label, i))
                 for i in range(2, int(self.polynomialexpansion) + 1)], class_vars=[class_var])

            self.Outputs.data.send(Table.from_numpy(out_domain, out_array))
            return

        self.Outputs.data.send(None)

    def add_bottom_buttons(self):
        pass


if __name__ == "__main__":
    import sys
    from AnyQt.QtWidgets import QApplication

    a = QApplication(sys.argv)
    ow = OWUnivariateRegression()
    learner = RidgeRegressionLearner(alpha=1.0)
    polylearner = PolynomialLearner(learner, degree=2)
    d = Table('iris')
    ow.set_data(d)
    ow.set_learner(learner)
    ow.show()
    a.exec_()
    ow.saveSettings()
