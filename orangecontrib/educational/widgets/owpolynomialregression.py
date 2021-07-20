import math

from Orange.evaluation import RMSE, TestOnTrainingData, MAE
from AnyQt.QtCore import Qt, QRectF, QPointF
from AnyQt.QtGui import QColor, QPalette, QPen, QFont

import sklearn.preprocessing as skl_preprocessing
import pyqtgraph as pg
import numpy as np

from orangewidget.report import report
from orangewidget.utils.widgetpreview import WidgetPreview

from Orange.data import Table, Domain
from Orange.data.variable import ContinuousVariable, StringVariable
from Orange.regression.linear import (RidgeRegressionLearner, PolynomialLearner,
                                      LinearRegressionLearner)
from Orange.regression import Learner
from Orange.regression.mean import MeanModel
from Orange.statistics.distribution import Continuous
from Orange.widgets import settings, gui
from Orange.widgets.widget import Msg, Input, Output
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.sql import check_sql_input


class RegressTo0(Learner):
    @staticmethod
    def fit(*args, **kwargs):
        return MeanModel(Continuous(np.empty(0)))


class OWPolynomialRegression(OWBaseLearner):
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
        "orangecontrib.prototypes.widgets.owpolynomialregression.",
        "orangecontrib.educational.widgets.owunivariateregression."
    ]

    LEARNER = PolynomialLearner

    learner_name = settings.Setting("Polynomial Regression")

    polynomialexpansion = settings.Setting(1)

    settingsHandler = DomainContextHandler()
    x_var = settings.ContextSetting(None)
    y_var = settings.ContextSetting(None)
    error_bars_enabled = settings.Setting(False)
    fit_intercept = settings.Setting(True)

    default_learner_name = "Linear Regression"
    error_plot_items = []

    rmse = ""
    mae = ""
    regressor_name = ""

    want_main_area = True
    graph_name = 'plot'

    class Error(OWBaseLearner.Error):
        all_none = \
            Msg("All rows have undefined data.")
        no_cont_variables =\
            Msg("Regression requires at least two numeric variables.")
        same_dep_indepvar =\
            Msg("Dependent and independent variables must be differnt.")

    def __init__(self):
        super().__init__()
        self.data = None
        self.learner = None

        self.scatterplot_item = None
        self.plot_item = None

    def add_main_layout(self):
        self.rmse = ""
        self.mae = ""
        self.regressor_name = self.default_learner_name

        self.var_model = DomainModel(
            valid_types=(ContinuousVariable, ),
            order=DomainModel.MIXED)

        box = gui.vBox(self.controlArea, "Predictor")
        gui.comboBox(
            box, self, value='x_var', model=self.var_model, callback=self.apply)
        gui.spin(
            box, self, "polynomialexpansion", label="Polynomial degree: ",
            minv=0, maxv=10, alignment=Qt.AlignmentFlag.AlignRight,
            callback=self.apply)
        gui.checkBox(
            box, self, "fit_intercept",
            label="Fit intercept", callback=self.apply, stateWhenDisabled=True,
            tooltip="Add an intercept term;\n"
                    "This is always checked if the model is defined on input.")

        box = gui.vBox(self.controlArea, "Target")
        gui.comboBox(
            box, self, value="y_var", model=self.var_model, callback=self.apply)
        gui.checkBox(
            widget=box, master=self, value='error_bars_enabled',
            label="Show error bars", callback=self.apply)

        gui.rubber(self.controlArea)

        info_box = gui.vBox(self.controlArea, "Info")
        gui.label(
            widget=info_box, master=self,
            label="Regressor: %(regressor_name).30s")
        gui.label(
            widget=info_box, master=self,
            label="Mean absolute error: %(mae).6s")
        gui.label(
            widget=info_box, master=self,
            label="Root mean square error: %(rmse).6s")

        self.plotview = pg.PlotWidget(background="w")
        self.plot = self.plotview.getPlotItem()
        axis_pen = QPen(self.palette().color(QPalette.Text))
        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))
        for axis in ("bottom", "left"):
            axis = self.plot.getAxis(axis)
            axis.setPen(axis_pen)
            axis.setTickFont(tickfont)
        self.plot.setRange(xRange=(0.0, 1.0), yRange=(0.0, 1.0),
                           disableAutoRange=True)
        self.mainArea.layout().addWidget(self.plotview)

    def add_bottom_buttons(self):
        pass

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
        self.Error.clear()
        self.closeContext()
        self.data = data
        if data is None:
            self.var_model.set_domain(None)
            return

        self.var_model.set_domain(data.domain)
        if len(self.var_model) < 2:
            self.data = None
            self.x_var = self.y_var = None
            self.Error.no_cont_variables()
            return

        self.x_var = self.var_model[0]
        if data.domain.class_var in self.var_model:
            self.y_var = data.domain.class_var
        else:
            self.y_var = self.var_model[1]
        self.openContext(data)

    @Inputs.learner
    def set_learner(self, learner):
        self.learner = learner
        if learner is None:
            self.controls.fit_intercept.setDisabled(False)
            self.regressor_name = self.default_learner_name
        else:
            self.controls.fit_intercept.setDisabled(True)
            self.regressor_name = learner.name

    def handleNewSignals(self):
        self.apply()

    def plot_scatter_points(self, x_data, y_data):
        if self.scatterplot_item:
            self.plotview.removeItem(self.scatterplot_item)
        self.scatterplot_item = pg.ScatterPlotItem(
            x=x_data, y=y_data,
            symbol="o", size=10, pen=pg.mkPen(0.2), brush=pg.mkBrush(0.7),
            antialias=True)
        self.scatterplot_item.opts["useCache"] = False
        self.plotview.addItem(self.scatterplot_item)
        self.plotview.replot()

    def set_range(self, x_data, y_data):
        min_x, max_x = np.nanmin(x_data), np.nanmax(x_data)
        min_y, max_y = np.nanmin(y_data), np.nanmax(y_data)
        if self.polynomialexpansion == 0 and not self._has_intercept:
            if min_y > 0:
                min_y = -0.1 * max_y
            elif max_y < 0:
                max_y = -0.1 * min_y

        self.plotview.setRange(
            QRectF(min_x, min_y, max_x - min_x, max_y - min_y),
            padding=0.025)
        self.plotview.replot()

    def plot_regression_line(self, x_data, y_data):
        item = pg.PlotCurveItem(
            x=x_data, y=y_data,
            pen=pg.mkPen(QColor(255, 0, 0), width=3),
            antialias=True
        )
        self._plot_regression_item(item)

    def plot_infinite_line(self, x, y, angle):
        item = pg.InfiniteLine(
            QPointF(x, y), angle,
            pen=pg.mkPen(QColor(255, 0, 0), width=3))
        self._plot_regression_item(item)

    def _plot_regression_item(self, item):
        if self.plot_item:
            self.plotview.removeItem(self.plot_item)
        self.plot_item = item
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

    def _varnames(self, name):
        # If variable name is short, use superscripts
        # otherwise "^" because superscripts would be lost
        def ss(x):
            # Compose a (potentially non-single-digit) superscript
            return "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[i] for i in (int(c) for c in str(x)))

        if len(name) <= 3:
            return [f"{name}{ss(i)}"
                    for i in range(not self._has_intercept,
                                   1 + self.polynomialexpansion)]
        else:
            return ["intercept"] * self._has_intercept + \
                   [name] * (self.polynomialexpansion >= 1) + \
                   [f"{name}^{i}" for i in range(2, 1 + self.polynomialexpansion)]

    @property
    def _has_intercept(self):
        return self.learner is not None or self.fit_intercept

    def apply(self):
        degree = self.polynomialexpansion
        if degree == 0 and not self.fit_intercept:
            learner = RegressTo0()
        else:
            # For LinearRegressionLearner, set fit_intercept to False:
            # the intercept is added as bias term in polynomial expansion
            # If there is a learner on input, we have not control over this;
            # we include_bias to have the placeholder for the coefficient
            lin_learner = self.learner \
                          or LinearRegressionLearner(fit_intercept=False)
            learner = self.LEARNER(
                preprocessors=self.preprocessors, degree=degree,
                include_bias=self.fit_intercept,
                learner=lin_learner)
        learner.name = self.learner_name
        predictor = None
        model = None

        self.Error.all_none.clear()
        self.Error.same_dep_indepvar.clear()

        if self.data is not None:
            if self.x_var is self.y_var:
                self.Error.same_dep_indepvar()
                self.clear_plot()
                return

            data_table = Table.from_table(
                Domain([self.x_var], self.y_var),
                self.data)

            # all lines has nan
            if np.all(np.isnan(data_table.X.flatten()) | np.isnan(data_table.Y)):
                self.Error.all_none()
                self.clear_plot()
                return

            predictor = learner(data_table)
            model = None
            if hasattr(predictor, "model"):
                model = predictor.model
                if hasattr(model, "model"):
                    model = model.model
                elif hasattr(model, "skl_model"):
                    model = model.skl_model

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
            x_data, y_data = linspace.ravel(), values.ravel()
            if self.polynomialexpansion == 0:
                self.plot_infinite_line(x_data[0], y_data[0], 0)
            elif self.polynomialexpansion == 1 and hasattr(model, "coef_"):
                k = model.coef_[1 if self._has_intercept else 0]
                self.plot_infinite_line(x_data[0], y_data[0],
                                        math.degrees(math.atan(k)))
            else:
                self.plot_regression_line(x_data, y_data)

            self.plot.getAxis("bottom").setLabel(self.x_var.name)
            self.plot.getAxis("left").setLabel(self.y_var.name)
            self.set_range(x, y)

        self.Outputs.learner.send(learner)
        self.Outputs.model.send(predictor)

        # Send model coefficents
        if model is not None and hasattr(model, "coef_"):
            domain = Domain([ContinuousVariable("coef")],
                            metas=[StringVariable("name")])
            names = self._varnames(self.x_var.name)
            coefs = list(model.coef_)
            if self._has_intercept:
                model.coef_[0] += model.intercept_
            coef_table = Table.from_list(domain, list(zip(coefs, names)))
            self.Outputs.coefficients.send(coef_table)
        else:
            self.Outputs.coefficients.send(None)

        self.send_data()

    def send_data(self):
        if self.data is not None:
            data_table = Table.from_table(
                Domain([self.x_var], self.y_var), self.data)
            polyfeatures = skl_preprocessing.PolynomialFeatures(
                self.polynomialexpansion, include_bias=self._has_intercept)

            valid_mask = ~np.isnan(data_table.X).any(axis=1)
            if not self._has_intercept and not self.polynomialexpansion:
                x = np.empty((len(data_table), 0))
            else:
                x = data_table.X[valid_mask]
                x = polyfeatures.fit_transform(x)

            out_array = np.hstack((x, data_table.Y[np.newaxis].T[valid_mask]))

            out_domain = Domain(
                [ContinuousVariable(name)
                 for name in self._varnames(self.x_var.name)],
                self.y_var)
            self.Outputs.data.send(Table.from_numpy(out_domain, out_array))
            return

        self.Outputs.data.send(None)

    def send_report(self):
        if self.data is None:
            return
        caption = report.render_items_vert((
            ("Polynomial Expansion", self.polynomialexpansion),
            ("Fit intercept",
             self._has_intercept and ["No", "Yes"][self.fit_intercept])
        ))
        self.report_plot()
        if caption:
            self.report_caption(caption)

    @classmethod
    def migrate_settings(cls, settings, version):
        # polynomialexpansion used to be controlled by doublespin and was hence
        # float. Just convert to `int`, ignore settings versions.
        settings["polynomialexpansion"] = \
            int(settings.get("polynomialexpansion", 1))


if __name__ == "__main__":
    learner = RidgeRegressionLearner(alpha=1.0)
    iris = Table('iris')
    WidgetPreview(OWPolynomialRegression).run(set_data=iris) #, set_learner=learner)
