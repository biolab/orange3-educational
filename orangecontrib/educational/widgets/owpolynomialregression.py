import math

from Orange.data.util import SharedComputeValue
from Orange.evaluation import RMSE, MAE, Results
from AnyQt.QtCore import Qt, QRectF, QPointF
from AnyQt.QtGui import QColor, QPalette, QPen, QFont

import sklearn.preprocessing as skl_preprocessing
import pyqtgraph as pg
import numpy as np

from orangewidget.report import report
from orangewidget.utils.widgetpreview import WidgetPreview

from Orange.data import Table, Domain
from Orange.data.variable import ContinuousVariable, StringVariable
from Orange.regression.linear import RidgeRegressionLearner, LinearRegressionLearner
from Orange.base import Learner
from Orange.regression.mean import MeanModel
from Orange.statistics.distribution import Continuous
from Orange.widgets import settings, gui
from Orange.widgets.widget import Msg, Input, Output
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.sql import check_sql_input


class PolynomialFeatureSharedCV(SharedComputeValue):
    def __init__(self, compute_shared, idx):
        super().__init__(compute_shared)
        self.idx = idx

    def compute(self, _, shared_data):
        return shared_data[:, self.idx]

    def __eq__(self, other):
        # Remove the first test after we require Orange 3.33
        return type(self) is type(other) \
               and super().__eq__(other) and self.idx == other.idx

    def __hash__(self):
        return hash((super().__hash__(), self.idx))


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def __call__(self, data):
        if self.degree == 0:
            # Zero degree without intercept shouldn't have a column of 1's,
            # otherwise we do have intercapt. But some data is needed by most
            # learners, so we provide a column of zeros
            variables = [
                ContinuousVariable(
                    "x0",
                    compute_value=TempMeanModel(int(self.include_bias)))]
        else:
            pf = skl_preprocessing.PolynomialFeatures(
                self.degree, include_bias=self.include_bias
            )
            pf.fit(data.X)
            cv = lambda table: pf.transform(table.transform(data.domain).X)
            features = pf.get_feature_names_out() if pf.n_output_features_ else []
            variables = [
                ContinuousVariable(f, compute_value=PolynomialFeatureSharedCV(cv, i))
                for i, f in enumerate(features)
            ]
        domain = Domain(
            variables,
            class_vars=data.domain.class_vars,
            metas=data.domain.metas,
        )
        return data.transform(domain)


class TempMeanModel(MeanModel):
    """
    Using MeanModel Model's __call__ is called that transform table to the
    original domain space which produces empty X - and so the error is raised
    Here we bypass model's __call__
    """
    InheritEq = True

    def __init__(self, const):
        distr = Continuous(np.array([[const], [1.0]]))
        super().__init__(distr)

    def __call__(self, data, *args, **kwargs):
        return self.predict(data)


class RegressTo0(Learner):
    @staticmethod
    def __call__(data, *args, **kwargs):
        model = TempMeanModel(0)
        return model


class PolynomialLearnerWrapper(Learner):
    def __init__(self, x_var, y_var, degree, learner, preprocessors, fit_intercept):
        super().__init__()
        self.x_var = x_var
        self.y_var = y_var
        self.degree = degree
        self.learner = learner
        self.preprocessors = preprocessors
        self.fit_intercept = fit_intercept

    def __call__(self, data: Table, progress_callback=None):
        data = data.transform(Domain([self.x_var], self.y_var))
        *_, model = self.data_and_model(data)
        return model

    def data_and_model(self, data: Table):
        """
        Trains the model, and also returns temporary tables

        The function is used in the widget instead of __call__ to avoid
        recomputing preprocessed and expanded data.
        """
        valid_mask = np.isfinite(data.get_column(self.x_var)) \
                     & np.isfinite(data.get_column(self.y_var))
        data_table = Table.from_table(
            Domain([self.x_var], self.y_var), data[valid_mask]
        )

        # all lines have nan
        if np.all(np.isnan(data_table.X.flatten()) | np.isnan(data_table.Y)):
            return None, None, None, None, None

        # apply preprocessors on the input first
        preprocessed_table = (
            self.preprocessors(data_table) if self.preprocessors else data_table
        )

        # use polynomial preprocessor after applying preprocessors from input
        poly_preprocessor = PolynomialFeatures(
            degree=self.degree, include_bias=self.fit_intercept
        )

        expanded_data = poly_preprocessor(preprocessed_table)
        predictor = self.learner(expanded_data)
        return (data_table, preprocessed_table, poly_preprocessor,
                expanded_data, predictor)


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
        coefficients = Output("Coefficients", Table, explicit=True)
        data = Output("Data", Table, explicit=True)


    replaces = [
        "Orange.widgets.regression.owunivariateregression."
        "OWUnivariateRegression",
        "orangecontrib.prototypes.widgets.owpolynomialregression.",
        "orangecontrib.educational.widgets.owunivariateregression."
    ]

    LEARNER = LinearRegressionLearner

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
    graph_name = 'plot'  # pg.GraphicsItem  (pg.PlotItem)

    class Warning(OWBaseLearner.Warning):
        large_diffs = Msg(
            "Polynomial feature values are very large. "
            "This may cause numerical instabilities."
        )

    class Error(OWBaseLearner.Error):
        all_none = Msg("All rows have undefined data.")
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
        def error_and_clear(error=None):
            if error:
                error()
            self.clear_plot()
            self.Outputs.data.send(None)
            self.Outputs.coefficients.send(None)
            self.Outputs.learner.send(None)
            self.Outputs.model.send(None)

        self.Error.all_none.clear()
        self.Error.same_dep_indepvar.clear()
        self.Warning.large_diffs.clear()
        if self.data is None:
            error_and_clear()
            return
        if self.x_var is self.y_var:
            error_and_clear(self.Error.same_dep_indepvar)
            return

        degree = self.polynomialexpansion
        if degree == 0 and not self.fit_intercept and (
                self.learner is None
                or not getattr(self.learner, "fit_intercept", True)):
            learner = RegressTo0()
        else:
            # For LinearRegressionLearner, set fit_intercept to False:
            # the intercept is added as bias term in polynomial expansion
            learner = self.learner \
                      or LinearRegressionLearner(fit_intercept=False)

        include_bias = self.learner is None and self.fit_intercept
        poly_learner = PolynomialLearnerWrapper(
            self.x_var, self.y_var, degree, learner,
            self.preprocessors, include_bias)
        poly_learner.name = self.learner_name

        data_table, preprocessed_table, poly_preprocessor, \
        expanded_data, predictor = poly_learner.data_and_model(self.data)
        if preprocessed_table is None:
            error_and_clear(self.Error.all_none)
            return
        if expanded_data.X.max() - expanded_data.X.min() > 1e14:
            # the threshold defined with experimenting, instability typically
            # started to have effects for values >= 1e15
            self.Warning.large_diffs()

        model = None
        if hasattr(predictor, "model"):
            model = predictor.model
        elif hasattr(predictor, "skl_model"):
            model = predictor.skl_model

        x = preprocessed_table.X.ravel()
        y = preprocessed_table.Y.ravel()

        linspace = Table.from_numpy(
            Domain(data_table.domain.attributes),
            np.linspace(np.nanmin(x), np.nanmax(x), 1000).reshape(-1, 1),
        )
        values = predictor(linspace, predictor.Value)

        predicted = predictor(data_table, predictor.Value)
        results = Results(
            domain=self.data.domain,
            nrows=len(data_table), learners=[poly_learner],
            row_indices=np.arange(len(data_table)),
            folds=(Ellipsis,),
            actual=data_table.Y,
            predicted=predicted[None, :])
        self.rmse = round(RMSE(results)[0], 6)
        self.mae = round(MAE(results)[0], 6)

        # plot error bars
        self.plot_error_bars(x, results.actual, results.predicted.ravel())

        # plot data points
        self.plot_scatter_points(x, y)

        # plot regression line
        x_data, y_data = linspace.X.ravel(), values.ravel()
        if self.polynomialexpansion == 0:
            self.plot_infinite_line(x_data[0], y_data[0], 0)
        elif self.polynomialexpansion == 1 and self.learner is None:
            k = model.coef_[1 if self._has_intercept else 0]
            self.plot_infinite_line(x_data[0], y_data[0],
                                    math.degrees(math.atan(k)))
        else:
            self.plot_regression_line(x_data, y_data)

        self.plot.getAxis("bottom").setLabel(self.x_var.name)
        self.plot.getAxis("left").setLabel(self.y_var.name)
        self.set_range(x, y)

        self.Outputs.learner.send(poly_learner)
        self.Outputs.model.send(predictor)

        # Send model coefficents
        if model is not None and hasattr(model, "coef_"):
            if getattr(learner, "fit_intercept", True):
                coefs = [model.intercept_]
            else:
                coefs = []
            coefs += list(model.coef_)
        elif self.learner is None \
                and isinstance(predictor, MeanModel) \
                and self.fit_intercept:
            coefs = [predictor.mean]
        else:
            coefs = None
        if coefs:
            domain = Domain([ContinuousVariable("coef")],
                            metas=[StringVariable("name")])
            names = self._varnames(self.x_var.name)
            coef_table = Table.from_list(domain, list(zip(coefs, names)))
            self.Outputs.coefficients.send(coef_table)
        else:
            self.Outputs.coefficients.send(None)

        self.Outputs.data.send(expanded_data)

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
    iris = Table("iris")
    WidgetPreview(OWPolynomialRegression).run(set_data=iris)  # , set_learner=learner)
