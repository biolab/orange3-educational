import copy
from colorsys import rgb_to_hsv, hsv_to_rgb
from itertools import chain

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage.filters import gaussian_filter

from AnyQt.QtCore import Qt, QRectF, QObject, QPointF, QEvent
from AnyQt.QtGui import QPalette, QPen, QFont, QCursor, QColor, QBrush
from AnyQt.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsTextItem

import pyqtgraph as pg

from Orange.data import \
    Table, Domain, ContinuousVariable, StringVariable, DiscreteVariable
from Orange.base import Learner
from Orange.classification import \
    LogisticRegressionLearner, RandomForestLearner, TreeLearner
from Orange.preprocess.transformation import Indicator

from Orange.widgets import gui
from Orange.widgets.settings import \
    DomainContextHandler, Setting, SettingProvider, ContextSetting
from Orange.widgets.utils.colorpalettes import DiscretePalette
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.widget import Msg, Input, Output

from orangecontrib.educational.widgets.utils.gradient_grid import \
    interpolate_grid
from orangecontrib.educational.widgets.utils.polynomialtransform \
    import PolynomialTransform
from orangecontrib.educational.widgets.utils.contour import Contour

GRID_SIZE = 60


class HoverEventDelegate(QObject):
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, obj, event):
        return isinstance(event, QGraphicsSceneMouseEvent) \
               and self.delegate(event)


class PolynomialPlot(OWScatterPlotBase):
    def __init__(self, scatter_widget, parent=None):
        super().__init__(scatter_widget, parent)
        self.alpha_value = 255

    def _get_discrete_colors(self, c_data, subset):
        pen, _ = super()._get_discrete_colors(c_data, subset)

        probs = self.master.get_probabilities()
        if probs is None:
            return pen, QBrush(self.palette.qcolors_w_nan[-1])

        probs = probs[:, 1]
        palette = self.master.get_model_palette().palette
        colors = np.hstack((palette[(probs > 0.5).astype(int)],
                            128 + 255 * np.abs(probs - 0.5)[:, None].astype(int)))
        brush = [QBrush(QColor(*[int(x) for x in col])) for col in colors]
        return pen, brush


class OWPolynomialClassification(OWBaseLearner):
    name = "Polynomial Classification"
    description = "Widget that demonstrates classification " \
                  "with polynomial expansion of variables."
    keywords = ["polynomial classification", "classification", "class",
                "classification visualization", "polynomial features"]
    icon = "icons/polynomialclassification.svg"
    want_main_area = True
    resizing_enabled = True
    priority = 600

    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, default=True)
        data = Output("Data", Table)

    LEARNER = LogisticRegressionLearner

    settingsHandler = DomainContextHandler(
        match_values=DomainContextHandler.MATCH_VALUES_CLASS)
    graph = SettingProvider(PolynomialPlot)

    learner_name = Setting("Polynomial Classification")
    attr_x = ContextSetting(None)
    attr_y = ContextSetting(None)
    target_class = Setting("")
    degree = Setting(1)
    contours_enabled = Setting(True)

    graph_name = 'graph.plot_widget'  # QGraphicsView (pg.PlotWidget)

    class Error(OWBaseLearner.Error):
        num_features = Msg("Data must contain at least two numeric variables.")
        no_class = Msg("Data must have a single target attribute.")
        no_nonnan_data = Msg("No points with defined values.")
        same_variable = Msg("Select two different variables.")

    def __init__(self, *args, **kwargs):
        # Some attributes must be created before super().__init__
        self._add_graph()
        self.var_model = DomainModel(valid_types=(ContinuousVariable, ))

        super().__init__(*args, **kwargs)
        self.input_learner = None
        self.data = None

        self.learner = None
        self.selected_data = None
        self.orig_class = None
        self.probabilities_grid = None
        self.xv = None
        self.yv = None
        self.contours = []
        self.init_learner()

    def add_main_layout(self):
        box = gui.widgetBox(self.controlArea, "Variables")
        gui.comboBox(
            box, self, "attr_x", model=self.var_model,
            callback=self._on_attr_changed)
        gui.comboBox(
            box, self, "attr_y", model=self.var_model,
            callback=self._on_attr_changed)
        gui.spin(
            box, self, value='degree', label='Polynomial expansion:',
            minv=1, maxv=5, step=1, alignment=Qt.AlignRight, controlWidth=70,
            callback=self._on_degree_changed)
        gui.comboBox(
            box, self, 'target_class', label='Target:',
            orientation=Qt.Horizontal, sendSelectedValue=True,
            callback=self._on_target_changed)

        box = gui.widgetBox(self.controlArea, box=True)
        gui.checkBox(
            box, self.graph, 'show_legend',"Show legend",
            callback=self.graph.update_legend_visibility)
        gui.checkBox(
            box, self, 'contours_enabled', label="Show contours",
            callback=self.plot_contour)

        gui.rubber(self.controlArea)

    def add_bottom_buttons(self):
        pass

    def _on_degree_changed(self):
        self.init_learner()
        self.apply()

    def _on_attr_changed(self):
        self.select_data()
        self.apply()

    def _on_target_changed(self):
        self.select_data()
        self.apply()

    ##############################
    # Input signal-related stuff
    @Inputs.learner
    def set_learner(self, learner):
        self.input_learner = learner
        self.init_learner()

    def set_preprocessor(self, preprocessor):
        self.preprocessors = [preprocessor] if preprocessor else []
        self.init_learner()

    @Inputs.data
    def set_data(self, data):
        combo = self.controls.target_class

        self.closeContext()
        self.Error.clear()
        combo.clear()
        self.var_model.set_domain(None)
        self.data = self.selected_data = self.orig_class = None
        self.xv = None
        self.yv = None
        self.probabilities_grid = None

        if not data:
            return
        domain = data.domain
        if domain.class_var is None or domain.class_var.is_continuous:
            self.Error.no_class()
            return
        if sum(var.is_continuous
                 for var in chain(domain.variables, domain.metas)) < 2:
            self.Error.num_features()
            return

        self.data = data
        non_empty = np.bincount(data.Y[np.isfinite(data.Y)].astype(int),
                                minlength=len(domain.class_var.values)) > 0
        values = np.array(domain.class_var.values)[non_empty]
        combo.addItems(values.tolist())
        self.var_model.set_domain(self.data.domain)
        self.attr_x, self.attr_y = self.var_model[:2]
        self.target_class = combo.itemText(0)

        hide_attrs = len(self.var_model) == 2
        self.controls.attr_x.setHidden(hide_attrs)
        self.controls.attr_y.setHidden(hide_attrs)

        self.openContext(self.data)
        self.select_data()

    def init_learner(self):
        self.learner = copy.copy(self.input_learner
                                 or self.LEARNER(penalty='l2', C=1e10))
        self.learner.preprocessors = (
            [PolynomialTransform(self.degree)] +
            list(self.preprocessors or []) +
            list(self.learner.preprocessors or []))
        self.send_learner()

    def handleNewSignals(self):
        self.apply()

    def select_data(self):
        """Put the two selected columns in a new Orange.data.Table"""
        self.Error.no_nonnan_data.clear()
        self.Error.same_variable.clear()

        attr_x, attr_y = self.attr_x, self.attr_y
        if self.attr_x is self.attr_y:
            self.selected_data = None
            self.orig_class = None
            self.Error.same_variable()
            return

        names = [var.name for var in (attr_x, attr_y)]
        if names == ["x", "y"]:
            names = [""] * 2
        for place, name in zip(("bottom", "left"), names):
            self.graph.plot_widget.getAxis(place).setLabel(name)
        old_class = self.data.domain.class_var
        values = old_class.values
        target_idx = values.index(self.target_class)

        binary = len(values) == 2
        new_class = DiscreteVariable(
            old_class.name + "'",
            values=(values[1 - target_idx] if binary else 'Others',
                    self.target_class),
            compute_value=Indicator(old_class, target_idx))
        new_class.palette = DiscretePalette(
            "indicator", "indicator",
            [list(old_class.palette.palette[1 - target_idx])
             if binary else [64, 64, 64],
             list(old_class.palette.palette[target_idx])])

        domain = Domain([attr_x, attr_y], new_class, [old_class])

        self.selected_data = self.data.transform(domain)
        valid_data = \
            np.flatnonzero(
                np.all(
                    np.isfinite(self.selected_data.X),
                    axis=1)
            )
        if not np.any(np.isfinite(self.data.Y[valid_data])):
            self.Error.no_nonnan_data()
            self.selected_data = None
            self.orig_class = None
        else:
            self.selected_data = self.selected_data[valid_data]
            self.orig_class = self.data.Y[valid_data]

    def apply(self):
        self.update_model()
        self.send_model()
        self.send_coefficients()
        self.send_data()

        self.graph.reset_graph()
        self.graph.plot_widget.addItem(self.contour_label)  # Re-add the label
        self.plot_gradient()
        self.plot_contour()

    def update_model(self):
        self.Error.fitting_failed.clear()
        self.model = None
        self.probabilities_grid = None
        if self.selected_data is not None and self.learner is not None:
            try:
                defined = self.selected_data[np.isfinite(self.selected_data.Y)]
                self.model = self.learner(defined)
                self.model.name = self.learner_name
            except Exception as e:
                self.Error.fitting_failed(str(e))

    ##############################
    # Graph and its contents
    def _add_graph(self):
        self.graph = PolynomialPlot(self)
        self.graph.plot_widget.setCursor(QCursor(Qt.CrossCursor))
        self.mainArea.layout().addWidget(self.graph.plot_widget)
        self.graph.point_width = 1

        axis_color = self.palette().color(QPalette.Text)
        axis_pen = QPen(axis_color)

        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))

        for pos in ("bottom", "left"):
            axis = self.graph.plot_widget.getAxis(pos)
            axis.setPen(axis_pen)
            axis.setTickFont(tickfont)
            axis.show()

        self._hover_delegate = HoverEventDelegate(self.help_event)
        self.graph.plot_widget.scene().installEventFilter(self._hover_delegate)

        self.contour_label = label = QGraphicsTextItem()
        label.setFlag(QGraphicsTextItem.ItemIgnoresTransformations)
        self.graph.plot_widget.addItem(label)
        label.hide()
        # I'm not proud of this and will brew a coffee to the person who
        # improves it (and comes to claim it)
        self.graph.plot_widget.scene().leaveEvent = lambda *_: label.hide()

    @staticmethod
    def _contour_pen(value, hovered):
        return pg.mkPen(0.2,
            width=1 + 2 * hovered + (value == 0.5),
            style=Qt.SolidLine)
            # Alternative:
            # Qt.SolidLine if hovered or value == 0.5 else Qt.DashDotLine

    def help_event(self, event):
        if self.probabilities_grid is None:
            return False

        pos = event.scenePos()
        pos = self.graph.plot_widget.mapToView(pos)

        # The mouse hover width is a bit larger for easier hovering, but
        # consequently more than one line can be hovered. Pick the middle one.
        hovereds = [item for item in self.contours
                    if item.mouseShape().contains(pos)]
        hovered = hovereds[len(hovereds) // 2] if hovereds else None
        # Set the pen for all lines, hovered and not hovered (any longer)
        for item in self.contours:
            item.setPen(self._contour_pen(item.value, item is hovered))

        # Set the probability for the textitem at mouse position.
        # If there is a hovered line - this acts as a line label.
        # Otherwise, take the probability from the grid
        label = self.contour_label
        if hovered:
            prob = hovered.value
        else:
            min_x, step_x, min_y, step_y = self.probabilities_grid_dimensions
            prob = self.probabilities_grid.T[
                int(np.clip(round((pos.x() - min_x) * step_x), 0, GRID_SIZE - 1)),
                int(np.clip(round((pos.y() - min_y) * step_y), 0, GRID_SIZE - 1))]
        prob_lab = f"{round(prob, 3):.3g}"
        if "." not in prob_lab:
            prob_lab += ".0"  # Showing just 0 or 1 looks ugly
        label.setHtml(prob_lab)
        font = label.font()
        font.setBold(hovered is not None)
        label.setFont(font)

        # Position the label above and left from mouse; looks nices
        rect = label.boundingRect()
        spos = event.scenePos()
        x, y = spos.x() - rect.width(), spos.y() - rect.height()
        label.setPos(self.graph.plot_widget.mapToView(QPointF(x, y)))

        label.show()
        return True

    def changeEvent(self, ev):
        # This hides the label if the user alt-tabs out of the window
        if ev.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.contour_label.hide()
        super().changeEvent(ev)

    def plot_gradient(self):
        if not self.model:
            self.probabilities_grid = None
            return

        (min_x, max_x), (min_y, max_y) = self.graph.view_box.viewRange()
        x = np.linspace(min_x, max_x, GRID_SIZE)
        y = np.linspace(min_y, max_y, GRID_SIZE)
        self.xv, self.yv = np.meshgrid(x, y)

        attr = np.hstack((self.xv.reshape((-1, 1)), self.yv.reshape((-1, 1))))
        nil = np.full((len(attr), 1), np.nan)
        attr_data = Table.from_numpy(self.selected_data.domain, attr, nil, nil)

        self.probabilities_grid = self.model(attr_data, 1)[:, 1] \
            .reshape(self.xv.shape)
        self.probabilities_grid_dimensions = (
            min_x, GRID_SIZE / (max_x - min_x),
            min_y, GRID_SIZE / (max_y - min_y))

        if not self._treelike:
            self.probabilities_grid = self.blur_grid(self.probabilities_grid)

        bitmap = interpolate_grid(self.probabilities_grid, 256)
        bitmap *= 255
        bitmap = bitmap.astype(np.uint8)

        class_var = self.selected_data.domain.class_var
        h1, s1, v1 = rgb_to_hsv(*class_var.colors[1] / 255)
        palette = np.vstack((
            np.linspace([h1, 0, 0.8], [h1, 0, 1], 128),
            np.linspace([h1, 0, 1], [h1, s1 * 0.5, 0.7 + 0.3 * v1], 128)
        ))
        palette = 255 * np.array([hsv_to_rgb(*col) for col in palette])
        palette = palette.astype(int)

        density_img = pg.ImageItem(bitmap.T, lut=palette)

        density_img.setRect(QRectF(min_x, min_y,
                                   max_x - min_x, max_y - min_y))
        density_img.setZValue(-1)
        self.graph.plot_widget.addItem(density_img, ignoreBounds=True)

    def remove_contours(self):
        while self.contours:
            self.graph.plot_widget.removeItem(self.contours.pop())

    @property
    def _treelike(self):
        return isinstance(self.learner, (RandomForestLearner, TreeLearner))

    def plot_contour(self):
        self.remove_contours()
        if self.probabilities_grid is None or not self.contours_enabled:
            return

        contour = Contour(self.xv, self.yv, self.probabilities_grid)
        contour_lines = contour.contours(np.arange(0.1, 1, 0.1))
        for key, value in contour_lines.items():
            for line in value:
                if len(line) > self.degree and not self._treelike:
                    tck, u = splprep(
                        [list(x) for x in zip(*reversed(line))],
                        s=0.001, k=self.degree,
                        per=(len(line) if np.allclose(line[0], line[-1])
                             else 0))
                    new_int = np.arange(0, 1.01, 0.01)
                    interpol_line = np.array(splev(new_int, tck)).T.tolist()
                else:
                    interpol_line = line
                contour = pg.PlotCurveItem(
                    *np.array(list(interpol_line)).T,
                    pen=self._contour_pen(key, False))
                # The hover region can be narrowed by calling setClickable
                # (with False, to keep it unclickable)
                contour.value = key
                self.graph.plot_widget.addItem(contour)
                self.contours.append(contour)

    @staticmethod
    def blur_grid(grid):
        filtered = gaussian_filter(grid, sigma=1)
        filtered[(grid > 0.45) & (grid < 0.55)] = grid[(grid > 0.45) &
                                                       (grid < 0.55)]
        return filtered

    # The following methods are required by OWScatterPlotBase
    def get_coordinates_data(self):
        if not self.selected_data:
            return None, None
        return self.selected_data.X.T

    def get_color_data(self):
        return self.orig_class

    def get_palette(self):
        return self.data.domain.class_var.palette

    def get_color_labels(self):
        return self.selected_data and self.data.domain.class_var.values

    def is_continuous_color(self):
        return False

    def get_model_palette(self):
        return self.model and self.selected_data.domain.class_var.palette

    def get_probabilities(self):
        return self.model and self.model(self.selected_data, 1)

    get_size_data = get_shape_data = get_shape_labels = \
        get_subset_mask = get_label_data = get_tooltip = selection_changed = \
        lambda *_: None

    ##############################
    # Output signal-related stuff
    def send_learner(self):
        if self.learner is not None:
            self.learner.name = self.learner_name
        self.Outputs.learner.send(self.learner)

    def send_model(self):
        self.Outputs.model.send(self.model)

    def send_coefficients(self):
        if (self.model is None
                or not isinstance(self.learner, LogisticRegressionLearner)
                or not hasattr(self.model, 'skl_model')):
            self.Outputs.coefficients.send(None)
            return

        model = self.model.skl_model
        domain = Domain(
            [ContinuousVariable("coef")], metas=[StringVariable("name")])
        coefficients = model.intercept_.tolist() + model.coef_[0].tolist()
        names = ["Intercept"] + [x.name for x in self.model.domain.attributes]
        coefficients_table = Table.from_list(
            domain, list(zip(coefficients, names)))
        self.Outputs.coefficients.send(coefficients_table)

    def send_data(self):
        if self.selected_data is None:
            self.Outputs.data.send(None)
        else:
            expanded = PolynomialTransform(self.degree)(self.selected_data)
            self.Outputs.data.send(expanded)

    def send_report(self):
        if self.selected_data is None:
            return
        name = "" if self.degree == 1 \
            else f"Model with polynomial expansion {self.degree}"
        self.report_plot(name=name)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPolynomialClassification).run(Table.from_file('iris'))
