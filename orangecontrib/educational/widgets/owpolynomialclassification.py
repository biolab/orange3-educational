import copy
from colorsys import rgb_to_hsv, hsv_to_rgb
from itertools import chain

import pyqtgraph as pg
from AnyQt.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsTextItem
from AnyQt.QtGui import QPalette, QPen, QFont

from Orange.preprocess.transformation import Indicator

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage.filters import gaussian_filter

from AnyQt.QtCore import Qt, QRectF, QObject, QPointF

from Orange.base import Learner
from Orange.data import ContinuousVariable, Table, Domain, StringVariable, \
    DiscreteVariable
from Orange.widgets import settings, gui
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.classification import (
    LogisticRegressionLearner,
    RandomForestLearner,
    TreeLearner
)
from Orange.widgets.visualize.owscatterplotgraph import OWScatterPlotBase
from Orange.widgets.widget import Msg, Input, Output

from orangecontrib.educational.widgets.utils.polynomialtransform \
    import PolynomialTransform
from orangecontrib.educational.widgets.utils.contour import Contour


# TODO: Disable zoom

class Scatterplot(OWScatterPlotBase):
    pass


class HoverEventDelegate(QObject):
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, obj, event):
        if isinstance(event, QGraphicsSceneMouseEvent):
            return self.delegate(event)
        return False


class OWPolynomialClassification(OWBaseLearner):
    name = "Polynomial Classification"
    description = "Widget that demonstrates classification in two classes " \
                  "with polynomial expansion of attributes."
    keywords = ["polynomial classification", "classification", "class",
                "classification visualization", "polynomial features"]
    icon = "icons/polynomialclassification.svg"
    want_main_area = True
    resizing_enabled = True
    priority = 600

    # inputs and outputs
    class Inputs(OWBaseLearner.Inputs):
        learner = Input("Learner", Learner)

    class Outputs(OWBaseLearner.Outputs):
        coefficients = Output("Coefficients", Table, default=True)
        data = Output("Data", Table)

    # data attributes
    data = None
    selected_data = None
    probabilities_grid = None
    xv = None
    yv = None

    # learners
    LEARNER = LogisticRegressionLearner
    learner_other = None
    default_preprocessor = PolynomialTransform

    learner_name = settings.Setting("Polynomial Classification")

    # widget properties
    attr_x = settings.Setting(None)
    attr_y = settings.Setting(None)
    target_class = settings.Setting("")
    degree = settings.Setting(1)
    legend_enabled = settings.Setting(True)
    contours_enabled = settings.Setting(False)
    contour_step = settings.Setting(0.1)

    graph_name = 'graph'

    # settings
    grid_size = 60
    contour_color = "#1f1f1f"

    # layout elements
    options_box = None
    cbx = None
    cby = None
    degree_spin = None
    plot_properties_box = None
    contours_enabled_checkbox = None
    legend_enabled_checkbox = None
    contour_step_slider = None
    target_class_combo = None

    class Error(OWBaseLearner.Error):
        num_features = Msg("Data must contain at least two numeric variables.")
        no_class = Msg("Data must have a single target attribute.")
        no_class_values = Msg("Target must have at least two different values.")
        no_nonnan_data = Msg("No points with defined values.")
        no_classifier = Msg("Learning algorithm must be a classifier, not regressor.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learner = None
        self.other_learner = None
        self.init_learner()
        self.contours = []

    def add_main_layout(self):
        self.options_box = gui.widgetBox(self.controlArea, "Options")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal)
        self.var_model = DomainModel(valid_types=(ContinuousVariable, ))
        for value in ("attr_x", "attr_y"):
            gui.comboBox(
                value=value, label=f'{value[-1].upper()}: ',
                callback=self.apply, model=self.var_model, **opts)
        self.target_class_combo = gui.comboBox(
            value='target_class', label='Target: ', callback=self.apply,
            sendSelectedValue=True, **opts)
        self.degree_spin = gui.spin(
            value='degree', label='Polynomial expansion:',
            minv=1, maxv=5, step=1, callback=self.on_degree_changed,
            alignment=Qt.AlignRight, controlWidth=70, **opts)

        # plot properties box
        self.plot_properties_box = gui.widgetBox(
            self.controlArea, "Plot Properties")
        self.legend_enabled_checkbox = gui.checkBox(
            self.plot_properties_box, self, 'legend_enabled',
            label="Show legend", callback=self.replot)
        self.contours_enabled_checkbox = gui.checkBox(
            self.plot_properties_box, self, 'contours_enabled',
            label="Show contours", callback=self.plot_contour)
        self.contour_step_slider = gui.spin(
            self.plot_properties_box, self, 'contour_step',
            minv=0.10, maxv=0.50, step=0.05, callback=self.plot_contour,
            label='Contour step:', decimals=2, spinType=float,
            alignment=Qt.AlignRight, controlWidth=70)

        gui.rubber(self.controlArea)

        self.graph = Scatterplot(self)
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

        self.mainArea.layout().addWidget(self.graph.plot_widget)

        self._tooltip_delegate = HoverEventDelegate(self.help_event)
        self.graph.plot_widget.scene().installEventFilter(self._tooltip_delegate)

        self.contour_label = None

    @staticmethod
    def _contour_pen(value, hovered):
        return pg.mkPen(0.2,
            width=1 + hovered + (value == 0.5),
            style=Qt.SolidLine)
            # Alternative:
            # Qt.SolidLine if hovered or value == 0.5 else Qt.DashDotLine

    def help_event(self, event):
        pos = event.scenePos()
        pos = self.graph.plot_widget.mapToView(pos)

        # The mouse hover width is a bit larger for easier hovering, but
        # consequently more than one line can be hovered. Pick the middle one.
        hovereds = [item for item in self.contours
                    if item.mouseShape().contains(pos)]
        hovered = hovereds[len(hovereds) // 2] if hovereds else None

        # Remove previous label if exists and no longer relevant
        label = self.contour_label
        if label and label.labelled is not hovered:
            self.graph.plot_widget.removeItem(label)
            self.contour_label = label = None

        # Add label if necessary. Put it to the left and above the mouse cursor
        if hovered and not label:
            label = QGraphicsTextItem()
            label.setFlag(label.ItemIgnoresTransformations)
            label.setHtml(f"{hovered.value:.1f}")
            rect = label.boundingRect()
            spos = event.scenePos()
            x, y = spos.x() - rect.width(), spos.y() - rect.height()
            label.setPos(self.graph.plot_widget.mapToView(QPointF(x, y)))
            label.labelled = hovered
            self.contour_label = label
            self.graph.plot_widget.addItem(label)

        # Set the pen for all lines
        for item in self.contours:
            item.setPen(self._contour_pen(item.value, item is hovered))
        return bool(hovered)

    @Inputs.learner
    def set_learner(self, learner):
        self.other_learner = learner
        self.init_learner()

    @Inputs.data
    def set_data(self, data):
        def init_class_combo():
            non_empty = np.bincount(data.Y[np.isfinite(data.Y)].astype(int)) > 0
            if not np.any(non_empty):
                return False
            values = np.array(domain.class_var.values)[non_empty]
            self.target_class_combo.addItems(values.tolist())
            return True

        self.Error.clear()
        self.data = None
        self.target_class_combo.clear()
        self.var_model.set_domain(None)

        self.xv = None
        self.yv = None
        self.probabilities_grid = None

        if not data:
            self.reset_graph()
            return
        domain = data.domain
        if domain.class_var is None or domain.class_var.is_continuous:
            self.Error.no_class()
        elif sum(var.is_continuous
                 for var in chain(domain.variables, domain.metas)) < 2:
            self.Error.num_features()
        elif not init_class_combo():
            self.Error.no_class_values()
        else:
            self.data = data
            self.var_model.set_domain(self.data.domain)
            self.attr_x, self.attr_y = self.var_model[:2]
            self.target_class = self.target_class_combo.itemText(0)
        self.graph.reset_graph()

    def init_learner(self):
        self.learner = copy.deepcopy(self.other_learner
                                     or self.LEARNER(penalty='l2', C=1e10))
        self.learner.preprocessors = (
            [self.default_preprocessor(self.degree)] +
            list(self.preprocessors or []) +
            list(self.learner.preprocessors or []))
        if self.data: # TODO: This is here because of the call from __init__. Improve the logic.
            self.apply()

    def handleNewSignals(self):
        self.apply()

    def on_degree_changed(self):
        self.init_learner()
        self.apply()

    def _attr_columns(self):
        return tuple(self.data.get_column_view(attr)[0]
                     for attr in (self.attr_x, self.attr_y))

    def get_coordinates_data(self):
        if not self.data:
            return None, None
        return tuple(c[self.valid_data] for c in self._attr_columns())

    def get_color_data(self):
        return self.data.Y[self.valid_data]

    def get_palette(self):
        return self.data.domain.class_var.palette

    def get_color_labels(self):
        return self.data.domain.class_var.values

    get_size_data = get_shape_data = get_shape_labels = \
        get_subset_mask = get_label_data = get_tooltip = selection_changed = \
        lambda *_: None

    def is_continuous_color(self):
        return False

    def replot(self):
        if self.data is None or self.selected_data is None:
            self.set_empty_plot()
            return

        self.plot_gradient()
        self.plot_contour()

    def plot_gradient(self):
        gsize = self.grid_size
        (min_x, max_x), (min_y, max_y) = self.graph.view_box.viewRange()
        x = np.linspace(min_x, max_x, gsize)
        y = np.linspace(min_y, max_y, gsize)
        self.xv, self.yv = np.meshgrid(x, y)

        attr = np.hstack((self.xv.reshape((-1, 1)), self.yv.reshape((-1, 1))))
        nil = np.full((len(attr), 1), np.nan)
        attr_data = Table.from_numpy(self.selected_data.domain, attr, nil, nil)

        self.probabilities_grid = self.model(attr_data, 1)[:, 1]\
            .reshape(self.xv.shape)

        if not isinstance(self.learner, (RandomForestLearner, TreeLearner)):
            self.probabilities_grid = self.blur_grid(self.probabilities_grid)

        bitmap = self.probabilities_grid.copy()
        bitmap *= 255
        bitmap = bitmap.astype(np.uint8)

        class_var = self.data.domain.class_var
        target_idx = class_var.values.index(self.target_class)
        h1, s1, v1 = rgb_to_hsv(*class_var.colors[target_idx] / 255)
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

    def plot_contour(self):
        """
        Function constructs contour lines
        """
        self.remove_contours()
        if not self.data:
            return
        if self.contours_enabled:
            contour = Contour(self.xv, self.yv, self.probabilities_grid)
            contour_lines = contour.contours(
                np.hstack(
                    (np.arange(0.5, 0, - self.contour_step)[::-1],
                     np.arange(0.5 + self.contour_step, 1, self.contour_step))))
            # we want to have contour for 0.5

            is_blurred = not isinstance(self.learner,
                                        (RandomForestLearner, TreeLearner))
            for key, value in contour_lines.items():
                for line in value:
                    if len(line) > self.degree and is_blurred:
                        tck, u = splprep(
                            [list(x) for x in zip(*reversed(line))],
                            s=0.001, k=self.degree,
                            per=(len(line)
                                 if np.allclose(line[0], line[-1])
                                 else 0))
                        new_int = np.arange(0, 1.01, 0.01)
                        interpol_line = np.array(splev(new_int, tck)).T.tolist()
                    else:
                        interpol_line = line
                    contour = pg.PlotCurveItem(
                        *np.array(list(interpol_line)).T,
                        pen=self._contour_pen(key, False))
                    # If you want to narrow the hover region, uncomment the
                    # following line. The counter will remain unclickable, but
                    # the call sets the mouse width for hover
                    #contour.setClickable(False, 3)
                    contour.value = key
                    self.graph.plot_widget.addItem(contour)
                    self.contours.append(contour)

    @staticmethod
    def blur_grid(grid):
        filtered = gaussian_filter(grid, sigma=1)
        filtered[(grid > 0.45) & (grid < 0.55)] = grid[(grid > 0.45) &
                                                       (grid < 0.55)]
        return filtered

    def select_data(self):
        """
        Function takes two selected columns from data table and merge them
        in new Orange.data.Table
        """
        self.Error.clear()

        attr_x = self.data.domain[self.attr_x]
        attr_y = self.data.domain[self.attr_y]
        for place, attr in (("bottom", attr_x), ("left", attr_y)):
            self.graph.plot_widget.getAxis(place).setLabel(attr.name)
        old_class = self.data.domain.class_var
        values = old_class.values
        target_idx = values.index(self.target_class)

        new_class = DiscreteVariable(
            old_class.name + "'",
            values=(values[1 - target_idx] if len(values) == 2 else 'Others',
                    self.target_class),
            compute_value=Indicator(old_class, target_idx))

        domain = Domain([attr_x, attr_y], new_class, [old_class])
        new_data = self.data.transform(domain)
        if np.isnan(new_data.X).all(axis=0).any():
            self.Error.all_none_data()
            return None
        return new_data


    def apply(self):
        """
        Applies leaner and sends new model and coefficients
        """
        self.valid_data = \
            np.flatnonzero(
                np.all(
                    np.isfinite(
                        np.vstack(self._attr_columns())
                    ),
                    axis=0)
            )
        self.graph.reset_graph()
        self.update_model()
        self.plot_gradient()
        self.plot_contour()
        self.send_learner()
        self.send_coefficients()
        self.send_data()

    def send_learner(self):
        """
        Function sends learner on widget's output
        """
        if self.learner is not None:
            self.learner.name = self.learner_name
        self.Outputs.learner.send(self.learner)

    def update_model(self):
        """
        Function sends model on widget's output
        """
        self.Error.fitting_failed.clear()
        self.model = None
        if self.data is not None and self.learner is not None:
            self.selected_data = self.select_data()
            if self.selected_data is not None:
                try:
                    self.model = self.learner(self.selected_data)
                    self.model.name = self.learner_name
                    self.model.instances = self.selected_data
                except Exception as e:
                    self.Error.fitting_failed(str(e))

        self.Outputs.model.send(self.model)

    def send_coefficients(self):
        """
        Function sends coefficients on widget's output if model has them
        """

        if (self.model is not None and
                isinstance(self.learner, LogisticRegressionLearner) and
                hasattr(self.model, 'skl_model')):
            model = self.model.skl_model
            domain = Domain(
                [ContinuousVariable("coef")], metas=[StringVariable("name")])
            coefficients = (model.intercept_.tolist() +
                            model.coef_[0].tolist())

            data = self.model.instances
            for preprocessor in self.learner.preprocessors:
                data = preprocessor(data)
            names = ["Intercept"] + [x.name for x in data.domain.attributes]

            coefficients_table = Table.from_list(
                domain, list(zip(coefficients, names)))
            self.Outputs.coefficients.send(coefficients_table)
        else:
            self.Outputs.coefficients.send(None)

    def send_data(self):
        """
        Function sends data on widget's output
        """
        if self.data is not None:
            data = self.selected_data
            self.Outputs.data.send(data)
            return
        self.Outputs.data.send(None)

    def add_bottom_buttons(self):
        pass

    def send_report(self):
        if self.data is None:
            return
        name = "" if self.degree == 1 \
            else f"Model with polynomial expansion {self.degree}"
        self.report_plot(name=name, plot=self.graph.plot_widget)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWPolynomialClassification).run(Table.from_file('iris'))
