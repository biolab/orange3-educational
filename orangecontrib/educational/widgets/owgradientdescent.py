import time
from colorsys import rgb_to_hsv, hsv_to_rgb

import numpy as np
from scipy.interpolate import splprep, splev

from AnyQt.QtCore import Qt, QObject, pyqtSignal, QThread, QEvent, QRectF
from AnyQt.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsTextItem
from AnyQt.QtGui import QCursor, QColor

import pyqtgraph as pg

from Orange.classification import Model
from Orange.data import \
    ContinuousVariable, DiscreteVariable, StringVariable, Domain, Table
from Orange.preprocess.transformation import Indicator
from Orange.widgets import gui
from Orange.widgets import settings
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.preprocess.preprocess import Normalize
from orangewidget.report import report
from orangewidget.utils.widgetpreview import WidgetPreview

from orangecontrib.educational.widgets.utils.gradient_grid import \
    interpolate_grid
from orangecontrib.educational.widgets.utils.linear_regression import \
    LinearRegression
from orangecontrib.educational.widgets.utils.logistic_regression \
    import LogisticRegression
from orangecontrib.educational.widgets.utils.contour import Contour

GRID_SIZE = 20


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
            time.sleep(0.2)
        self.ow_gradient_descent.stop_auto_play_trigger.emit()


class HoverEventDelegate(QObject):
    def __init__(self, delegate, parent=None):
        super().__init__(parent)
        self.delegate = delegate

    def eventFilter(self, _, event):
        return isinstance(event, QGraphicsSceneMouseEvent) \
               and self.delegate(event)


class OWGradientDescent(OWWidget):
    name = "Gradient Descent"
    description = "Demonstration of gradient descent " \
                  "in logistic or linear regression"
    keywords = ["gradient descent", "optimization", "gradient"]
    icon = "icons/GradientDescent.svg"
    want_main_area = True
    priority = 400

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        model = Output("Model", Model)
        coefficients = Output("Coefficients", Table)

    graph_name = "graph"  # QGraphicsView (pg.PlotWidget)

    settingsHandler = settings.DomainContextHandler(
        match_values=settings.DomainContextHandler.MATCH_VALUES_CLASS)
    attr_x = settings.ContextSetting(None)
    attr_y = settings.ContextSetting(None)
    target_class = settings.Setting('')
    alpha = settings.Setting(0.4)
    step_size = settings.Setting(30)  # step size for stochastic gds
    auto_play_speed = settings.Setting(1)
    stochastic = settings.Setting(False)

    default_background_color = [0, 0xbf, 0xff]
    auto_play_button_text = ["Run", "Stop"]

    step_trigger = pyqtSignal()
    stop_auto_play_trigger = pyqtSignal()

    class Error(OWWidget.Error):
        num_features = Msg("Data must contain at least {}.")
        no_class = Msg("Data must have a single target variable.")
        no_class_values = Msg("Target variable must have at least two values.")
        no_nonnan_data = Msg("No points with defined values.")
        same_variable = Msg("Select two different variables.")

    def __init__(self):
        super().__init__()
        self.learner = None
        self.data = None
        self.selected_data = None
        self.cost_grid = None
        self.min_x = self.max_x = self.min_y = self.max_y = None
        self.contours = []
        self.optimization_path = self.last_point = None
        self.hover_label = self.optimization_label = None

        self.auto_play_enabled = False
        self.auto_play_button_text = ["Run", "Stop"]
        self.auto_play_thread = None

        self.var_model = DomainModel(valid_types=ContinuousVariable,
                                     order=DomainModel.ATTRIBUTES)

        self.options_box = gui.widgetBox(self.controlArea, "Variables")
        opts = dict(
            widget=self.options_box, master=self, orientation=Qt.Horizontal,
            callback=self.change_attributes)
        gui.comboBox(value='attr_x', model=self.var_model, **opts)
        gui.comboBox(value='attr_y', model=self.var_model, **opts)
        gui.comboBox(value='target_class', label='Target: ',
                     sendSelectedValue=True, **opts)

        self.properties_box = gui.widgetBox(self.controlArea, "Properties")
        self.alpha_spin = gui.spin(
            widget=self.properties_box, master=self, callback=self.change_alpha,
            value="alpha", label="Learning rate: ",
            minv=0.001, maxv=100, step=0.001, spinType=float, decimals=3,
            alignment=Qt.AlignRight, controlWidth=80)
        self.stochastic_checkbox = gui.checkBox(
            widget=self.properties_box, master=self,
            callback=self.change_stochastic, value="stochastic",
            label="Stochastic descent")
        self.step_size_spin = gui.spin(
            widget=self.properties_box, master=self, callback=self.change_step,
            value="step_size", label="Step size: ",
            minv=1, maxv=100, step=1, alignment=Qt.AlignRight, controlWidth=80)

        self.step_box = gui.widgetBox(self.controlArea, box=True,
                                      margin=0, spacing=0)
        hbox = gui.hBox(self.step_box)
        self.step_button = gui.button(
            hbox, self, callback=self.step, label="Step", default=True)
        self.step_back_button = gui.button(
            hbox, self, callback=self.step_back, label="Back")
        self.restart_button = gui.button(
            widget=self.step_box, master=self,
            callback=self.restart, label="Restart")

        gui.separator(self.step_box)
        self.auto_play_button = gui.button(
            widget=self.step_box, master=self,
            label=self.auto_play_button_text[0], callback=self.auto_play)

        gui.rubber(self.controlArea)

        self._add_graph()

        self.step_size_lock()
        self.step_back_button_lock()

    def _add_graph(self):
        self.graph = pg.PlotWidget(background="w", autoRange=False)
        self.graph.setCursor(QCursor(Qt.CrossCursor))
        self._hover_delegate = HoverEventDelegate(self.help_event)
        self.graph.scene().installEventFilter(self._hover_delegate)
        self.mainArea.layout().addWidget(self.graph)

        self.hover_label = label = QGraphicsTextItem()
        label.setFlag(QGraphicsTextItem.ItemIgnoresTransformations)
        self.graph.addItem(label)
        label.setZValue(10)
        label.hide()
        # I'm not proud of this and will brew a coffee to the person who
        # improves it (and comes to claim it). Same in Polynomial Classification
        self.graph.scene().leaveEvent = lambda *_: self.hover_label.hide()

        self.optimization_label = label = QGraphicsTextItem()
        label.setFlag(QGraphicsTextItem.ItemIgnoresTransformations)
        self.graph.addItem(label)
        label.setZValue(10)

    ##############################
    # User-interaction

    def change_attributes(self):
        self.set_axis_titles()
        self.learner = None  # that theta does not same equal
        self.restart()

    def change_alpha(self):
        if self.learner is not None:
            self.learner.set_alpha(self.alpha)

    def change_stochastic(self):
        if self.learner is not None:
            self.learner.stochastic = self.stochastic
        self.step_size_lock()

    def change_step(self):
        if self.learner is not None:
            self.learner.stochastic_step_size = self.step_size

    def change_theta(self, x, y):
        if self.learner is None:
            return
        self.learner.set_theta([x, y])
        self.update_history()

    def step(self):
        if self.data is None:
            return
        if self.learner.step_no > 500:  # limit step no to avoid freezes
            return
        self.learner.step()
        self.update_history()
        self.send_output()
        self.step_back_button_lock()

    def step_back(self):
        if self.data is None:
            return
        if self.learner.step_no > 0:
            self.learner.step_back()
            self.update_history()
            self.send_output()
        self.step_back_button_lock()

    def restart(self):
        self.clear_plot()
        self.select_columns()
        self.set_axis_titles()
        if self.selected_data is None:
            return

        theta = self.learner.history[0][0] if self.learner is not None else None
        selected_learner = (LogisticRegression if self.is_classification
                            else LinearRegression)
        self.learner = selected_learner(
            data=self.selected_data,
            alpha=self.alpha, stochastic=self.stochastic,
            theta=theta, step_size=self.step_size,
            intercept=not self.is_classification)
        self.replot()
        if theta is None:  # no previous theta exist
            self.change_theta(np.random.uniform(self.min_x, self.max_x),
                              np.random.uniform(self.min_y, self.max_y))
        else:
            self.change_theta(theta[0], theta[1])
        self.send_output()
        self.step_back_button_lock()

    def auto_play(self):
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
        self.auto_play_enabled = False
        self.disable_controls(self.auto_play_enabled)
        self.auto_play_button.setText(
            self.auto_play_button_text[self.auto_play_enabled])

    def step_back_button_lock(self):
        self.step_back_button.setDisabled(
            self.learner is None
            or self.learner.step_no == 0
            or self.auto_play_enabled)

    def step_size_lock(self):
        self.step_size_spin.setDisabled(not self.stochastic)

    def disable_controls(self, disabled):
        for item in [self.step_button, self.step_back_button,
                     self.restart_button, self.properties_box, self.options_box]:
            item.setDisabled(disabled)

    key_actions = {(Qt.NoModifier, Qt.Key_Space): step}  # space button for step

    def keyPressEvent(self, e):
        """Bind 'back' key to step back"""
        if (e.modifiers(), e.key()) in self.key_actions:
            fun = self.key_actions[(e.modifiers(), e.key())]
            fun(self)
        else:
            super().keyPressEvent(e)

    ##############################
    # Signals and data-handling

    @Inputs.data
    def set_data(self, data):
        target_combo = self.controls.target_class

        self.closeContext()
        self.Error.clear()

        self.cost_grid = None
        self.learner = None
        self.selected_data = None
        self.data = None
        target_combo.clear()
        self.clear_plot()

        if data:
            self.var_model.set_domain(data.domain)
            if data.domain.class_var is None:
                data = None
                self.Error.no_class()
            # don't use self.is_classification -- widget has no self.data yet
            elif data.domain.class_var.is_discrete:
                if len(data.domain.class_var.values) < 2:
                    self.Error.no_class_values()
                    data = None
                elif len(self.var_model) < 2:
                    self.Error.num_features("two numeric variables")
                    data = None
            elif len(self.var_model) < 1:
                self.Error.num_features("one numeric variable")
                data = None

        if not data:
            self.var_model.set_domain(None)
            self.send_output()
            return

        self.data = data
        self.controls.attr_y.setHidden(not self.is_classification)
        target_combo.box.setHidden(not self.is_classification)
        self.attr_x = self.var_model[0]
        if self.is_classification:
            self.attr_y = self.var_model[min(1, len(self.var_model))]
            target_combo.clear()
            values = data.domain.class_var.values
            target_combo.addItems(values)
            # For binary class, take value with index 1 as a target, so the
            # output class is the same as the input
            self.target_class = values[1 if len(values) == 2 else 0]
        self.step_size_spin.setMaximum(len(data))
        self.openContext(self.data)
        self.set_axis_titles()
        self.restart()

    @property
    def is_classification(self):
        return self.data and self.data.domain.class_var.is_discrete

    def select_columns(self):
        self.Error.no_nonnan_data.clear()
        self.Error.same_variable.clear()
        self.selected_data = None
        if self.data is None:
            return

        old_class = self.data.domain.class_var
        if self.is_classification:
            if self.attr_x is self.attr_y:
                self.Error.same_variable()
                return
            values = old_class.values
            target_idx = values.index(self.target_class)
            new_class = DiscreteVariable(
                old_class.name + "'",
                values=(values[1 - target_idx] if len(values) == 2 else 'Others',
                        self.target_class),
                compute_value=Indicator(old_class, target_idx))
            domain = Domain([self.attr_x, self.attr_y], new_class, [old_class])
        else:
            domain = Domain([self.attr_x], old_class)

        data = self.data.transform(domain)
        valid_data = \
            np.flatnonzero(
                np.all(
                    np.isfinite(data.X),
                    axis=1)
            )

        if not valid_data.size:
            self.Error.no_nonnan_data()
            return

        data = data[valid_data]
        self.selected_data = Normalize(
            transform_class=not self.is_classification)(
            data)

    def send_output(self):
        self.send_model()
        self.send_coefficients()

    def send_model(self):
        if self.learner is not None and self.learner.theta is not None:
            self.Outputs.model.send(self.learner.model)
        else:
            self.Outputs.model.send(None)

    def send_coefficients(self):
        if self.learner is None or self.learner.theta is None:
            self.Outputs.coefficients.send(None)
            return

        domain = Domain(
            [ContinuousVariable("Coefficient")],
            metas=[StringVariable("Variable")])
        if self.is_classification:
            names = [self.attr_x.name, self.attr_y.name]
        else:
            names = ["intercept", self.attr_x.name]

        coefficients_table = Table.from_list(
            domain, list(zip(list(self.learner.theta), names)))
        self.Outputs.coefficients.send(coefficients_table)

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
        self.report_plot()
        self.report_caption(caption)

    ##############################
    # Plot-related methods

    def clear_plot(self):
        self.graph.clear()
        self.graph.addItem(self.hover_label)
        self.graph.addItem(self.optimization_label)
        self.optimization_label.hide()

    def replot(self):
        if self.data is None or self.selected_data is None:
            self.clear_plot()
            return

        optimal_theta = self.learner.optimized()
        self.min_x = optimal_theta[0] - 10
        self.max_x = optimal_theta[0] + 10
        self.min_y = optimal_theta[1] - 10
        self.max_y = optimal_theta[1] + 10
        self.graph.setRange(xRange=(self.min_x, self.max_x),
                            yRange=(self.min_y, self.max_y))

        x = np.linspace(self.min_x, self.max_x, GRID_SIZE)
        y = np.linspace(self.min_y, self.max_y, GRID_SIZE)
        xv, yv = np.meshgrid(x, y)
        thetas = np.column_stack((xv.flatten(), yv.flatten()))
        cost_values = self.learner.j(thetas)
        self.cost_grid = cost_values.reshape(xv.shape)

        self.plot_gradient()
        self.plot_contour(xv, yv)
        self.add_paths()

    def plot_gradient(self):
        if self.cost_grid is None:
            return

        cg = interpolate_grid(self.cost_grid, 256)
        bitmap = cg * (255 / np.max(cg))  # make a copy
        bitmap = bitmap.astype(np.uint8)

        h, s, v = rgb_to_hsv(*np.array(self.default_background_color) / 255)
        palette = np.linspace([h, 0, v], [h, s, 1], 255)
        palette = 255 * np.array([hsv_to_rgb(*col) for col in palette])
        palette = palette.astype(int)

        density_img = pg.ImageItem(bitmap.T, lut=palette)
        density_img.setRect(
            QRectF(self.min_x, self.min_y,
                   self.max_x - self.min_x, self.max_y - self.min_y))
        density_img.setZValue(-1)
        self.graph.addItem(density_img, ignoreBounds=True)

    def plot_contour(self, xv, yv):
        while self.contours:
            self.graph.removeItem(self.contours.pop())

        if self.cost_grid is None:
            return

        contour = Contour(xv, yv, self.cost_grid)
        contour_lines = contour.contours(
            np.linspace(np.min(self.cost_grid), np.max(self.cost_grid), 20))
        for key, value in contour_lines.items():
            for line in value:
                if len(line) > 3:
                    tck, u = splprep(np.array(line).T, s=0.0, per=0)
                    u_new = np.linspace(u.min(), u.max(), 100)
                    x_new, y_new = splev(u_new, tck, der=0)
                    interpol_line = np.c_[x_new, y_new]
                else:
                    interpol_line = line

                contour = pg.PlotCurveItem(
                    *np.array(list(interpol_line)).T,
                    pen=self._contour_pen(False),
                    antialias=True
                )
                contour.value = key
                self.graph.addItem(contour)
                self.contours.append(contour)

    def add_paths(self):
        yellow = QColor(255, 165, 0)
        dark = yellow.darker()
        self.optimization_path = pg.PlotDataItem(
            [], [], pen=pg.mkPen(dark, width=2),
            symbolPen=pg.mkPen(yellow), symbolBrush=pg.mkBrush(dark),
            symbol="o", symbolSize=4)
        self.last_point = pg.PlotDataItem(
            [], [],
            symbolPen=pg.mkPen(yellow), symbolBrush=pg.mkBrush(yellow),
            symbol="o", symbolSize=7)
        self.graph.addItem(self.optimization_path)
        self.graph.addItem(self.last_point)

    @staticmethod
    def _contour_pen(is_hovered=False):
        return pg.mkPen(0.2, width=1 + 2 * is_hovered)

    def set_axis_titles(self):
        if self.selected_data is None:
            names = ["", ""]
        elif self.is_classification:
            names = [f"Θ({self.attr_x.name})", f"Θ({self.attr_y.name})"]
        else:
            names = ["Θ₀", "Θ₁"]
        for axname, name in zip(("bottom", "left"), names):
            axis = self.graph.getAxis(axname)
            axis.setLabel(name)

    def update_history(self):
        steps = self.learner.step_no
        x, y = zip(*(coords for coords, *_ in self.learner.history[:steps + 1]))
        self.optimization_path.setData(x, y, data=np.arange(steps + 1))

        lx, ly = x[-1], y[-1]
        self.last_point.setData([lx], [ly], shape="o")
        self.optimization_label.setPos(lx, ly)
        self.optimization_label.setHtml(self._format_label(lx, ly, steps))
        self.optimization_label.show()

        self._update_hover_visibility()

        self.send_output()

    ##############################
    # Plot labels

    def help_event(self, event):
        if self.cost_grid is None:
            return False

        pos = event.scenePos()
        pos = self.graph.mapToView(pos)
        xc, yc = pos.x(), pos.y()

        if event.button() == Qt.LeftButton:
            self.change_theta(xc, yc)
            return True

        label = self.hover_label
        hovered_line = None

        # In case there are multiple hovered points or lines, pick the middle
        opt_points = self.optimization_path.scatter.pointsAt(pos)
        if opt_points.size:
            point = opt_points[len(opt_points) // 2]
            ppos = point.pos()
            label.setHtml(self._format_label(ppos.x(), ppos.y(), point.data()))
        else:
            hovereds = [item for item in self.contours
                        if isinstance(item, pg.PlotCurveItem)
                        and item.mouseShape().contains(pos)]
            if hovereds:
                hovered_line = hovereds[len(hovereds) // 2]
                # Show the cost for the line at mouse position, not the one
                # computed from coordinates. Use a different format to
                # distinguish this "line label" from the usual hover label
                label.setHtml(f"<b>Cost: {hovered_line.value:.3f}</b>")
            else:
                cost = self.learner.j(np.array([xc, yc]))
                label.setHtml(f"{xc:.3f}, {yc:.3f}<br/>Cost: {cost:.5f}")

        # Set the pen for all lines, hovered and not hovered (any longer)
        for item in self.contours:
            item.setPen(self._contour_pen(item is hovered_line))

        label.setPos(pos)
        label.show()
        self._update_hover_visibility()
        return True

    def _update_hover_visibility(self):
        # QGraphicsTextItem.collidesWith doesn't seem to work. I'm stupid,
        # or it may be related to ignoring transformations.
        opt_label = self.optimization_label
        hover_label = self.hover_label
        bopt = opt_label.boundingRect()
        bopt.moveTo(self.graph.mapFromView(opt_label.pos()))
        bhov = hover_label.boundingRect()
        bhov.moveTo(self.graph.mapFromView(hover_label.pos()))
        if hover_label.isVisible() and bhov.intersects(bopt):
            opt_label.hide()
        else:
            opt_label.show()

    def _format_label(self, x, y, step):
        return \
            f"<b>Step {step}:</b><br/>" \
            f"{x:.3f}, {y:.3f}<br/>" \
            f"Cost: {self.learner.j(np.array([x, y])):.5f}"

    def changeEvent(self, ev):
        # This hides the label if the user alt-tabs out of the window
        if ev.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self.hover_label.hide()
            self._update_hover_visibility()
        super().changeEvent(ev)


if __name__ == "__main__":
    WidgetPreview(OWGradientDescent).run(Table.from_file('iris'))
    # WidgetPreview(OWGradientDescent).run(Table.from_file('housing'))
