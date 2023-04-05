import numpy as np
import time
import os

import pyqtgraph as pg

from AnyQt.QtCore import QThread, Qt, pyqtSignal as Signal, QTimer, QPointF
from AnyQt.QtGui import QPen, QFont, QPalette, QColor, QIcon
from AnyQt.QtWidgets import QGraphicsTextItem, QGraphicsRectItem, \
    QGraphicsItemGroup, QSizePolicy, QToolButton

try:
    from AnyQt.QtMultimedia import QSound
    enable_sounds = True

    icons_dir = os.path.join(os.path.split(__file__)[0], "icons")
    speaker_on_icon = QIcon(os.path.join(icons_dir, "speaker-on.svg"))
    speaker_off_icon = QIcon(os.path.join(icons_dir, "speaker-off.svg"))
    del icons_dir
except ImportError:
    enable_sounds = False

from Orange.widgets import gui, settings
from Orange.widgets.utils.colorpalettes import DefaultRGBColors
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget, Msg, Input, Output
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table

from orangecontrib.educational.widgets.utils.kmeans import Kmeans


class Autoplay(QThread):
    """
    Class used for separated thread when using "Autoplay" for k-means

    Parameters
    ----------
    owkmeans : OWKmeans
        Instance of OWKmeans class
    """

    def __init__(self, owkmeans):
        QThread.__init__(self)
        self.owkmeans = owkmeans
        self.is_running = True
        self.step_count = 0

    def __del__(self):
        self.wait()

    def stop(self):
        self.is_running = False

    def run(self):
        while (self.is_running and
               not self.owkmeans.k_means.converged and
               self.step_count < 100 and
               self.owkmeans.auto_play_enabled):
            try:
                self.owkmeans.step_trigger.emit()
                self.step_count += 1
            except RuntimeError:
                return
            time.sleep(1.5)
        self.owkmeans.stop_auto_play_trigger.emit()


class AnimateNumpy:
    factors = [0.07, 0.26, 0.52, 0.77, 0.95, 1]

    def __init__(self, start, final, callback, done):
        self.start = start
        self.final = final
        self.diff = final - start
        self.callback = callback
        self.done = done
        self.step = 0

    def __call__(self):
        self.step += 1
        if self.step == len(self.factors):
            self.done(self.final)
        else:
            try:
                self.callback(self.start + self.diff * self.factors[self.step])
            except:
                # this is bad, but move on, otherwise you'll be stuck here forever
                pass


class KMeansPlotWidget(pg.PlotWidget):
    centroid_dragged = Signal(int, float, float)
    centroid_done_dragging = Signal(int, float, float)
    centroid_clicked = Signal(int)
    graph_clicked = Signal(float, float)
    mouse_entered = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centroids_item = None
        self.moved = False
        self.mouse_down = False
        self.centroid_index = None

    def set_centroids_item(self, centroids_item):
        self.centroids_item = centroids_item

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton \
                or self.centroids_item is None:
            ev.ignore()
            return

        ev.accept()
        self.mouse_down = True
        pos = self.plotItem.mapToView(QPointF(ev.pos()))
        pts = self.centroids_item.pointsAt(pos)
        if len(pts) != 0:
            self.centroid_index = \
                self.centroids_item.points().tolist().index(pts[0])

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            ev.ignore()
            return

        ev.accept()
        pos = self.plotItem.mapToView(QPointF(ev.pos()))
        x, y = pos.x(), pos.y()
        if self.centroid_index is not None:
            if self.moved:
                self.centroid_done_dragging.emit(self.centroid_index, x, y)
            else:
                self.centroid_clicked.emit(self.centroid_index)
        else:
            if not self.moved:
                self.graph_clicked.emit(x, y)
        self.centroid_index = None
        self.mouse_down = False
        self.moved = False

    def mouseMoveEvent(self, ev):
        if not self.mouse_down:
            ev.ignore()
            return

        ev.accept()
        if self.centroid_index is not None:
            self.moved = True
            pos = self.plotItem.mapToView(QPointF(ev.pos()))
            self.centroid_dragged.emit(self.centroid_index, pos.x(), pos.y())

    def enterEvent(self, ev):
        super().enterEvent(ev)
        self.mouse_entered.emit()


class OWKmeans(OWWidget):
    name = "Interactive k-Means"
    description = "Widget demonstrates working of k-means algorithm."
    keywords = ["kmeans", "clustering", "interactive"]
    icon = "icons/InteractiveKMeans.svg"
    want_main_area = True
    want_control_area = False
    priority = 300

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output("Annotated Data", Table, default=True)
        centroids = Output("Centroids", Table)

    class Error(OWWidget.Error):
        num_features = Msg("Data must contain at least two numeric variables.")
        no_nonnan_data = Msg("No points with defined values.")

    settingsHandler = settings.DomainContextHandler()
    attr_x = settings.ContextSetting(None)
    attr_y = settings.ContextSetting(None)
    sound_effects = settings.Setting(False)

    graph_name = 'plot'  # pg.GraphicsItem  (pg.PlotItem)
    move_sound = regroup_sound = None

    step_trigger = Signal()
    stop_auto_play_trigger = Signal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.reduced_data = None  # data from selected columns, no nans
        self.selected_rows = None  # rows without nans, which are used in kmeans
        self.in_animation = False
        self.auto_play_enabled = False
        self.auto_play_thread = None
        self.k_means = None
        self.color_map = np.empty(0, dtype=int)
        self.number_of_clusters = 3

        self._create_variables_box()
        self._create_plot()
        self._create_buttons_box()

        if enable_sounds:
            sound_dir = os.path.join(os.path.split(__file__)[0], "sounds")
            self.regroup_sound = QSound(os.path.join(sound_dir, "clack.wav"))
            self.move_sound = QSound(os.path.join(sound_dir, "arrow.wav"))

    def _simplify_widget(self):
        axes = ("bottom", "left")
        self.variables_box.setVisible(len(self.var_model) > 2)
        if [var.name for var in self.var_model] == ["x", "y"]:
            if np.min(self.reduced_data) >= 0 and np.max(self.reduced_data) <= 1:
                for axis in axes:
                    self.plot.hideAxis(axis)
            else:
                for axis in axes:
                    self.plot.getAxis(axis).showLabel(False)
        else:
            for axis, attr in zip(axes, (self.attr_x, self.attr_y)):
                axis = self.plot.getAxis(axis)
                axis.showLabel(True)
                axis.setLabel(attr.name)

    #########################################
    # Variable combo boxes
    def _create_variables_box(self):
        self.variables_box = gui.hBox(self.mainArea, box=True)
        gui.widgetLabel(self.variables_box, "Variables:")
        self.var_model = DomainModel(
            valid_types=(ContinuousVariable, ),
            order=DomainModel.MIXED)
        for attr in ("attr_x", "attr_y"):
            gui.comboBox(
                self.variables_box, self, value=attr, model=self.var_model,
                callback=self.restart, orientation=Qt.Horizontal,
                sizePolicy=(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
            )

    #########################################
    # Buttons: creation and updates, and actions
    def _create_buttons_box(self):
        box = gui.hBox(self.plot_box, spacing=0, margin=0)
        self.step_button = \
            gui.button(box, self, "", callback=self.step)
        self.step_back_button = \
            gui.button(box, self, "Step Back", callback=self.step_back)
        if enable_sounds:
            self.speaker_button = QToolButton(self)
            self.speaker_button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self._update_sound_effect_button()
            self.speaker_button.clicked.connect(self._toggle_sound_effects)
            self.speaker_button.setStyleSheet("border: none")
            box.layout().addWidget(self.speaker_button)
        gui.rubber(box)
        self.restart_button = \
            gui.button(box, self, "Randomize", callback=self.restart)
        self.auto_play_button = \
            gui.button(box, self, "", callback=self.auto_play)
        self._update_buttons()

    if enable_sounds:
        def _toggle_sound_effects(self):
            self.sound_effects = not self.sound_effects
            self._update_sound_effect_button()

        def _update_sound_effect_button(self):
            self.speaker_button.setIcon(
                QIcon([speaker_off_icon, speaker_on_icon][self.sound_effects]))

    def _update_buttons(self):
        animation = self.in_animation
        no_data = self.data is None or self.k_means is None
        running = self.auto_play_enabled
        disabled = animation or no_data or running

        self.variables_box.setDisabled(disabled)

        self.step_button.setDisabled(disabled)
        self.step_button.setText(
            "Reassign Membership"
            if self.k_means and self.k_means.waits_reassignment
            else "Recompute Centroids")

        history = bool(self.k_means and self.k_means.history)
        self.step_back_button.setVisible(history)
        self.step_back_button.setDisabled(
            disabled
            or history
            and len(self.k_means.history[-1].centroids)
            != self.number_of_clusters)

        self.restart_button.setDisabled(disabled)

        # Stop button is never disabled when running (even during animation)
        self.auto_play_button.setDisabled(disabled and not running)
        self.auto_play_button.setText("Stop" if running else "Run Simulation")

    def step(self):
        if self.k_means.waits_reassignment:
            self.k_means.assign_membership()
            self.animate_membership()
        else:
            self.k_means.move_centroids()
            self.animate_centroids()
        self._update_buttons()
        self.send_data()

    def step_back(self):
        self.k_means.step_back()
        self.animate_centroids()
        self.animate_membership()
        self._update_buttons()
        self.send_data()
        self.number_of_clusters = self.k_means.k

    def auto_play(self):
        self.auto_play_enabled = not self.auto_play_enabled
        self._update_buttons()
        if self.auto_play_enabled:
            # This will disable all except the auto_play button
            self.auto_play_thread = Autoplay(self)
            self.step_trigger.connect(self.step)
            self.stop_auto_play_trigger.connect(self.stop_auto_play)
            self.auto_play_thread.start()
        else:
            self.stop_auto_play()

    def stop_auto_play(self):
        self.auto_play_enabled = False
        self._update_buttons()

    #########################################
    # Plot creation
    def _create_plot(self):
        self.points_item = None
        self.centroids_item = None
        self.lines_item = None

        self.plotview = KMeansPlotWidget(background="w", autoRange=False)
        self.plot = self.plotview.getPlotItem()
        axis_pen = QPen(self.palette().color(QPalette.Text))
        tickfont = QFont(self.font())
        tickfont.setPixelSize(max(int(tickfont.pixelSize() * 2 // 3), 11))
        for axis in ("bottom", "left"):
            axis = self.plot.getAxis(axis)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
            axis.setTickFont(tickfont)
        self.plot.hideButtons()
        self.plot.setMouseEnabled(x=False, y=False)
        self.plotview.centroid_clicked.connect(self.on_centroid_clicked)
        self.plotview.centroid_dragged.connect(self.on_centroid_dragged)
        self.plotview.centroid_done_dragging.connect(self.on_centroid_done_dragging)
        self.plotview.graph_clicked.connect(self.on_centroid_add)

        self.plot_box = gui.vBox(self.mainArea, box=True)
        self.plot_box.layout().addWidget(self.plotview)

        self._create_tooltip()
        self.plotview.mouse_entered.connect(self.show_tooltip)

    def _create_tooltip(self):
        text = QGraphicsTextItem()
        text.setHtml("Drag centroids to move them, click to add and remove them.")
        text.setPos(4, 2)
        r = text.boundingRect()
        text.setTextWidth(r.width())
        rect = QGraphicsRectItem(0, 0, r.width() + 8, r.height() + 4)
        rect.setBrush(QColor(224, 224, 224, 212))
        rect.setPen(QPen(Qt.NoPen))

        tooltip_group = QGraphicsItemGroup()
        tooltip_group.addToGroup(rect)
        tooltip_group.addToGroup(text)
        tooltip_group.setFlag(tooltip_group.ItemIgnoresTransformations)
        self.tooltip = tooltip_group

    def show_tooltip(self):
        if self.tooltip is None or self.data is None or self.k_means is None:
            return

        tooltip = self.tooltip
        self.tooltip = None

        self.plot.scene().addItem(tooltip)
        tooltip.setPos(10, 10)
        timer = QTimer(self)
        timer.singleShot(5000, lambda: self.plot.scene().removeItem(tooltip))

    def _set_plot_items(self):
        self._set_membership_lines()
        self._set_points()
        self._set_centroids()

    def _set_membership_lines(self):
        m = np.zeros(2 * len(self.reduced_data))
        self.lines_item = pg.PlotCurveItem(
            x=m, y=m, pen=pg.mkPen(0.5), connect="pairs", antialias=True)
        self.plotview.addItem(self.lines_item)

    def _set_points(self):
        assert self.k_means is not None
        km = self.k_means
        self.points_item = pg.ScatterPlotItem(
            *self.reduced_data.T,
            symbol="o", size=8, antialias=True, useCache=False)
        self.update_membership()
        self.plotview.addItem(self.points_item)
        self.plotview.autoRange()
        self.plotview.replot()

    def _set_centroids(self):
        assert self.k_means is not None
        k = self.k_means.k
        m = np.zeros(k)
        self.centroids_item = pg.ScatterPlotItem(
            x=m, y=m, pen=self.centr_pens[:k], brush=self.brushes[:k],
            symbol="s", size=13, antialias=True, useCache=False)
        self.plotview.set_centroids_item(self.centroids_item)
        self.plotview.addItem(self.centroids_item)
        self.update_centroid_positions()
        self.update_membership_lines()

    #########################################
    # Plot element updates and animations
    def update_plot(self):
        self.update_centroid_positions()
        self.update_membership()

    def update_membership(self):
        self.update_point_colors()
        self.update_membership_lines()

    def update_point_colors(self):
        assert self.points_item is not None
        km = self.k_means
        self.points_item.setPen(self.pens[km.clusters])
        self.points_item.setBrush(self.brushes[km.clusters])

    def update_centroid_positions(self, cx=None, cy=None):
        assert self.centroids_item is not None
        assert self.lines_item is not None

        km = self.k_means
        k = km.k
        if cx is None:
            cx, cy = km.centroids.copy().T
        self.centroids_item.setData(
            cx, cy,
            pen=self.centr_pens[:k], brush=self.brushes[:k])

    def update_membership_lines(self, cx=None, cy=None):
        x, y = self._membership_lines_data(cx, cy)
        self.lines_item.setData(x, y)

    def _membership_lines_data(self, cx=None, cy=None):
        km = self.k_means
        if cx is None:
            cx, cy = km.centroids.T
        n = len(self.reduced_data)
        x = np.empty(2 * n)
        y = np.empty(2 * n)
        x[::2], y[::2] = self.reduced_data.T
        x[1::2] = cx[km.clusters]
        y[1::2] = cy[km.clusters]
        return x, y

    def animate_centroids(self):
        def update(pos):
            self.update_centroid_positions(*pos.T)
            self.update_membership_lines(*pos.T)

        start = np.array(self.centroids_item.getData()).T
        final = self.k_means.centroids.copy()
        self._animate(start, final, update, update)
        if enable_sounds and self.sound_effects:
            self.move_sound.play()

    def animate_membership(self):
        def update(pos):
            self.lines_item.setData(*pos.T)

        def done(pos):
            update(pos)
            self.update_point_colors()

        start = np.array(self.lines_item.getData()).T
        final = np.array(self._membership_lines_data()).T
        diff = np.any((start != final)[1::2], axis=1)
        pens = self.pens[self.k_means.clusters]
        pens[diff] = [pg.mkPen(pen.color(), width=2) for pen in pens[diff]]
        self.points_item.setPen(pens)
        self._animate(start, final, update, done)
        if enable_sounds and self.sound_effects:
            self.regroup_sound.play()

    def _animate(self, start, final, update, done):
        def my_done(pos):
            timer.stop()
            self.in_animation = False
            self._update_buttons()
            done(pos)

        timer = QTimer(self, interval=50)
        self.in_animation = True
        self._update_buttons()
        timer.start()
        timer.timeout.connect(AnimateNumpy(start, final, update, my_done))

    #########################################
    # Plot: user interaction
    def on_centroid_dragged(self, centroid_index, x, y):
        if self.auto_play_enabled:
            return
        centroids = self.k_means.centroids.copy()
        centroids[centroid_index] = [x, y]
        self.update_centroid_positions(*centroids.T)
        self.update_membership_lines(*centroids.T)

    def on_centroid_done_dragging(self, centroid_index, x, y):
        if self.auto_play_enabled:
            return
        self.k_means.move_centroid(centroid_index, x, y)
        self.animate_membership()
        self.send_data()

    def on_centroid_clicked(self, centroid_index):
        if self.number_of_clusters == 1 or self.auto_play_enabled:
            return
        self.k_means.delete_centroid(centroid_index)
        self.color_map = np.hstack((self.color_map[:centroid_index],
                                    self.color_map[centroid_index + 1:]))
        self._set_colors()
        self.number_of_clusters -= 1
        self.update_centroid_positions()
        self.animate_membership()
        self.send_data()

    def on_centroid_add(self, x, y):
        if not self.data \
                or self.k_means is None \
                or self.number_of_clusters == self.max_clusters() \
                or self.auto_play_enabled:
            return

        self.number_of_clusters += 1
        self.color_map = np.hstack(
            (self.color_map,
             [min(set(range(10))- set(self.color_map))]))
        self._set_colors()
        self.k_means.add_centroid(x, y)
        self.update_centroid_positions()
        self.animate_membership()
        self.send_data()

    def _set_colors(self):
        colors = DefaultRGBColors.qcolors[self.color_map]
        self.pens = np.array([pg.mkPen(col.darker(120)) for col in colors])
        self.centr_pens = np.array([pg.mkPen(col.darker(120), width=2) for col in colors])
        self.brushes = np.array([pg.mkBrush(col) for col in colors])

    #########################################
    # Signals reports ...
    @Inputs.data
    def set_data(self, data):
        self.Error.clear()
        self.plotview.clear()
        self._update_buttons()
        if self.auto_play_thread:
            self.auto_play_thread.stop()

        self.data = data
        self.reduced_data = None  # updated below, at restart
        if not data:
            self.var_model.set_domain(None)
            return
        else:
            self.var_model.set_domain(data.domain)
            if len(self.var_model) < 2:
                self.Error.num_features()
                self.var_model.set_domain(None)
                return

        self.attr_x, self.attr_y = self.var_model[:2]
        self.restart()

    def restart(self):
        """Triggered on data change, attribute change or restart button"""
        self.plotview.clear()
        self.Error.no_nonnan_data.clear()
        self.reduced_data = self._prepare_data()
        if self.reduced_data is None:
            self.Error.no_nonnan_data()
            self.k_means = None
        else:
            if self.number_of_clusters > self.max_clusters():
                self.number_of_clusters = self.max_clusters()
            self.k_means = Kmeans(self.reduced_data, self.number_of_clusters)
            self.color_map = np.arange(self.number_of_clusters)
            self._set_colors()
            self._set_plot_items()
            self._simplify_widget()
        self._update_buttons()
        self.send_data()

    def _prepare_data(self):
        """
        Prepare 2d data for clustering

        Put the two columns into a new table and remove rows with nans.
        Return None if there are no non-nan columns
        """
        attrs = [self.attr_x, self.attr_y]
        x = np.vstack(tuple(self.data.get_column(attr) for attr in attrs)).T
        not_nan = ~np.isnan(x).any(axis=1)
        x = x[not_nan]  # remove rows with nan
        if not x.size:
            return None
        self.selected_rows = np.where(not_nan)
        return x

    def max_clusters(self):
        if self.reduced_data is None:
            return 10
        return min(10, len(self.reduced_data))

    def send_data(self):
        km = self.k_means
        if km is None or km.clusters is None:
            self.Outputs.annotated_data.send(None)
            self.Outputs.centroids.send(None)
        else:
            clust_var = DiscreteVariable(
                "cluster",
                values=["C%d" % (x + 1) for x in range(km.k)],
            )
            clust_var.colors = DefaultRGBColors.palette[self.color_map]
            attributes = self.data.domain.attributes
            classes = self.data.domain.class_vars
            meta_attrs = self.data.domain.metas
            if classes:
                meta_attrs += classes
            classes = [clust_var]
            domain = Domain(attributes, classes, meta_attrs)
            annotated_data = Table.from_table(domain, self.data)
            with annotated_data.unlocked(annotated_data.Y):
                annotated_data.Y[self.selected_rows] = km.clusters

            if self.attr_x is self.attr_y:
                attrs = [self.attr_x.renamed(name) for name in "xy"]
            else:
                attrs = [self.attr_x, self.attr_y]
            centroids = Table.from_numpy(Domain(attrs), km.centroids.copy())
            self.Outputs.annotated_data.send(annotated_data)
            self.Outputs.centroids.send(centroids)

    def send_report(self):
        if self.data is None:
            return
        self.report_plot()


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWKmeans).run(Table.from_file('iris'))
