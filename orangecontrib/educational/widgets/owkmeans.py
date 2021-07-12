import numpy as np
import time

import pyqtgraph as pg

from AnyQt.QtCore import QThread, Qt, pyqtSignal as Signal, QTimer, QPointF
from AnyQt.QtGui import QPen, QFont, QPalette, QColor
from AnyQt.QtWidgets import QGraphicsTextItem, QGraphicsRectItem, \
    QGraphicsItemGroup

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

    def __del__(self):
        self.wait()

    def stop(self):
        self.is_running = False

    def run(self):
        """
        Stepping through the algorithm until converge or user interrupts
        """
        while (self.is_running and
               not self.owkmeans.k_means.converged and
               self.owkmeans.auto_play_enabled):
            try:
                self.owkmeans.step_trigger.emit()
            except RuntimeError:
                return
            time.sleep(2 - self.owkmeans.auto_play_speed)
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
        self.moved = True
        if self.centroid_index is not None:
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
    priority = 300

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        annotated_data = Output("Annotated Data", Table, default=True)
        centroids = Output("Centroids", Table)

    class Warning(OWWidget.Warning):
        num_features = Msg("Data must contain at least two numeric variables.")

    settingsHandler = settings.DomainContextHandler()
    attr_x = settings.ContextSetting(None)
    attr_y = settings.ContextSetting(None)
    number_of_clusters = settings.Setting(3)
    auto_play_speed = settings.Setting(1)

    graph_name = 'scatter'
    STEP_BUTTONS = ["Reassign Membership", "Recompute Centroids"]
    AUTOPLAY_BUTTONS = ["Run", "Stop"]

    step_trigger = Signal()
    stop_auto_play_trigger = Signal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.selected_rows = None  # rows that are selected for kmeans (not nan rows)
        self.auto_play_enabled = False
        self.auto_play_thread = None
        self.k_means = None
        self.color_map = np.empty(0, dtype=int)

        self.variables_box = gui.widgetBox(self.controlArea, "Variables")
        self.var_model = DomainModel(
            valid_types=(ContinuousVariable, ),
            order=DomainModel.MIXED)
        opts = dict(
            widget=self.variables_box, master=self, orientation=Qt.Horizontal,
            callback=self.restart, sendSelectedValue=True, model=self.var_model,
            #sizePolicy=(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
        )

        self.cbx = gui.comboBox(value='attr_x', **opts)
        self.cby = gui.comboBox(value='attr_y', **opts)

        self.centroids_box = gui.widgetBox(self.controlArea, "Centroids")
        self.restart_button = gui.button(
            self.centroids_box, self, "Randomize Positions",
            callback=self.restart)

        # control box
        self.step_box = gui.widgetBox(
            self.controlArea, "Manually step through", spacing=0)
        self.step_button = gui.button(
            self.step_box, self, self.STEP_BUTTONS[1], callback=self.step)
        self.step_back_button = gui.button(
            self.step_box, self, "Step Back", callback=self.step_back)

        self.run_box = gui.vBox(self.controlArea, "Run")

        self.auto_play_speed_spinner = gui.hSlider(
            self.run_box, self, 'auto_play_speed', label='Speed:',
            minValue=0, maxValue=1.91, step=0.1, intOnly=False,
            createLabel=False)
        self.auto_play_button = gui.button(
            self.run_box, self, self.AUTOPLAY_BUTTONS[0],
            callback=self.auto_play)

        gui.rubber(self.controlArea)

        # disable until data loaded
        self.set_disabled_all(True)

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
        self.mainArea.layout().addWidget(self.plotview)

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

    def set_points(self):
        km = self.k_means
        self.points_item = pg.ScatterPlotItem(
            x=km.data.get_column_view(0)[0],
            y=km.data.get_column_view(1)[0],
            symbol="o", size=8, antialias=True, useCache=False)
        self.update_membership()
        self.plotview.addItem(self.points_item)
        self.plotview.autoRange()
        self.plotview.replot()

    def update_membership(self):
        self.update_point_colors()
        self.update_membership_lines()

    def update_point_colors(self):
        assert self.points_item is not None
        km = self.k_means
        self.points_item.setPen(self.pens[km.clusters])
        self.points_item.setBrush(self.brushes[km.clusters])

    def set_membership_lines(self):
        m = np.zeros(2 * len(self.data))
        self.lines_item = pg.PlotCurveItem(
            x=m, y=m, pen=pg.mkPen(0.5), connect="pairs", antialias=True)
        self.plotview.addItem(self.lines_item)

    def set_centroids(self):
        k = self.k_means.k
        m = np.zeros(k)
        self.centroids_item = pg.ScatterPlotItem(
            x=m, y=m, pen=self.centr_pens[:k], brush=self.brushes[:k],
            symbol="s", size=13, antialias=True, useCache=False)
        self.plotview.set_centroids_item(self.centroids_item)
        self.plotview.addItem(self.centroids_item)
        self.update_centroid_positions()
        self.update_membership_lines()

    def on_centroid_dragged(self, centroid_index, x, y):
        self.k_means.centroids[centroid_index] = [x, y]
        self.update_centroid_positions()
        self.update_membership_lines()

    def on_centroid_done_dragging(self, centroid_index, x, y):
        self.k_means.move_centroid(centroid_index, x, y)
        self.animate_membership()
        self.send_data()

    def on_centroid_clicked(self, centroid_index):
        if self.number_of_clusters == 1:
            return
        self.k_means.delete_centroid(centroid_index)
        self.color_map = np.hstack((self.color_map[:centroid_index],
                                    self.color_map[centroid_index + 1:]))
        self._set_colors()
        self.number_of_clusters -= 1
        self.update_centroid_positions()
        self.animate_membership()
        self.send_data()

    def max_clusters(self):
        if self.k_means is None:
            return 10
        return max(10, len(self.k_means.data))

    def on_centroid_add(self, x, y):
        if not self.data or self.number_of_clusters == self.max_clusters():
            return

        self.number_of_clusters += 1
        self.color_map = np.hstack(
            (self.color_map,
             [min(set(range(10)) - set(self.color_map))]))
        self._set_colors()
        self.k_means.add_centroids([[x, y]])
        self.update_centroid_positions()
        self.animate_membership()
        self.send_data()
        self.button_text_change()

    def update_centroid_positions(self, cx=None, cy=None):
        assert self.centroids_item is not None
        assert self.lines_item is not None

        km = self.k_means
        k = km.k
        if cx is None:
            cx, cy = km.centroids.T
        self.centroids_item.setData(
            cx, cy,
            pen=self.centr_pens[:k], brush=self.brushes[:k])

    def update_plot(self):
        self.update_centroid_positions()
        self.update_membership()

    def update_membership_lines(self, cx=None, cy=None):
        x, y = self._membership_lines_data(cx, cy)
        self.lines_item.setData(x, y)

    def _membership_lines_data(self, cx=None, cy=None):
        km = self.k_means
        if cx is None:
            cx, cy = km.centroids.T
        n = len(self.data)
        x = np.empty(2 * n)
        y = np.empty(2 * n)
        x[::2] = km.data.get_column_view(0)[0]
        x[1::2] = cx[km.clusters]
        y[::2] = km.data.get_column_view(1)[0]
        y[1::2] = cy[km.clusters]
        return x, y

    def animate_centroids(self):
        def update(pos):
            self.update_centroid_positions(*pos.T)
            self.update_membership_lines(*pos.T)

        def done(pos):
            timer.stop()
            self.set_disabled_all(False)
            update(pos)

        timer = QTimer(self.centroids_item, interval=50)
        start = np.array(self.centroids_item.getData()).T
        final = self.k_means.centroids
        timer.timeout.connect(AnimateNumpy(start, final, update, done))
        self.set_disabled_all(True)
        timer.start()

    def animate_membership(self):
        def update(pos):
            self.lines_item.setData(*pos.T)

        def done(pos):
            timer.stop()
            self.set_disabled_all(False)
            update(pos)
            self.update_point_colors()

        timer = QTimer(self.lines_item, interval=100)
        start = np.array(self.lines_item.getData()).T
        final = np.array(self._membership_lines_data()).T
        diff = np.any((start != final)[1::2], axis=1)
        pens = self.pens[self.k_means.clusters]
        pens[diff] = [pg.mkPen(pen.color(), width=2) for pen in pens[diff]]
        self.points_item.setPen(pens)
        timer.timeout.connect(AnimateNumpy(start, final, update, done))
        self.set_disabled_all(True)
        timer.start()

    def concat_x_y(self):
        """
        Function takes two selected columns from data table and merge them in
        new Orange.data.Table

        Returns
        -------
        Orange.data.Table
            table with selected columns
        """
        attrs = [self.attr_x, self.attr_y]
        x = np.vstack(tuple(self.data.get_column_view(attr)[0]
                            for attr in attrs)).T
        # Prevent crash due to having the same attribute in the domain twice
        # (alternative, having a single column, would complicate other code)
        if self.attr_x is self.attr_y:
            attrs = [self.attr_x.renamed(name) for name in "xy"]
        not_nan = ~np.isnan(x).any(axis=1)
        x = x[not_nan]  # remove rows with nan
        self.selected_rows = np.where(not_nan)
        domain = Domain(attrs)
        return Table.from_numpy(domain, x)

    def set_disabled_all(self, disabled):
        """
        Function disable all controls
        """
        self.variables_box.setDisabled(disabled)
        self.centroids_box.setDisabled(disabled)
        self.step_box.setDisabled(disabled)
        self.run_box.setDisabled(disabled)

    @Inputs.data
    def set_data(self, data):
        """
        Function receives data from input and init part of widget if data are
        ok. Otherwise set empty plot and notice
        user about that

        Parameters
        ----------
        data : Orange.data.Table or None
            input data
        """
        self.Warning.clear()
        self.plotview.clear()
        self.set_disabled_all(True)
        if self.auto_play_thread:
            self.auto_play_thread.stop()

        self.data = data
        if not data:
            self.var_model.set_domain(None)
            return
        else:
            self.var_model.set_domain(data.domain)
            if len(self.var_model) < 2:
                self.Warning.num_features()
                return

        self.set_disabled_all(False)
        self.attr_x, self.attr_y = self.var_model[:2]

        if self.number_of_clusters > self.max_clusters():
            self.number_of_clusters = self.max_clusters()
        self.k_means = Kmeans(self.concat_x_y())

        self.color_map = np.arange(self.k_means.k)
        self.number_of_clusters_change()
        self._simplify_widget()

    def _set_colors(self):
        colors = DefaultRGBColors.qcolors[self.color_map]
        self.pens = np.array([pg.mkPen(col.darker(120)) for col in colors])
        self.centr_pens = np.array([pg.mkPen(col.darker(120), width=2) for col in colors])
        self.brushes = np.array([pg.mkBrush(col) for col in colors])

    def _simplify_widget(self):
        axes = ("bottom", "left")
        self.variables_box.setVisible(len(self.var_model) > 2)
        if [var.name for var in self.var_model] == ["x", "y"]:
            if np.min(self.k_means.data) >= 0 and np.max(self.k_means.data) <= 1:
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

    def restart(self):
        """
        Function triggered on data change or restart button pressed
        """
        self.k_means = Kmeans(self.concat_x_y())
        self.color_map = np.arange(self.number_of_clusters)
        self.number_of_clusters_change()

    def step(self):
        """
        Function called on every step
        """
        self.k_means.step()
        self.replot()
        self.button_text_change()
        self.send_data()

    def step_back(self):
        """
        Function called for step back
        """
        self.k_means.step_back()
        self.replot()
        self.button_text_change()
        self.send_data()
        self.number_of_clusters = self.k_means.k

    def button_text_change(self):
        """
        Function changes text on ste button and chanbe the button text
        """
        self.step_button.setText(self.STEP_BUTTONS[self.k_means.step_completed])
        if self.k_means.step_no <= 0:
            self.step_back_button.setDisabled(True)
        elif not self.auto_play_enabled:
            self.step_back_button.setDisabled(False)

    def auto_play(self):
        """
        Function called when autoplay button pressed
        """
        self.auto_play_enabled = not self.auto_play_enabled
        self.auto_play_button.setText(
            self.AUTOPLAY_BUTTONS[self.auto_play_enabled])
        if self.auto_play_enabled:
            self.variables_box.setDisabled(True)
            self.centroids_box.setDisabled(True)
            self.step_box.setDisabled(True)
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
        self.variables_box.setDisabled(False)
        self.centroids_box.setDisabled(False)
        self.step_box.setDisabled(False)
        self.auto_play_enabled = False
        self.auto_play_button\
            .setText(self.AUTOPLAY_BUTTONS[self.auto_play_enabled])
        self.button_text_change()

    def replot(self):
        """
        Function refreshes the chart
        """
        if self.data is None or not self.attr_x or not self.attr_y:
            return
        km = self.k_means
        if not km.centroids_moved:
            self.animate_membership()
        else:
            self.animate_centroids()

    def set_plot_items(self):
        self.plotview.clear()
        self.set_membership_lines()
        self.set_points()
        self.set_centroids()

    def number_of_clusters_change(self):
        """
        Function that change number of clusters if required
        """
        if self.data is None:
            return

        self.step_box.setDisabled(False)
        self.run_box.setDisabled(False)
        increase = self.number_of_clusters - self.k_means.k
        if increase > 0:
            available = sorted(set(range(10)) - set(self.color_map))
            self.color_map = np.hstack((self.color_map, available[:increase]))
            self.k_means.add_centroids(increase)
        elif increase < 0:
            self.color_map = self.color_map[:self.number_of_clusters]
            self.k_means.delete_centroids(-increase)
        self._set_colors()
        self.set_plot_items()
        self.send_data()
        self.button_text_change()

    def send_data(self):
        """
        Function sends data with clusters column and data with centroids
        position to the output
        """
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
            annotated_data.Y[self.selected_rows] = km.clusters

            centroids = Table.from_numpy(
                Domain(km.data.domain.attributes), km.centroids
            )
            self.Outputs.annotated_data.send(annotated_data)
            self.Outputs.centroids.send(centroids)

    def send_report(self):
        if self.data is None:
            return
        self.report_plot(self.plot)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    WidgetPreview(OWKmeans).run(Table.from_file('iris'))
