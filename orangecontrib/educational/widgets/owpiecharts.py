import sys
from math import sin, cos, pi

import numpy as np

from AnyQt.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsEllipseItem, QLabel, QGridLayout)
from AnyQt.QtGui import QColor, QPainter, QPixmap
from AnyQt.QtCore import Qt, QSize

import Orange.data
from Orange.statistics import contingency, distribution

from Orange.widgets import widget, gui
from Orange.widgets.settings import DomainContextHandler, ContextSetting, \
    Setting
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import Input

SCALE = 200


class OWPieChart(widget.OWWidget):
    name = "Pie Chart"
    description = "Make fun of Pie Charts."
    keywords = ["pie chart", "chart", "visualisation"]
    icon = "icons/PieChart.svg"
    priority = 700

    class Inputs:
        data = Input("Data", Orange.data.Table)

    settingsHandler = DomainContextHandler()
    attribute = ContextSetting(None)
    split_var = ContextSetting(None)
    explode = Setting(False)
    graph_name = "scene"

    def __init__(self):
        super().__init__()
        self.dataset = None

        self.attrs = DomainModel(
            valid_types=Orange.data.DiscreteVariable, separators=False)
        cb = gui.comboBox(
            self.controlArea, self, "attribute", box=True,
            model=self.attrs, callback=self.update_scene, contentsLength=12)
        grid = QGridLayout()
        self.legend = gui.widgetBox(gui.indentedBox(cb.box), orientation=grid)
        grid.setColumnStretch(1, 1)
        grid.setHorizontalSpacing(6)
        self.legend_items = []
        self.split_vars = DomainModel(
            valid_types=Orange.data.DiscreteVariable, separators=False,
            placeholder="None", )
        self.split_combobox = gui.comboBox(
            self.controlArea, self, "split_var", box="Split by",
            model=self.split_vars, callback=self.update_scene)
        self.explode_checkbox = gui.checkBox(
            self.controlArea, self, "explode", "Explode pies", box=True,
            callback=self.update_scene)
        gui.rubber(self.controlArea)
        gui.widgetLabel(
            gui.hBox(self.controlArea, box=True),
            "The aim of this widget is to\n"
            "demonstrate that pie charts are\n"
            "a terrible visualization. Please\n"
            "don't use it for any other purpose.")

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(
            QPainter.Antialiasing | QPainter.TextAntialiasing |
            QPainter.SmoothPixmapTransform)
        self.mainArea.layout().addWidget(self.view)
        self.mainArea.setMinimumWidth(500)

    def sizeHint(self):
        return QSize(200, 150)  # Horizontal size is regulated by mainArea

    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None and (
                not bool(dataset) or not len(dataset.domain)):
            dataset = None
        self.closeContext()
        self.dataset = dataset
        self.attribute = None
        self.split_var = None
        domain = dataset.domain if dataset is not None else None
        self.attrs.set_domain(domain)
        self.split_vars.set_domain(domain)
        if dataset is not None:
            self.select_default_variables(domain)
            self.openContext(self.dataset)
        self.update_scene()

    def select_default_variables(self, domain):
        if len(self.attrs) > len(domain.class_vars):
            first_attr = self.split_vars[len(domain.class_vars)]
        else:
            first_attr = None
        if len(self.attrs):
            self.attribute, self.split_var = self.attrs[0], first_attr
        else:
            self.attribute, self.split_var = self.split_var, None

    def update_scene(self):
        self.scene.clear()
        if self.dataset is None or self.attribute is None:
            return
        dists, labels = self.compute_box_data()
        colors = self.attribute.colors
        for x, (dist, label) in enumerate(zip(dists, labels)):
            self.pie_chart(SCALE * x, 0, 0.8 * SCALE, dist, colors)
            self.pie_label(SCALE * x, 0, label)
        self.update_legend(
            [QColor(*col) for col in colors], self.attribute.values)
        self.view.centerOn(SCALE * len(dists) / 2, 0)

    def update_legend(self, colors, labels):
        layout = self.legend.layout()
        while self.legend_items:
            w = self.legend_items.pop()
            layout.removeWidget(w)
            w.deleteLater()
        for row, (color, label) in enumerate(zip(colors, labels)):
            icon = QLabel()
            p = QPixmap(12, 12)
            p.fill(color)
            icon.setPixmap(p)
            label = QLabel(label)
            layout.addWidget(icon, row, 0)
            layout.addWidget(label, row, 1, alignment=Qt.AlignLeft)
            self.legend_items += (icon, label)

    def pie_chart(self, x, y, r, dist, colors):
        start_angle = 0
        dist = np.asarray(dist)
        spans = dist / (float(np.sum(dist)) or 1) * 360 * 16
        for span, color in zip(spans, colors):
            if not span:
                continue
            if self.explode:
                mid_ang = (start_angle + span / 2) / 360 / 16 * 2 * pi
                dx = r / 30 * cos(mid_ang)
                dy = r / 30 * sin(mid_ang)
            else:
                dx = dy = 0
            ellipse = QGraphicsEllipseItem(x - r / 2 + dx, y - r / 2 - dy, r, r)
            if len(spans) > 1:
                ellipse.setStartAngle(start_angle)
                ellipse.setSpanAngle(span)
            ellipse.setBrush(QColor(*color))
            self.scene.addItem(ellipse)
            start_angle += span

    def pie_label(self, x, y, label):
        if not label:
            return
        text = QGraphicsSimpleTextItem(label)
        for cut in range(1, len(label)):
            if text.boundingRect().width() < 0.95 * SCALE:
                break
            text = QGraphicsSimpleTextItem(label[:-cut] + "...")
        text.setPos(x - text.boundingRect().width() / 2, y + 0.5 * SCALE)
        self.scene.addItem(text)

    def compute_box_data(self):
        if self.split_var:
            return (
                contingency.get_contingency(
                    self.dataset, self.attribute, self.split_var),
                self.split_var.values)
        else:
            return [
                distribution.get_distribution(
                    self.dataset, self.attribute)], [""]

    def send_report(self):
        self.report_plot()
        text = ""
        if self.attribute is not None:
            text += "Box plot for '{}' ".format(self.attribute.name)
        if self.split_var is not None:
            text += "split by '{}'".format(self.split_var.name)
        if text:
            self.report_caption(text)


def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    if argv is None:
        argv = sys.argv
    argv = list(argv)
    app = QApplication(argv)
    filename = "heart_disease"
    data = Orange.data.Table(filename)
    w = OWPieChart()
    w.show()
    w.raise_()
    w.set_data(data)
    w.handleNewSignals()
    rval = app.exec_()
    w.set_data(None)
    w.handleNewSignals()
    w.saveSettings()
    return rval


if __name__ == "__main__":
    sys.exit(main())
