from collections import namedtuple
from functools import partial
from math import ceil, log10

import numpy as np
from scipy import stats

from AnyQt.QtCore import Qt, pyqtSignal as Signal, QTimer
from AnyQt.QtWidgets import QComboBox, QFormLayout, \
    QLineEdit, QGroupBox, QStyle, QPushButton, QLabel, QVBoxLayout, \
    QSizePolicy, QSpacerItem
from AnyQt.QtGui import QIntValidator, QDoubleValidator

from Orange.data import Table, ContinuousVariable, Domain, DiscreteVariable
from Orange.widgets.settings import Setting
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Output, Msg
from Orange.widgets import gui


ParameterDef = namedtuple(
    "ParameterDef",
    ("label", "arg_name", "default", "arg_type"))


class ParameterError(ValueError):
    pass


class pos_int(int):  # pylint: disable=invalid-name
    validator = QIntValidator()


class float_convert(float):  # pylint: disable=invalid-name
    @staticmethod
    def convert(s):
        s = s.replace(",", ".")
        if s.endswith("."):
            s += "0"
        return float(s)


class any_float(float_convert):  # pylint: disable=invalid-name
    validator = QDoubleValidator()


class pos_float(float_convert):  # pylint: disable=invalid-name
    validator = QDoubleValidator()
    validator.setBottom(0.0001)


class prob_float(float_convert):  # pylint: disable=invalid-name
    validator = QDoubleValidator(0, 1, 5)


class ParametersEditor(QGroupBox):
    remove_clicked = Signal()

    default_prefix = "Var"
    parameters = ()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle(self.name)
        self.setLayout(QVBoxLayout())

        hbox = gui.hBox(self)
        self.add_standard_parameters(hbox)
        gui.separator(hbox, 20)
        gui.rubber(hbox)
        self.add_specific_parameters(hbox)

        hbox = gui.hBox(self)
        self.error = QLabel()
        self.error.setHidden(True)
        gui.rubber(hbox)
        hbox.layout().addWidget(self.error)

        self.trash_button = trash = QPushButton(self, text='×')
        trash.setFixedWidth(35)
        size = trash.sizeHint()
        trash.setGeometry(0, 20, size.width(), size.height())
        trash.setHidden(True)
        trash.clicked.connect(self.on_trash_clicked)

    def on_trash_clicked(self):
        self.remove_clicked.emit()

    def enterEvent(self, e):
        super().enterEvent(e)
        self.trash_button.setHidden(False)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        self.trash_button.setHidden(True)

    def add_standard_parameters(self, parent):
        form = QFormLayout()
        parent.layout().addLayout(form)

        self.number_of_vars = edit = QLineEdit()
        edit.setValidator(pos_int.validator)
        edit.setText("10")
        edit.setAlignment(Qt.AlignRight)
        edit.setFixedWidth(50)
        form.addRow("Variables", edit)

        self.name_prefix = edit = QLineEdit()
        edit.setPlaceholderText(self.default_prefix)
        edit.setFixedWidth(50)
        form.addRow("Name prefix", edit)

    def fix_standard_parameters(self, number_of_vars, name_prefix):
        self.number_of_vars.setDisabled(number_of_vars is not None)
        self.name_prefix.setDisabled(name_prefix is not None)
        if number_of_vars is not None:
            self.number_of_vars.setText(str(number_of_vars))
        if name_prefix is not None:
            self.name_prefix.setText(name_prefix)

    def add_specific_parameters(self, parent):
        form = QFormLayout()
        self.edits = {}
        for parameter in self.parameters:
            edit = self.edits[parameter.arg_name] = QLineEdit()
            edit.convert = getattr(
                parameter.arg_type, "convert", parameter.arg_type)
            validator = getattr(parameter.arg_type, "validator", None)
            if validator is not None:
                edit.setValidator(validator)
            edit.setText(str(parameter.default))
            edit.setAlignment(Qt.AlignRight)
            edit.setFixedWidth(50)
            form.addRow(parameter.label, edit)
        parent.layout().addLayout(form)

    @property
    def nvars(self):
        return int(self.number_of_vars.text())

    def get(self, name):
        edit = self.edits[name]
        try:
            return edit.convert(edit.text())
        except ValueError:
            return None

    def get_parameters(self):
        parameters = {}
        for (name, edit), parameter in zip(self.edits.items(), self.parameters):
            value = self.get(name)
            if value is None:
                raise ParameterError(
                    f"Invalid value ({edit.text()}) "
                    f"for <em>{parameter.label.lower()}</em>")
            parameters[name] = value
        return parameters

    @staticmethod
    def check(**_):
        return None

    def set_error(self, error):
        self.error.setText(f"<font color='red'>{error}</font>" if error else "")
        self.error.setHidden(error is None)

    def prepare_variables(self, used_names, ndigits):
        raise NotImplementedError

    def get_name_prefix(self, used_names):
        name = self.name_prefix.text() or self.default_prefix
        start = used_names.get(name, 0) + 1
        return name, start

    def generate_partial_data(self, ninstances):
        data = None
        error = None
        try:
            parameters = self.get_parameters()
        except ParameterError as exc:
            error = str(exc)
        if error is None:
            # pylint: disable=assignment-from-none
            error = self.check(**parameters)
        if error is None:
            try:
                data = self.rvs(size=(ninstances, self.nvars), **parameters)
            except:  # can throw anything, pylint: disable=bare-except
                error = "Error while sampling. Check distribution parameters."
        self.set_error(error)
        return data

    def pack_settings(self):
        return dict(number_of_vars=self.number_of_vars.text(),
                    name_prefix=self.name_prefix.text(),
                    **{name: edit.text() for name, edit in self.edits.items()})

    def unpack_settings(self, settings):
        edits = dict(number_of_vars=self.number_of_vars,
                     name_prefix=self.name_prefix,
                     **self.edits)
        for name, value in settings.items():
            edits[name].setText(value)


class ParametersEditorContinuous(ParametersEditor):
    def prepare_variables(self, used_names, ndigits):
        name, start = self.get_name_prefix(used_names)
        used_names[name] = start + self.nvars - 1
        return [ContinuousVariable.make(f"{name}{i:0{ndigits}}")
                for i in range(start, start + self.nvars)]


class ParametersEditorDiscrete(ParametersEditor):
    def prepare_variables(self, used_names, ndigits):
        name, start = self.get_name_prefix(used_names)
        used_names[name] = start + self.nvars - 1
        try:
            parameters = self.get_parameters()
        except ParameterError:
            return None
        values = self.get_values(**parameters)
        return [DiscreteVariable.make(f"{name}{i:0{ndigits}}", values=values)
                for i in range(start, start + self.nvars)]


class Bernoulli(ParametersEditorDiscrete):
    name = "Bernoulli distribution"
    parameters = (ParameterDef("Probability", "p", 0.5, prob_float), )
    rvs = stats.bernoulli.rvs

    @staticmethod
    def get_values(**_):
        return "0", "1"


class ContinuousUniform(ParametersEditorContinuous):
    name = "Uniform distribution"
    parameters = (
        ParameterDef("Low bound", "loc", 0, any_float),
        ParameterDef("High bound", "scale", 1, any_float)
    )
    rvs = stats.uniform.rvs

    @staticmethod
    def check(*, loc, scale):  # pylint: disable=arguments-differ
        if loc >= scale:
            return "Lower bound is must be below the upper."
        return None


class DiscreteUniform(ParametersEditorDiscrete):
    name = "Discrete uniform distribution"
    parameters = (ParameterDef("Number of values", "k", 6, pos_int), )

    @staticmethod
    def rvs(k, size):
        return stats.randint.rvs(0, k, size=size)

    @staticmethod
    def get_values(k):
        return [str(i) for i in range(1, k + 1)]


class Multinomial(ParametersEditorContinuous):
    name = "Multinomial distribution"
    parameters = (
        ParameterDef("Probabilities", "ps", "0.5, 0.3, 0.2", str),
        ParameterDef("Number of trials", "n", 100, pos_int)
    )

    def __init__(self):
        super().__init__()
        self.edits["ps"].setFixedWidth(120)
        self.edits["ps"].setAlignment(Qt.AlignLeft)
        self.get("ps")  # trigger an update of standard parameters

    @property
    def nvars(self):
        return len(self.get("ps"))

    def get(self, name):
        val = super().get(name)
        if name != "ps":
            return val
        ps = self._ps_or_error()
        if isinstance(ps, str):
            return []
        else:
            self.fix_standard_parameters(len(ps), None)
        return ps

    def _ps_or_error(self, s=None):
        if s is None:
            s = self.edits["ps"].text()
        try:
            ps = [float(x)
                  for x in s.replace(",", " ").replace(";", " ").split()]
        except ValueError:
            return "Probabilities must be given as list of numbers."
        tot = sum(ps)
        if abs(tot - 1) > 1e-6:
            return f"Probabilities must sum to 1, not {tot:.4f}."
        return ps

    @staticmethod
    def rvs(ps, n, size):
        if not ps:
            return None
        return stats.multinomial.rvs(p=ps, n=n, size=size[0])

    def check(self, **_):
        ps = self._ps_or_error()
        return ps if isinstance(ps, str) else None


class HyperGeometric(ParametersEditorContinuous):
    name = "Hypergeometric distribution"
    parameters = (
        ParameterDef("Number of objects", "M", 100, pos_int),
        ParameterDef("Number of positives", "n", 20, pos_int),
        ParameterDef("Number of trials", "N", 20, pos_int)
    )
    rvs = stats.hypergeom.rvs

    @staticmethod
    def check(*, M, n, N):  # pylint: disable=arguments-differ
        if n > M:
            return "Number of positives exceeds number of objects."
        if N > M:
            return "Number of trials exceeds number of objects."
        return None


class BivariateNormal(ParametersEditor):
    name = "Bivariate normal distribution"
    parameters = (
        ParameterDef("Mean x", "mu1", 0, any_float),
        ParameterDef("Variance x", "var1", 1, pos_float),
        ParameterDef("Mean y", "mu2", 0, any_float),
        ParameterDef("Variance y", "var2", 1, pos_float),
        ParameterDef("Covariance", "covar", 0.5, pos_float)
    )
    nvars = 2

    def add_standard_parameters(self, parent):
        super().add_standard_parameters(parent)
        self.fix_standard_parameters("2", "x, y")

    def prepare_variables(self, used_names, ndigits):
        start = 1 + max(used_names.get("x", 0), used_names.get("y", 0))
        used_names["x"] = used_names["y"] = start
        return [ContinuousVariable.make(f"x{start:0{ndigits}}"),
                ContinuousVariable.make(f"y{start:0{ndigits}}")]

    @staticmethod
    def rvs(*, mu1, mu2, var1, var2, covar, size):
        return stats.multivariate_normal.rvs(
            mean=np.array([mu1, mu2]),
            cov=np.array([[var1, covar], [covar, var2]]),
            size=size[0])


def cd(name, rvs, *parameters):  # short-lived, pylint: disable=invalid-name
    return type(
        name.title().replace(" ", ""),
        (ParametersEditorContinuous,),
        dict(name=name, rvs=rvs,
             parameters=[ParameterDef(*p) for p in parameters]))


dist_defs = [
    cd("Normal distribution", stats.norm.rvs,
       ("Mean", "loc", 0, any_float),
       ("Standard deviation", "scale", 1, pos_float)),
    Bernoulli,
    cd("Binomial distribution", stats.binom.rvs,
       ("Number of trials", "n", 100, pos_int),
       ("Probability of success", "p", 0.5, prob_float)),
    ContinuousUniform,
    DiscreteUniform,
    Multinomial,
    HyperGeometric,
    cd("Negative binomial distribution", stats.nbinom.rvs,
       ("Number of successes", "n", 10, pos_int),
       ("Probability of success", "p", 0.5, prob_float)),
    cd("Poisson distribution", stats.poisson.rvs,
       ("Event rate (λ)", "mu", 5, pos_float)),
    cd("Exponential distribution", stats.expon.rvs),
    cd("Gamma distribution", stats.gamma.rvs,
       ("Shape (α)", "a", 2, pos_float),
       ("Scale", "scale", 2, pos_float)),
    cd("Student's t distribution", stats.t.rvs,
       ("Degrees of freedom", "df", 1, pos_float)),
    BivariateNormal
]


class RandomDataVerticalScrollArea(gui.VerticalScrollArea):
    def sizeHint(self):
        sh = super().sizeHint()
        sh.setHeight(350)
        return sh


distributions = {dist.name: dist for dist in dist_defs}
del dist_defs
del cd


class OWRandomData(OWWidget):
    name = "Random Data"
    description = "Generate random data sample"
    keywords = ["random data", "data", "data generation"]
    icon = "icons/RandomData.svg"
    priority = 2100

    class Error(OWWidget.Error):
        sampling_error = \
            Msg("Error while sampling. Check distribution arguments.")

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False

    n_instances = Setting(1000)
    distributions = Setting([
        ('Normal distribution',
         {'number_of_vars': '10', 'name_prefix': '', 'loc': '0', 'scale': '1'}),
        ('Binomial distribution',
         {'number_of_vars': '1', 'name_prefix': '', 'n': '100', 'p': '0.5'})])

    def __init__(self):
        super().__init__()
        self.editors = []

        self.scroll_area = RandomDataVerticalScrollArea(self.controlArea)
        self.scroll_area.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.editor_vbox = gui.vBox(self.controlArea, spacing=0)
        self.editor_vbox.layout().addSpacerItem(
            QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding)
        )
        self.scroll_area.setWidget(self.editor_vbox)
        self.controlArea.layout().addWidget(self.scroll_area)

        class IgnoreWheelCombo(QComboBox):
            @staticmethod
            def wheelEvent(event):
                event.ignore()

        # self.add_combo is needed so that tests can manipulate it
        combo = self.add_combo = IgnoreWheelCombo()
        combo.addItem("Add more variables ...")
        combo.addItems(list(distributions))
        combo.currentTextChanged.connect(self.on_add_distribution)
        combo.setFocusPolicy(Qt.NoFocus)
        self.controlArea.layout().addWidget(combo)
        gui.separator(self.controlArea, 16)

        box = gui.vBox(self.controlArea, box=True)
        box2 = gui.hBox(box)
        gui.lineEdit(
            box2, self, "n_instances", "Sample size",
            orientation=Qt.Horizontal, controlWidth=70, alignment=Qt.AlignRight,
            valueType=int, validator=QIntValidator(1, 1_000_000))
        gui.rubber(box2)
        gui.button(
            box2, self, label="Generate", callback=self.generate, width=160)

        self.settingsAboutToBePacked.connect(self.pack_editor_settings)
        self.unpack_editor_settings()
        self.generate()

    def on_add_distribution(self, dist_name):
        combo = self.sender()
        if not combo.currentIndex():
            return
        editor_class = distributions[dist_name]
        self.add_editor(editor_class())
        combo.setCurrentIndex(0)

    def add_editor(self, editor):
        editor.remove_clicked.connect(self.remove_editor)
        self.editor_vbox.layout().insertWidget(len(self.editors), editor)
        self.editors.append(editor)
        self.editor_vbox.updateGeometry()
        QTimer.singleShot(0, partial(self.scroll_area.ensureWidgetVisible, editor))
        self.generate()

    def remove_editor(self):
        editor = self.sender()
        self.controlArea.layout().removeWidget(editor)
        self.editors.remove(editor)
        editor.deleteLater()
        self.generate()

    def generate_data(self):
        self.Error.clear()
        used_names = {}

        editors = self.editors
        if not editors:
            return None
        ndigits = int(ceil(log10(1 + sum(e.nvars for e in editors))))

        attrs = []
        data_parts = []
        for editor in editors:
            part_attrs = editor.prepare_variables(used_names, ndigits)
            if part_attrs is None:
                return None
            attrs += part_attrs

            part_data = editor.generate_partial_data(self.n_instances)
            if part_data is None:
                self.Error.sampling_error()
                return None
            data_parts.append(part_data)

        return Table(Domain(attrs), np.hstack(data_parts))

    def generate(self):
        data = self.generate_data()
        self.Outputs.data.send(data)

    def pack_editor_settings(self):
        self.distributions = [(editor.name, editor.pack_settings())
                              for editor in self.editors]

    def unpack_editor_settings(self):
        self.editors = []
        for name, editor_args in self.distributions:
            editor = distributions[name]()
            editor.unpack_settings(editor_args)
            self.add_editor(editor)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWRandomData).run()
