import copy
from typing import Optional

import numpy as np
from AnyQt.QtCore import Qt, QAbstractTableModel, QModelIndex, QSize
from AnyQt.QtWidgets import QTableView, QItemDelegate, QLineEdit, QCompleter

from Orange.data import (
    Table,
    Domain,
    ContinuousVariable,
    DiscreteVariable,
    TimeVariable,
)
from Orange.widgets import gui
from Orange.widgets.settings import (
    Setting,
    ContextSetting,
    PerfectDomainContextHandler,
    ContextHandler,
)
from Orange.widgets.utils import vartype
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Output, Input, Msg


DEFAULT_DATA = [[None] * 3 for y in range(3)]


class EditableTableItemDelegate(QItemDelegate):
    def createEditor(self, parent, options, index: QModelIndex):
        model = index.model()  # type: EditableTableModel
        if not model.is_discrete(index.column()):
            return super().createEditor(parent, options, index)

        vals = model.discrete_vals(index.column())
        edit = QLineEdit(parent)
        edit.setCompleter(QCompleter(vals, edit, filterMode=Qt.MatchContains))

        def save():
            if edit.text():
                model.setData(index, edit.text())

        edit.editingFinished.connect(save)
        return edit

    def setEditorData(self, editor, index):
        current_val = index.model().data(index, Qt.DisplayRole)
        editor.setText(current_val)


class EditableTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._table = [[None]]
        self._domain = None

    def _var(self, index):
        # todo: remove if/when domain[idx] is fixed
        return (self._domain.variables + self._domain.metas)[index]

    def set_domain(self, domain: Optional[Domain]):
        self._domain = domain
        if domain is not None:
            # Todo: change to len(domain) when Domain len's behaviour is fixed
            n_columns = len(domain.variables) + len(domain.metas)
        else:
            n_columns = 3
        self.setColumnCount(n_columns)
        self.clear()

    def is_discrete(self, column):
        column_data = set(row[column] for row in self._table) - {None}

        def is_number(x):
            """
            Check if x is number
            x.is_digit only works for usigned ints this on works for ints and
            floats
            """
            try:
                float(x)
                return True
            except ValueError:
                return False

        return (
            self._domain is not None
            and self._var(column).is_discrete
            or (
                column_data
                and not self.is_time_variable(column)
                and not all(map(lambda s: is_number(s), column_data))
            )
        )

    def is_time_variable(self, column):
        values = self.time_vals(column)
        return values and not all(e is None for e in values)

    def time_vals(self, column):
        column_data = [row[column] for row in self._table]
        try:
            tvar = TimeVariable("_")
            values = [
                tvar.parse_exact_iso(d) if d is not None else None
                for d in column_data
            ]
            return values
        except ValueError:
            return None

    def discrete_vals(self, column):
        if self._domain is not None and self._var(column).is_discrete:
            return self._var(column).values
        else:
            return list(set(row[column] for row in self._table) - {None})

    def rowCount(self, parent=None):
        return len(self._table)

    def columnCount(self, parent=None):
        return len(self._table[0])

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsEditable

    def data(self, index: QModelIndex, role=None):
        value = self._table[index.row()][index.column()]
        if role == Qt.DisplayRole and value is not None:
            return str(value)
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignVCenter | Qt.AlignLeft

    def setData(self, index: QModelIndex, value: str, role=None):
        value = None if not value else value
        self._table[index.row()][index.column()] = value
        self.dataChanged.emit(index, index)
        return True

    def set_table(self, table):
        self.beginResetModel()
        # must be a copy since model changes it inplace
        self._table = copy.deepcopy(table)
        self.endResetModel()

    def get_raw_table(self):
        # the model updates the table inplace
        return copy.deepcopy(self._table)

    def setRowCount(self, n_rows):
        diff = n_rows - self.rowCount()
        if diff > 0:
            self.beginInsertRows(QModelIndex(), self.rowCount(), n_rows - 1)
            self._table += [[None] * self.columnCount() for _ in range(diff)]
            self.endInsertRows()
        elif diff < 0:
            self.beginRemoveRows(QModelIndex(), n_rows, self.rowCount() - 1)
            del self._table[n_rows:]
            self.endRemoveRows()

    def setColumnCount(self, n_columns):
        diff = n_columns - self.columnCount()
        if diff > 0:
            self.beginInsertColumns(
                QModelIndex(), self.columnCount(), n_columns - 1)
            for row in self._table:
                row += [None] * diff
            self.endInsertColumns()
        elif diff < 0:
            self.beginRemoveColumns(
                QModelIndex(), n_columns, self.columnCount() - 1)
            for row in self._table:
                del row[n_columns:]
            self.endRemoveColumns()

    def headerData(self, section, orientation, role=None):
        if orientation == Qt.Vertical:
            return super().headerData(section, orientation, role)

        if role == Qt.DisplayRole:
            if self._domain is None:
                return str(section + 1)
            else:
                return self._var(section).name

    def clear(self):
        self.set_table(
            [[None] * self.columnCount()
             for _ in range(self.rowCount())]
        )

    def get_table(self):
        domain = self.get_domain()
        data = np.array(self._table)  # type:
        for ci in range(data.shape[1]):
            if isinstance((domain.variables + domain.metas)[ci], TimeVariable):
                data[:, ci] = self.time_vals(ci)
        return Table.from_list(domain, data)

    def get_domain(self):
        if self._domain is not None:
            return self._domain

        vars = []
        for ci in range(self.columnCount()):
            if self.is_discrete(ci):
                values = set(
                    row[ci] for row in self._table if row[ci] is not None
                )
                var = DiscreteVariable(name=str(ci + 1), values=values)
            elif self.is_time_variable(ci):
                var = TimeVariable(name=str(ci + 1))
            else:
                var = ContinuousVariable(name=str(ci + 1))
            vars.append(var)
        return Domain(vars)


class CreateTableContextHandler(PerfectDomainContextHandler):
    def open_context(self, widget, domain):
        ContextHandler.open_context(
            self, widget, domain, *self.encode_domain(domain)
        )

    def encode_domain(self, domain):
        """
        Encode domain into tuples (name, type)
        A tuple is returned for each of attributes, class_vars and metas.
        """
        if self.match_values == self.MATCH_VALUES_ALL:

            def _encode(attrs):
                return tuple(
                    (v.name, list(v.values) if v.is_discrete else vartype(v))
                    for v in attrs
                )

        else:

            def _encode(attrs):
                return tuple((v.name, vartype(v)) for v in attrs)

        if domain is None:
            return (None, None, None)
        return (
            _encode(domain.attributes),
            _encode(domain.class_vars),
            _encode(domain.metas),
        )


class OWCreateTable(OWWidget):
    name = "Create Table"
    icon = "icons/CreateTable.svg"
    priority = 50
    keywords = []

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    class Error(OWWidget.Error):
        transform_err = Msg("Data does not fit to domain")

    settingsHandler = CreateTableContextHandler()

    n_rows = Setting(len(DEFAULT_DATA))
    n_columns = Setting(len(DEFAULT_DATA[0]))
    auto_commit = Setting(True)
    # since data is a small (at most 20x20) table we can afford to store it
    # as a context
    context_data = ContextSetting(copy.deepcopy(DEFAULT_DATA), schema_only=True)

    def __init__(self):
        super().__init__()

        options = {"labelWidth": 100, "controlWidth": 50}
        box = gui.vBox(self.controlArea, "Control")
        self.r_spin = gui.spin(
            box,
            self,
            "n_rows",
            1,
            20,
            1,
            **options,
            label="Rows:",
            callback=self.nrows_changed
        )
        self.c_spin = gui.spin(
            box,
            self,
            "n_columns",
            1,
            20,
            1,
            **options,
            label="Columns:",
            callback=self.ncolumns_changed
        )
        box.setMinimumWidth(200)

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

        box = gui.vBox(self.mainArea, True, margin=0)
        self.table = QTableView(box)
        self.table.setItemDelegate(EditableTableItemDelegate())
        self.table.setEditTriggers(self.table.CurrentChanged)
        box.layout().addWidget(self.table)

        self.table_model = EditableTableModel()
        self.table.setModel(self.table_model)
        self.table_model.dataChanged.connect(self.data_changed)
        self.table_model.set_table(self.context_data)
        self.set_dataset(None)  # to init the context

    def nrows_changed(self):
        self.table_model.setRowCount(self.n_rows)
        self.commit()

    def ncolumns_changed(self):
        self.table_model.setColumnCount(self.n_columns)
        self.commit()

    def data_changed(self):
        self.context_data = self.table_model.get_raw_table()
        self.commit()

    def commit(self):
        data = None
        try:
            data = self.table_model.get_table()
            self.Error.transform_err.clear()
        except Exception as ex:
            self.Error.transform_err()
        self.Outputs.data.send(data)

    @Inputs.data
    def set_dataset(self, data):
        self.closeContext()
        if data is not None:
            self.table_model.set_domain(data.domain)
            self.context_data = [
                [None] * (len(data.domain.variables) + len(data.domain.metas))
                for _ in range(self.table_model.rowCount())
            ]
        else:
            self.table_model.set_domain(None)
            self.context_data = copy.deepcopy(DEFAULT_DATA)
        self.c_spin.setEnabled(data is None)
        self.c_spin.setValue(self.table_model.columnCount())
        self.openContext(data)
        self.unconditional_commit()

    @staticmethod
    def sizeHint():
        return QSize(800, 500)

    def openContext(self, data):
        super(OWCreateTable, self).openContext(data.domain if data else None)
        self.table_model.set_table(self.context_data)


if __name__ == "__main__":  # pragma: no cover
    WidgetPreview(OWCreateTable).run(Table("iris"))
