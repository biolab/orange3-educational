import csv
import logging
import re
from io import StringIO
from itertools import chain, repeat
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import numpy as np

from AnyQt.QtCore import QTimer, Qt, QPoint
from AnyQt.QtGui import QValidator
from AnyQt.QtWidgets import QApplication, QComboBox, QLabel, QToolTip, QStyledItemDelegate

from Orange.data.io import TabReader
from Orange.util import try_
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.itemmodels import PyListModel
from Orange.widgets import widget, gui, settings
from Orange.data import Table
from Orange.widgets.utils.signals import Output
from Orange.widgets.utils.webview import WebviewWidget
try:
    from Orange.widgets.utils.webview import wait
except ImportError:  # WebKit and before wait() was toplevel
    import time
    from AnyQt.QtWidgets import qApp
    from AnyQt.QtCore import QEventLoop

    def wait(until: callable, timeout=5000):
        started = time.clock()
        while not until():
            qApp.processEvents(QEventLoop.ExcludeUserInputEvents)
            if (time.clock() - started) * 1000 > timeout:
                raise TimeoutError()


log = logging.getLogger(__name__)


class DataEmptyError(Exception):
    pass

class DataIsAnalError(Exception):
    pass


class URLComboBox(QComboBox):

    class TitleShowingPopupDelegate(QStyledItemDelegate):
        TitleRole = Qt.UserRole + 1

        def displayText(self, url, _):
            i = self.parent().findText(url, Qt.MatchExactly)
            model = self.parent().model()
            title = model.data(model.index(i, 0), self.TitleRole)
            return ('{0} ({1})' if title else '{1}').format(title, url)

    class Validator(QValidator):
        def validate(self, input, pos):
            if is_valid_url(input):
                return QValidator.Acceptable, input, pos
            return QValidator.Intermediate, input, pos

    def __init__(self, parent, model_list, **kwargs):
        super().__init__(parent, **kwargs)
        self.setValidator(self.Validator())
        self.setModel(PyListModel(iterable=model_list,
                                  flags=Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable,
                                  parent=self))
        self.view().setItemDelegate(self.TitleShowingPopupDelegate(self))

    def setTitleFor(self, i, title):
        self.model().setData(self.model().index(i, 0), title,
                             self.TitleShowingPopupDelegate.TitleRole)


VALID_URL_HELP = 'https://www.1ka.si/podatki/<SURVEY_ID>/<ACCESS_KEY>/'
VALID_URL_PATH_REGEX = re.compile(r'/podatki/\d+/[\dA-F]{6,10}/?$')


def is_valid_url(url):
    # Only match path as hosting domain can be arbitrary
    return bool(VALID_URL_PATH_REGEX.match(urlparse(url).path))


class OW1ka(widget.OWWidget):
    name = "EnKlik Anketa"
    description = "Import data from EnKlikAnketa (1ka.si) public URL."
    keywords = ["1ka", "load data", "load survey", "survey"]
    icon = "icons/1ka.svg"
    priority = 200

    class Outputs:
        data = Output("Data", Table)

    want_main_area = False
    resizing_enabled = False

    settingsHandler = settings.PerfectDomainContextHandler(
        match_values=settings.PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    recent = settings.Setting([])
    reload_idx = settings.Setting(0)
    autocommit = settings.Setting(True)
    domain_editor = settings.SettingProvider(DomainEditor)

    UserAdviceMessages = [
        widget.Message(
            'You can import data from public links to 1ka surveys results. '
            'Click to learn more on how to get a shareable public link URL for '
            '1ka surveys that you manage.',
            'public-link',
            icon=widget.Message.Information,
            moreurl='http://english.1ka.si/db/24/468/Guides/Public_link_to_access_data_and_analysis/'
        ),
    ]

    class Error(widget.OWWidget.Error):
        net_error = widget.Msg("Couldn't load data: {}. Ensure network connection, firewall ...")
        parse_error = widget.Msg("Couldn't parse data: {}. Ensure well-formatted data or submit a bug report.")
        invalid_url = widget.Msg('Invalid URL. Public shareable link should match: ' + VALID_URL_HELP)
        data_is_anal = widget.Msg("The provided URL is a public link to 'Analysis'. Need public link to 'Data'.")

    class Information(widget.OWWidget.Information):
        response_data_empty = widget.Msg('Response data is empty. Get some responses first.')

    def __init__(self):
        super().__init__()
        self.table = None
        self._html = None

        def _loadFinished(is_ok):
            if is_ok:
                QTimer.singleShot(1, lambda: setattr(self, '_html', self.webview.html()))

        self.webview = WebviewWidget(loadFinished=_loadFinished)

        vb = gui.vBox(self.controlArea, 'Import Data')
        hb = gui.hBox(vb)
        self.combo = combo = URLComboBox(
            hb, self.recent, editable=True, minimumWidth=400,
            insertPolicy=QComboBox.InsertAtTop,
            toolTip='Format: ' + VALID_URL_HELP,
            editTextChanged=self.is_valid_url,
            # Indirect via QTimer because calling wait() -> processEvents,
            # while our currentIndexChanged event hadn't yet finished.
            # Avoids calling handler twice.
            currentIndexChanged=lambda: QTimer.singleShot(1, self.load_url))
        hb.layout().addWidget(QLabel('Public link URL:', hb))
        hb.layout().addWidget(combo)
        hb.layout().setStretch(1, 2)

        RELOAD_TIMES = (
            ('No reload',),
            ('5 s', 5000),
            ('10 s', 10000),
            ('30 s', 30000),
            ('1 min', 60*1000),
            ('2 min', 2*60*1000),
            ('5 min', 5*60*1000),
        )

        reload_timer = QTimer(self, timeout=lambda: self.load_url(from_reload=True))

        def _on_reload_changed():
            if self.reload_idx == 0:
                reload_timer.stop()
                return
            reload_timer.start(RELOAD_TIMES[self.reload_idx][1])

        gui.comboBox(vb, self, 'reload_idx', label='Reload every:',
                     orientation=Qt.Horizontal,
                     items=[i[0] for i in RELOAD_TIMES],
                     callback=_on_reload_changed)

        box = gui.widgetBox(self.controlArea, "Columns (Double-click to edit)")
        self.domain_editor = DomainEditor(self)
        editor_model = self.domain_editor.model()

        def editorDataChanged():
            self.apply_domain_edit()
            self.commit()

        editor_model.dataChanged.connect(editorDataChanged)
        box.layout().addWidget(self.domain_editor)

        box = gui.widgetBox(self.controlArea, "Info", addSpace=True)
        info = self.data_info = gui.widgetLabel(box, '')
        info.setWordWrap(True)

        self.controlArea.layout().addStretch(1)
        gui.auto_commit(self.controlArea, self, 'autocommit', label='Commit')

        self.set_info()

    def set_combo_items(self):
        self.combo.clear()
        for sheet in self.recent:
            self.combo.addItem(sheet.name, sheet.url)

    def commit(self):
        self.Outputs.data.send(self.table)

    def is_valid_url(self, url):
        if is_valid_url(url):
            self.Error.invalid_url.clear()
            return True
        self.Error.invalid_url()
        QToolTip.showText(self.combo.mapToGlobal(QPoint(0, 0)), self.combo.toolTip())

    def load_url(self, from_reload=False):
        self.closeContext()
        self.domain_editor.set_domain(None)

        url = self.combo.currentText()
        if not self.is_valid_url(url):
            self.table = None
            self.commit()
            return

        if url not in self.recent:
            self.recent.insert(0, url)

        prev_table = self.table
        with self.progressBar(3) as progress:
            try:
                self._html = None
                self.webview.setUrl(url)
                wait(until=lambda: self._html is not None)
                progress.advance()
                # Wait some seconds for discrete labels to have loaded via AJAX,
                # then re-query HTML.
                # *Webview.loadFinished doesn't guarantee it sufficiently
                try:
                    wait(until=lambda: False, timeout=1200)
                except TimeoutError:
                    pass
                progress.advance()
                html = self.webview.html()
            except Exception as e:
                log.exception("Couldn't load data from: %s", url)
                self.Error.net_error(try_(lambda: e.args[0], ''))
                self.table = None
            else:
                self.Error.clear()
                self.Information.clear()
                self.table = None
                try:
                    table = self.table = self.table_from_html(html)
                except DataEmptyError:
                    self.Information.response_data_empty()
                except DataIsAnalError:
                    self.Error.data_is_anal()
                except Exception as e:
                    log.exception('Parsing error: %s', url)
                    self.Error.parse_error(try_(lambda: e.args[0], ''))
                else:
                    self.openContext(table.domain)
                    self.combo.setTitleFor(self.combo.currentIndex(), table.name)

        def _equal(data1, data2):
            NAN = float('nan')
            return (try_(lambda: data1.checksum(), NAN) ==
                    try_(lambda: data2.checksum(), NAN))

        self._orig_table = self.table
        self.apply_domain_edit()

        if not (from_reload and _equal(prev_table, self.table)):
            self.commit()

    def apply_domain_edit(self):
        data = self._orig_table
        if data is None:
            self.set_info()
            return

        domain, cols = self.domain_editor.get_domain(data.domain, data)

        # Copied verbatim from OWFile
        if not (domain.variables or domain.metas):
            table = None
        else:
            X, y, m = cols
            table = Table.from_numpy(domain, X, y, m, data.W)
            table.name = data.name
            table.ids = np.array(data.ids)
            table.attributes = getattr(data, 'attributes', {})

        self.table = table
        self.set_info()

    DATETIME_VAR = 'Paradata (insert)'

    def table_from_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        try:
            html_table = soup.find_all('table')[-1]
        except IndexError:
            raise DataEmptyError

        if '<h2>Anal' in html or 'div_analiza_' in html:
            raise DataIsAnalError

        def _header_row_strings(row):
            return chain.from_iterable(
                repeat(th.get_text(), int(th.get('colspan') or 1))
                for th in html_table.select('thead tr:nth-of-type(%d) th[title]' % row))

        # self.DATETIME_VAR (available when Paradata is enabled in 1ka UI)
        # should match this variable name format
        header = [th1.rstrip(':') + ('' if th3 == th1 else ' ({})').format(th3.rstrip(':'))
                  for th1, th3 in zip(_header_row_strings(1),
                                      _header_row_strings(3))]
        values = [[(# If no span, feature is a number or a text field
                    td.get_text() if td.span is None else
                    # If have span, it's a number, but if negative, replace with NaN
                    '' if td.contents[0].strip().startswith('-') else
                    # Else if span, the number is its code, but we want its value
                    td.span.get_text()[1:-1])
                   for td in tr.select('td')
                   if 'data_uid' not in td.get('class', ())]
                  for tr in html_table.select('tbody tr')]

        # Save parsed values into in-mem file for default values processing
        buffer = StringIO()
        writer = csv.writer(buffer, delimiter='\t')
        writer.writerow(header)
        writer.writerows(values)
        buffer.flush()
        buffer.seek(0)

        data = TabReader(buffer).read()

        title = soup.select('body h2:nth-of-type(1)')[0].get_text().split(': ', maxsplit=1)[-1]
        data.name = title

        return data

    def set_info(self):
        data = self.table
        if data is None:
            self.data_info.setText('No spreadsheet loaded.')
            return
        text = "{}\n\n{} instance(s), {} feature(s), {} meta attribute(s)\n".format(
            data.name, len(data), len(data.domain.attributes), len(data.domain.metas))
        text += try_(lambda: '\nFirst entry: {}'
                             '\nLast entry: {}'.format(data[0, self.DATETIME_VAR],
                                                       data[-1, self.DATETIME_VAR]), '')
        self.data_info.setText(text)


if __name__ == "__main__":
    a = QApplication([])
    ow = OW1ka()
    ow.combo.setEditText('https://www.1ka.si/podatki/139234/A4228E24/')
    ow.show()
    a.exec_()
