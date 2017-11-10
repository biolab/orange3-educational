#!/usr/bin/env python

from os import path, walk

import sys
from setuptools import setup, find_packages

NAME = "Orange3-Educational"

VERSION = "0.1.4"

DESCRIPTION = "Orange Educational add-on for Orange data mining software package."
LONG_DESCRIPTION = open(path.join(path.dirname(__file__), 'README.md')).read()

LICENSE = "BSD"

KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    'orange3 add-on',
)

PACKAGES = find_packages()

PACKAGE_DATA = {
    'orangecontrib.educational': ['tutorials/*.ows'],
    'orangecontrib.educational.widgets': ['icons/*', 'resources/*'],
    'orangecontrib.educational.widgets.highcharts': ['_highcharts/*'],
}

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    'Orange3',
]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    'orange3.addon': (
        'educational = orangecontrib.educational',
    ),
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    'orange.widgets.tutorials': (
        # Syntax: any_text = path.to.package.containing.tutorials
        'educationaltutorials = orangecontrib.educational.tutorials',
    ),

    # Entry point used to specify packages containing widgets.
    'orange.widgets': (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        'Educational = orangecontrib.educational.widgets',
    ),

    # Register widget help
    "orange.canvas.help": (
        'html-index = orangecontrib.educational.widgets:WIDGET_HELP_PATH',)
}

NAMESPACE_PACKAGES = ["orangecontrib"]


def _discover_tests():
    import unittest
    return unittest.defaultTestLoader.discover('orangecontrib.educational',
                                               pattern='test_*.py',
                                               top_level_dir='.')

TEST_SUITE = "setup._discover_tests"

AUTHOR = 'Bioinformatics Laboratory, FRI UL'
AUTHOR_EMAIL = 'contact@orange.biolab.si'
URL = "https://github.com/biolab/orange3-educational"
DOWNLOAD_URL = "https://github.com/biolab/orange3-educational/releases"

def include_documentation(local_dir, install_dir):
    global DATA_FILES
    if 'bdist_wheel' in sys.argv and not path.exists(local_dir):
        print("Directory '{}' does not exist. "
              "Please build documentation before running bdist_wheel."
              .format(path.abspath(local_dir)))
        sys.exit(0)

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append((dirpath.replace(local_dir, install_dir),
                          [path.join(dirpath, f) for f in files]))
    DATA_FILES.extend(doc_files)


if __name__ == '__main__':
    include_documentation('doc/build/html', 'help/orange3-educational')
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=NAMESPACE_PACKAGES,
        test_suite=TEST_SUITE,
        include_package_data=True,
        zip_safe=False,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers = []
    )
