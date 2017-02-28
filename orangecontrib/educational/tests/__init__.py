import os
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def suite(loader=None, pattern='test*.py'):
    if loader is None:
        loader = unittest.TestLoader()
    if pattern is None:
        pattern = 'test*.py'

    all_tests = [loader.discover(path, pattern, path)
                 for path in (os.path.join(ROOT_DIR, 'tests'),
                              os.path.join(ROOT_DIR, 'widgets', 'tests'))]
    return unittest.TestSuite(all_tests)


def load_tests(loader, tests, pattern):
    return suite(loader, pattern)


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
