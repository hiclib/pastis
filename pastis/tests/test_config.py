from pastis import config
import os


def test_parse():
    config.parse()
    options = config.parse(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "config.ini")))


def test_get_default_options():
    options = config.get_default_options()
