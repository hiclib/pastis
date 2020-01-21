import os
from os.path import dirname
from iced import io


def test_load_counts():
    module_path = dirname(__file__)
    counts_filename = os.path.join(
        module_path,
        "../../datasets/data/duan2009/duan.SC.10000.raw_sub.matrix")

    counts = io.load_counts(counts_filename)
    assert counts is not None
