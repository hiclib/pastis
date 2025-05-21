
PYTHON ?= python
PYTEST ?= pytest
CTAGS ?= ctags

all:

install:
	$(PYTHON) -m pip install .

trailing-spaces:
	find pastit -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

clean:
	rm -rf build
	rm -rf sdist
	rm -rf dist
	rm -rf example/*PM*
	rm -rf example/*MDS*
	rm -rf examples/filtering_example/*.png

test:
	$(PYTEST) --showlocals -v pastis --durations=20

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) pastis --showlocals -v --cov=pastis

doc:
	$(MAKE) -C doc html

doc-noplot:
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 pastis | grep -v __init__
	pylint -E -i y pastis/ -d E1103,E0611,E1101
