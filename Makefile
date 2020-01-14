
PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

all:

install: cython
	$(PYTHON) setup.py install

trailing-spaces:
	find pastit -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

clean:
	rm -rf build
	rm -rf sdist
	rm -rf dist
	rm -rf example/*PM*
	rm -rf example/*MDS*

in: inplace # just a shortcut
inplace: cython
	$(PYTHON) setup.py build_ext -i

test: in
	$(PYTEST) --showlocals -v pastis --durations=20

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) pastis --showlocals -v --cov=pastis

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 pastis | grep -v __init__ | grep -v external
	pylint -E -i y pastis/ -d E1103,E0611,E1101

cython:
	find pastis -name "*.pyx" -exec $(CYTHON) {} \;
