
PYTHON ?= python
CYTHON ?= cython
NOSETESTS ?= nosetests
CTAGS ?= ctags

all:

install:
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
inplace:
	$(PYTHON) setup.py build_ext -i

test: in
	$(NOSETESTS) -s -v pastis

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s -v --with-coverage pastis --cover-package pastis

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

code-analysis:
	flake8 pastis | grep -v __init__ | grep -v external
	pylint -E -i y pastis/ -d E1103,E0611,E1101

