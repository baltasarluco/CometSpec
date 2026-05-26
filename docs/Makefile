# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build
WATCHDIRS     = --watch ../cometspec

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" $(WATCHDIRS) $(SPHINXOPTS) $(O)

.PHONY: help livehtml Makefile

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
