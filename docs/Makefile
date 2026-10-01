# Makefile for Sphinx documentation

# Configuration
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Default target
.PHONY: help clean html linkcheck doctest spelling

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) \
		-b html \
		-W \
		--keep-going \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/html" \
		$(SPHINXOPTS) $(O)

linkcheck:
	@$(SPHINXBUILD) \
		-b linkcheck \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/linkcheck" \
		$(SPHINXOPTS) $(O)

doctest:
	@$(SPHINXBUILD) \
		-b doctest \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/doctest" \
		$(SPHINXOPTS) $(O)

spelling:
	@$(SPHINXBUILD) \
		-b spelling \
		"$(SOURCEDIR)" \
		"$(BUILDDIR)/spelling" \
		$(SPHINXOPTS) $(O)

# Catch-all target
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
