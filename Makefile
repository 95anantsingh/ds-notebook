SHELL         := /bin/bash
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR      = source
BUILDDIR       = build
ENV_NAME      := $(shell grep '^name:' env.yaml | cut -d' ' -f2)

.SILENT:
.DEFAULT_GOAL := help

#* Getting Started *
.PHONY: help install update
help: ## Show this help message
	@printf "\033[1;34mUsage:\033[0m\n  make [target]\n\n\033[1;34mTargets:\033[0m\n"
	@awk 'BEGIN {FS = ":.*?## "; cat = ""; first = 1} /^#\* .* \*$$/ {cat = substr($$0, 4, length($$0) - 5); next} /^[a-zA-Z_-]+:.*?## / {if (cat) {printf "%s \033[3;35m%s\033[0m\n", (first ? "" : "\n"), cat; cat = ""; first = 0} printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install all dependencies
	conda env update -n $(ENV_NAME) -f env.yaml
	rm -rf "$(SOURCEDIR)/pages/Demo"
	ln -s "../../docs/Demo" "$(SOURCEDIR)/pages/Demo"
	@echo "Linked source/pages/Demo -> docs/Demo"

update: ## Upgrade all dependencies to latest versions
	pip install --upgrade -r requirements.txt
	@echo "All dependencies upgraded!"

#* Build *
.PHONY: html view check clean
html: ## Run indexer and build HTML documentation
	python $(SOURCEDIR)/indexer.py
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

check: ## Verify files are correctly named and TOC is up to date (used in CI)
	python $(SOURCEDIR)/indexer.py --check

view: ## Open built HTML documentation in browser
	open "$(BUILDDIR)/html/index.html"

clean: ## Remove the build directory
	rm -rf "$(BUILDDIR)"
