# DS Notebook

A personal data science reference — ML, DL, CV, NLP, system design, and more — built with Sphinx and published to **[95anantsingh.github.io/ds-notebook](https://95anantsingh.github.io/ds-notebook)**.

## Setup

```bash
conda activate notes
make install   # install dependencies
```

## Usage

```bash
make html      # build docs (runs indexer + Sphinx)
make view      # open in browser
make clean     # remove build artifacts
```

Drop a `.md` file into `source/pages/<Topic>/` and run `make html` — the indexer handles renaming and TOC updates automatically.

## Stack

- [Sphinx](https://www.sphinx-doc.org) + [MyST Markdown](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)
- [Read the Docs theme](https://sphinx-rtd-theme.readthedocs.io)
- GitHub Actions → GitHub Pages (deploys on push to `main`)
