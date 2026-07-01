# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# DS-Notebook

A Sphinx-based data science reference notebook deployed to GitHub Pages at https://95anantsingh.github.io/ds-notebook.

## Stack

- **Sphinx** with MyST Markdown parser and `sphinx_rtd_theme`
- **myst_nb** executes Jupyter notebooks at build time (`nb_execution_mode = "cache"`)
- **Python 3.11**, conda environment defined in `env.yaml`, pinned deps in `requirements.txt`
- **GitHub Actions** builds and deploys on push to `main`

## Environment

```bash
conda activate notes   # activate the project Python environment before any commands
```

## Common Commands

```bash
make install   # conda env update from env.yaml + link source/pages/Demo -> docs/Demo
make update    # pip upgrade all deps to latest
make html      # Run indexer + build HTML docs (notebooks execute with caching)
make dev       # sphinx-autobuild: live-reload preview at localhost:8000
make check     # Dry-run indexer (used in CI); exits 1 if files need renaming
make serve     # Open built docs in browser
make clean     # Remove build/
```

## Project Structure

```
source/
  conf.py         # Sphinx config; prepends source/_lib to PYTHONPATH for notebooks
  index.md        # Root TOC (auto-updated by indexer)
  indexer.py      # Auto-renames files and regenerates TOC
  _lib/           # Python utilities importable from notebook code cells
    plotly_utils.py   # Reusable Plotly matrix-visualisation library
  pages/          # Content organized by topic (Demo/ is a symlink to docs/Demo/)
    Machine Learning/
    Deep Learning/
    Computer Vision/
    Natural Language Processing/
    System_Design/
    Hardware Architecture/
    Interview/
  assets/         # Images, favicon
  custom/         # Templates and CSS overrides
docs/
  Demo/           # Component reference and demos (symlinked into source/pages/Demo)
```

## Indexer Behavior (`source/indexer.py`)

- Recursively renames `.md` files in `source/pages/` with zero-padded numeric prefixes (`01-topic.md`)
- Converts filenames to kebab-case
- Auto-generates `index.md` files for subdirectories
- Updates the `<!-- TOC START -->` / `<!-- TOC END -->` block in both the root `source/index.md` **and** every subdirectory `index.md`
- Run `make check` to validate without making changes (CI uses this)

### index.md sentinel pattern (applies everywhere)

Both the root `source/index.md` and all subdirectory `index.md` files use the same sentinel comments:

```
<!-- TOC START -->
…toctree managed by indexer…
<!-- TOC END -->
```

Only the block between those sentinels is rewritten on each run. This block must be under the top heading of the page (`#`). Everything outside — the heading, intro prose, custom sections — is preserved. **Never manually edit inside the sentinel block.**

- New subdirectory indexes are created with `# <Section Name>` heading + sentinel block.
- Existing indexes: only the sentinel block is updated; all other content survives.

## Adding Content

### Markdown pages

1. Drop a `.md` file into the relevant `source/pages/<Topic>/` directory
2. Run `make html` — the indexer renames files and updates the TOC automatically
3. To annotate a subdirectory index, edit anything outside its `<!-- TOC START -->` / `<!-- TOC END -->` block

### Jupyter notebooks

Notebooks (`.ipynb`) in `source/pages/` are executed by `myst_nb` at build time with result caching. To embed interactive Plotly figures from a notebook into a Markdown page, use `myst_nb`'s glue mechanism:

```python
# In a notebook code cell (the notebook should be an orphan or linked page):
from plotly_utils import figure, show, matmul, Matrix
import numpy as np

A = Matrix(np.array([[1, 2], [3, 4]]), "A")
B = Matrix(np.array([[5, 6], [7, 8]]), "B")
glue("my_plot", show(figure(matmul(A, B), animate=True)), display=False)
```

```markdown
<!-- In the .md page that embeds it: -->
{glue:}`my_plot`
```

`plotly_utils.py` is auto-importable in all notebook cells because `conf.py` prepends `source/_lib` to `PYTHONPATH`. Always use `include_plotlyjs="cdn"` (the default in `show()`) — **never** `require.js`, which conflicts with the RTD theme.

## Available Components

See [docs/Demo/1-demo.md](docs/Demo/1-demo.md) for a living reference of every component: admonitions, math (inline and block), code blocks with line numbers/highlights, Mermaid diagrams, toggle buttons, dropdowns, tabs, cards (with and without links), badges, tables (pipe, list-table, HTML spanning), cross-references, figures, block quotes, and definition lists.

For Mermaid diagrams, use the `mermaid` skill — it generates well-structured diagrams with vertical top-to-bottom layouts.

### MyST Directive Nesting Rule

Inner directives must use **fewer** colons than their parent. When you need to nest a `:::` directive inside another, bump the parent up — not the child:

```
:::::{grid}          ← 5 colons (outermost)
::::{grid-item}      ← 4 colons
:::{dropdown}        ← 3 colons (innermost)
:::
::::
:::::
```

Never increase a nested directive's colon count to resolve conflicts — always increase the parent's count instead.

### Shorthand URL schemes

MyST supports these in link targets:

```markdown
[Attention is All You Need](arxiv:1706.03762)
[DOI link](doi:10.1145/12345)
[Wikipedia](wiki:Transformer_(deep_learning_architecture))
```

## CI/CD

- Triggered on pushes to `main` that touch `source/**`, `requirements.txt`, `Makefile`, or `.github/workflows/**`
- Build job runs `make html`; deploy job publishes `build/html` to GitHub Pages
- PRs only run the build job (no deploy)
