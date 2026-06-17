# DS-Notebook

A Sphinx-based data science reference notebook deployed to GitHub Pages at https://95anantsingh.github.io/ds-notebook.

## Stack

- **Sphinx** with MyST Markdown parser and `sphinx_rtd_theme`
- **Python 3.11**, dependencies in `requirements.txt`
- **GitHub Actions** builds and deploys on push to `main`

## Project Structure

```
source/
  conf.py         # Sphinx config
  index.md        # Root TOC (auto-updated by indexer)
  indexer.py      # Auto-renames files and regenerates TOC
  pages/          # Content organized by topic
    Machine Learning/
    Deep Learning/
    Computer Vision/
    Natural Language Processing/
    System_Design/
    Hardware Architecture/
    Interview/
  assets/         # Images, favicon
  custom/         # Templates and CSS overrides
```

## Environment

```bash
conda activate notes   # activate the project Python environment before any commands
```

## Common Commands

```bash
make install   # pip install -r requirements.txt
make html      # Run indexer + build HTML docs
make check     # Dry-run indexer (used in CI); exits 1 if files need renaming
make serve     # Open built docs in browser
make clean     # Remove build/
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

Only the block between those sentinels is rewritten on each run. This block needs must be under the top heading of the page ("#"). Everything outside — the heading, intro prose, custom sections — is preserved. **Never manually edit inside the sentinel block.**

- New subdirectory indexes are created with `# <Section Name>` heading + sentinel block.
- Existing indexes: only the sentinel block is updated; all other content survives.

## Adding Content

1. Drop a `.md` file into the relevant `source/pages/<Topic>/` directory
2. Run `make html` — the indexer renames files and updates the TOC automatically
3. To annotate a subdirectory index, edit anything outside its `<!-- TOC START -->` / `<!-- TOC END -->` block

### Available Components

See [docs/Demo/1-demo.md](docs/Demo/1-demo.md) for a living reference of every component available in this Sphinx setup: admonitions, math (inline and block), code blocks with line numbers/highlights, Mermaid diagrams, toggle buttons, dropdowns, tabs, cards (with and without links), badges, tables (pipe, list-table, HTML spanning), cross-references, figures, block quotes, and definition lists.

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

## CI/CD

- Triggered on pushes to `main` that touch `source/**`, `requirements.txt`, `Makefile`, or `.github/workflows/**`
- Build job runs `make html`; deploy job publishes `build/html` to GitHub Pages
- PRs only run the build job (no deploy)
