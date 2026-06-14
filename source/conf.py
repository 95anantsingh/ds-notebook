# -- Project information
project = "Data Science Notebook"
copyright = "2026"
author = "Anant Singh"

release = "1.0"
version = "1.0.0"


# -- Source configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.mermaid",
    "sphinx_last_updated_by_git",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "sphinx_tippy",
    "myst_parser",
]

templates_path = ["custom/templates"]
suppress_warnings = ["toc.not_included", "autosectionlabel.*"]
autosectionlabel_prefix_document = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

mermaid_light_theme = "default"
mermaid_dark_theme = "default"
# mermaid_d3_zoom = True

# sphinx-tippy: hover preview tooltips for cross-references.
# RTD theme renders article content inside .rst-content.
tippy_anchor_parent_selector = ".rst-content"
# Only generate tooltips for internal references — avoid live calls to
# Wikipedia/DOI/etc. at build time (keeps CI fast and network-independent).
tippy_enable_wikitips = False
tippy_enable_doitips = False
tippy_skip_urls = []

myst_enable_extensions = [
    "tasklist",
    "amsmath",
    "attrs_inline",
    "attrs_block",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "gfm_autolink",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
]

# Render task-list checkboxes as interactive (omit the `disabled` attribute).
myst_enable_checkboxes = True

# Generate GitHub-style slug anchors on headings (h1, h2) for #fragment links.
myst_heading_anchors = 2

# Allow $$…$$ display math to appear inline within a paragraph.
myst_dmath_double_inline = True
# Render a transition (horizontal rule) before the footnotes block.
myst_footnote_transition = True

# Shorthand link schemes, e.g. [paper](arxiv:1706.03762), [doi](doi:10.…).
myst_url_schemes = {
    "http": None,
    "https": None,
    "mailto": None,
    "ftp": None,
    "arxiv": "https://arxiv.org/abs/{{path}}",
    "doi": "https://doi.org/{{path}}",
    "wiki": "https://en.wikipedia.org/wiki/{{path}}",
}

# Cross-link into external project docs by symbol (intersphinx).
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# -- Options for HTML output

html_baseurl = "https://95anantsingh.github.io/ds-notebook/"
html_theme = "sphinx_rtd_theme"
# html_logo = "assets/html_logo.png"
html_favicon = "assets/favicon.png"
html_show_sourcelink = False
html_last_updated_fmt = "%b %d, %Y"
html_static_path = ["custom/static"]
html_css_files = ["css/custom.css"]
# Theme options refer - https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#282828",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": False,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}
