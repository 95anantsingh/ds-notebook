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
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_togglebutton",
    "sphinxcontrib.mermaid",
    "sphinx_last_updated_by_git",
    "sphinx_sitemap",
    "myst_parser",
]

templates_path = ["custom/templates"]
suppress_warnings = ["toc.not_included", "autosectionlabel.*"]
autosectionlabel_prefix_document = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

mermaid_light_theme = "default"
mermaid_dark_theme = "default"
mermaid_d3_zoom = True

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

# -- Options for HTML output

html_baseurl = "https://95anantsingh.github.io/ds-notebook/"
html_theme = "sphinx_rtd_theme"
# html_logo = "assets/html_logo.png"
html_favicon = "assets/favicon.png"
html_show_sourcelink = False
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
