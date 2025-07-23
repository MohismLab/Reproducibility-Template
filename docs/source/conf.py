# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Your-Project"
copyright = "2025, Shangmin Guo"
author = "Shangmin Guo"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_book_theme",
    "myst_parser",
    "sphinx_design",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output
html_title = "Your Docs"
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/MohismLab/Reproducibility-Template",
    "use_repository_button": False,
    "use_issues_button": False,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "home_page_in_toc": True,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# -- Options for MyST-NB
nb_execution_mode = "cache"  # "force", "cache", "off"
nb_execution_timeout = 30  # seconds
nb_execution_allow_errors = False
nb_execution_raise_on_error = True


# -- Options for Markdown input
# jupyter_execute_notebooks = "auto"

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
]
