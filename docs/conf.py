# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information ------------------------------------------------------
project = "CometSpec"
author = "Baltasar Luco"
copyright = "2026, Baltasar Luco"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
]

# -- plot_directive ----------------------------------------------------------
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 110)]

# -- mathjax ------------------------------------------------------------------
# `\AA` is a text-mode LaTeX command, undefined in MathJax 3 math mode.
# Define it (and a couple of other text-mode shortcuts) as math macros.
mathjax3_config = {
    "tex": {
        "macros": {
            "AA": r"{\mathring{A}}",
        },
    },
}

# -- sphinx-copybutton --------------------------------------------------------
# Strip ">>> ", "... " and shell "$ " prompts when copying.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regex = True
copybutton_only_copy_prompt_lines = False

# -- Syntax highlighting (Pygments) -------------------------------------------
# Furo supports separate light/dark Pygments styles. These distinguish
# keywords, builtins, booleans (True/False/None), function names, class names,
# strings, and numbers with distinct colors.
pygments_style = "tango"
pygments_dark_style = "dracula"

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "none"
napoleon_numpy_docstyle = True
napoleon_numpy_docstring   = True
napoleon_google_docstring  = False   # pick one; mixing is messy
napoleon_use_param         = False  # render Parameters as inline bullets: param (type) – desc
napoleon_use_rtype         = False  # merge return type inline with Returns description
napoleon_use_ivar          = True   # render Attributes as :ivar:
napoleon_attr_annotations  = True   # pick up PEP 526 type hints
napoleon_preprocess_types  = True   # convert type strings to cross-reference links


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Intersphinx mapping ------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output ---------------------------------------------------
html_theme = "furo"
html_title = "CometSpec"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#059669",
        "color-brand-content": "#059669",
        "color-admonition-background": "rgba(5, 150, 105, 0.1)",
        "color-api-name": "#2D6FB8",
        "color-api-pre-name": "#2D6FB8",
        "color-api-background": "#e6e6e6",
        "color-api-background-hover": "#d9d9d9",
        "color-inline-code-background": "#ececec",
        "color-link--hover": "#960505",
    },
    "dark_css_variables": {
        "color-brand-primary": "#10b981",
        "color-brand-content": "#10b981",
        "color-background-primary": "#0D1117",
        "color-background-secondary": "#161B22",
        "color-background-border": "#30363D",
        "color-foreground-primary": "#C9D1D9",
        "color-foreground-secondary": "#DEDFDF",
        "color-admonition-background": "rgba(16, 185, 129, 0.1)",
        "color-api-name": "#7EB8F7",
        "color-api-pre-name": "#7EB8F7",
        "color-api-background": "#3b3b3bb8",
        "color-api-background-hover": "#5c5c5cb8",
        "color-inline-code-background": "#3b3b3bb8",
        "color-link--hover": "#00fff7",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/baltasarluco/CometSpec",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
                "2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49"
                "-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15"
                "-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
                "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 "
                "2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 "
                "2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 "
                '2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 '
                '1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 '
                '0016 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
