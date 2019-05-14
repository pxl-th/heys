#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os.path import (
    abspath,
    join,
)
import sys

project_name = "heys"
sys.path.insert(0, abspath(join("..", "..", project_name)))

autodoc_default_flags = [
    'members',
    'private-members',
    'show-inheritance',
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js"
    "?config=TeX-AMS-MML_HTMLorMML"
)

pygments_style = "sphinx"
source_suffix = ".rst"
master_doc = "index"
language = None

todo_include_todos = True
exclude_patterns = []

htmlhelp_basename = "heysdoc"
html_theme = "sphinx_rtd_theme"

author = "Anton Smirnov"
project_decription = (
    "Implementation of Linear Cryptanalysis on Heys block cipher"
)

latex_documents = [(
    master_doc,
    "hekn.tex",
    project_decription,
)]

man_pages = [(
    master_doc,
    project_name,
    project_decription,
    author,
)]

texinfo_documents = [(
    master_doc,
    project_name,
    project_decription,
    author,
)]

epub_title = project_decription
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ["search.html"]
