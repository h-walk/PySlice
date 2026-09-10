"""Sphinx configuration for the PySlice documentation.

Builds the five ecosystem sections (Guides, Example Notebooks, AI Tools, Into
the SEA-weeds, API Reference) from Markdown via MyST, with autodoc pointed at
``src/pyslice``. Heavy optional dependencies are mocked so the API reference
builds without a GPU, OVITO, or ML-potential weights present.
"""
from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

project = "PySlice"
author = "The pySEA ecosystem"
copyright = f"{date.today().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Optional/native dependencies that must not gate a docs build.
autodoc_mock_imports = [
    "ovito",
    "torch",
    "torchvision",
    "orb_models",
    "fairchem",
    "mcp",
    "h5py",
    "tqdm",
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# New code is NumPy-style (the repository rule), but much of the pre-existing
# simulation code uses Google-style "Args:" blocks. Enable both so legacy
# docstrings render instead of breaking the build as malformed definition lists.
napoleon_numpy_docstring = True
napoleon_google_docstring = True

templates_path = ["_templates"]
# Research plans remain in place, but are not part of the published manual.
exclude_patterns = ["_build", "plans/**", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# Fetching intersphinx inventories needs network egress. Sphinx emits one
# unsuppressable warning per unreachable inventory, which fails a ``-W`` build
# for reasons that have nothing to do with this documentation. Set
# ``PYSLICE_DOCS_OFFLINE=1`` to drop the mappings so the strict build is
# verifiable on an air-gapped machine; cross-project links become plain text.
if os.environ.get("PYSLICE_DOCS_OFFLINE") == "1":
    intersphinx_mapping = {}

# A missing sea-eco must not fail the build; it is an optional extra.
nitpicky = False
