# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — docs/conf.py                                     ║
# ║  « Sphinx configuration »                                        ║
# ╚══════════════════════════════════════════════════════════════════╝
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

project = "pyLACEpostHoc"
author = "Bart R. H. Geurten"
copyright = "2026, Bart R. H. Geurten"

try:
    release = importlib.metadata.version("pyLACEpostHoc")
except importlib.metadata.PackageNotFoundError:
    release = "0.0.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Heavy / optional backends are mocked so docs build on a bare runner.
autodoc_mock_imports = [
    "cv2",
    "pims",
    "vidstab",
    "neo",
    "quantities",
    "tables",
    "pyarrow",
    "fastparquet",
    "dill",
    "lifelines",
    "mlxtend",
    "lxml",
    "PIL",
    "seaborn",
    "scipy",
    "matplotlib",
    "pandas",
    "numpy",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "pyLACEpostHoc"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}
