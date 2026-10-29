"""Run the suite from a bare checkout: put the package root on the path.

`pytest tests/` does not add the repository root to `sys.path` (the tests live in a directory
without `__init__.py`), so `import regmem` would fail unless the package is installed. Adding the
root here keeps `pytest`, `python -m pytest` and an installed package all working.
"""
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
