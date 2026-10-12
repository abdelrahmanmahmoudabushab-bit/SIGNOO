"""Puts `ml/` on the import path, the way running `python ml/x.py` does.

The scripts in `ml/` import their siblings by bare name (`from data import ...`), so
the tests have to see them the same way.
"""
import sys
from pathlib import Path

ML = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML.parent))      # for `core`, which the scripts also add themselves
sys.path.insert(0, str(ML))
