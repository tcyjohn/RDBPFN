"""Wrapper to run src.eval as a module so relative imports work."""
import sys
from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

if __name__ == "__main__":
    from src import eval
    eval.main()
