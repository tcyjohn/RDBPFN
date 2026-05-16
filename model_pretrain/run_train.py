"""Wrapper to run src.train as a module so relative imports work with torchrun."""
import sys
from pathlib import Path

# Ensure model_pretrain/ is on sys.path so `src` is importable as a package
_pkg_dir = Path(__file__).resolve().parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

if __name__ == "__main__":
    from src import train
    train.main()
