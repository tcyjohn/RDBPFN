"""Lightweight wrapper for tab2graph preprocess — avoids heavy CLI-level autogluon imports."""
from __future__ import annotations

import sys
import logging
from pathlib import Path

# Add local packages to path BEFORE importing tab2graph internals
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent / "model_pretrain" / "src"))

import dbinfer_bench as dbb
from tab2graph.device import get_device_info
from tab2graph.preprocess import get_rdb_preprocess_class, get_rdb_preprocess_choice
from tab2graph import yaml_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    if len(sys.argv) != 6:
        print(
            "Usage: python run_preprocess.py <dataset_path> <preprocess_name> "
            "<output_path> <config_path> <depth>"
        )
        sys.exit(1)

    dataset_path = Path(sys.argv[1])
    preprocess_name = sys.argv[2]
    output_path = Path(sys.argv[3])
    config_path = Path(sys.argv[4])
    depth = int(sys.argv[5])

    device = get_device_info()
    RDBPreprocessChoice = get_rdb_preprocess_choice()
    choice = RDBPreprocessChoice(preprocess_name)
    preprocess_class = get_rdb_preprocess_class(choice.value)

    if config_path.suffix in (".yaml", ".yml"):
        config = yaml_utils.load_pyd(preprocess_class.config_class, config_path)
    else:
        config = preprocess_class.default_config

    logger.info("Loading data from %s ...", dataset_path)
    dataset = dbb.load_rdb_data(str(dataset_path))

    logger.info("Running %s (depth=%d)...", preprocess_name, depth)
    preprocess_instance = preprocess_class(config)
    preprocess_instance.run(dataset, output_path, device)
    logger.info("Done: %s", output_path)


if __name__ == "__main__":
    main()
