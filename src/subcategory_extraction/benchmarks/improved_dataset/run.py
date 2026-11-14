"""Micro-F1 на исправленном датасете; общий runner двух F1-сценариев."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from subcategory_extraction.benchmarks.classifier_methods.run import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main(Path(__file__).resolve().parent, "improved_dataset"))
