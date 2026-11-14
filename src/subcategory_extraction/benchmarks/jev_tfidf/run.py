"""Сравнить Jev, TF-IDF и их среднее по общему протоколу классификаторов."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from subcategory_extraction.benchmarks.classifier_methods.run import main


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(
        default_methods=("jev", "tfidf", "jev_tfidf"),
        default_report=Path(__file__).with_name("results.json"),
    )
