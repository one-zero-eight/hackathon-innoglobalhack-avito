"""Выделение категорий по признакам, выученным из размеченных объявлений."""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from main import BasePipeline, MICROCATEGORIES, PipelineInput

DEFAULT_ARTIFACT = Path(__file__).resolve().parents[3] / "out/learned-keywords.json"


class SubcategoryExtractionPipeline(BasePipeline):
    """Применить веса TF-IDF и бинарных совпадений с выученными фразами."""

    name = "learned_keywords"

    def __init__(self) -> None:
        path = Path(os.getenv("LEARNED_KEYWORDS_ARTIFACT", str(DEFAULT_ARTIFACT)))
        if not path.is_file():
            raise ValueError(f"Нет артефакта learned_keywords: {path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        vocabulary = artifact["vocabulary"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, vocabulary=vocabulary,
        )
        self.vectorizer.idf_ = np.asarray(artifact["idf"], dtype=float)
        self.weights = np.asarray(artifact["weights"], dtype=float)
        self.keyword_weights = np.asarray(artifact["keyword_weights"], dtype=float)
        self.intercepts = np.asarray(artifact["intercepts"], dtype=float)
        if (
            artifact["version"] != 1
            or artifact["categories"] != list(MICROCATEGORIES)
            or self.weights.shape != (len(MICROCATEGORIES), len(vocabulary))
            or self.keyword_weights.shape != self.weights.shape
            or self.intercepts.shape != (len(MICROCATEGORIES),)
            or self.vectorizer.idf_.shape != (len(vocabulary),)
            or not all(np.isfinite(values).all() for values in (
                self.weights, self.keyword_weights, self.intercepts, self.vectorizer.idf_,
            ))
        ):
            raise ValueError("Некорректный артефакт learned_keywords")

    def run(self, inp: PipelineInput) -> list[int]:
        """Вернуть дополнительные категории с вероятностью от 0.5."""
        features = self.vectorizer.transform([inp.description])
        logits = features @ self.weights.T + features.sign() @ self.keyword_weights.T + self.intercepts
        return [cid for cid, logit in zip(MICROCATEGORIES, logits.ravel(), strict=True)
                if cid != inp.mc_id and logit >= 0]
