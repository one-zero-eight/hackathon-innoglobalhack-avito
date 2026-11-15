"""Индекс полных размеченных описаний официального обучающего набора."""

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from main import MICROCATEGORIES
from run import normalize_description

REFERENCE_SOURCE_MC_ID = 101
INDEX_SPEC = {
    "analyzer": "char_wb",
    "ngram_range": [3, 5],
    "min_df": 2,
    "max_features": 40000,
    "norm": "l2",
    "similarity": "cosine",
    "tie_break": "original_jsonl_row_ascending",
}


@dataclass(frozen=True)
class Reference:
    """Полный исходный текст и дополнительные категории одной строки train."""

    row: int
    description: str
    normalized: str
    labels: frozenset[int]


@dataclass(frozen=True)
class ReferenceIndex:
    """Один переиспользуемый индекс с происхождением исходных данных."""

    references: tuple[Reference, ...]
    vectorizer: TfidfVectorizer
    matrix: csr_matrix
    sha256: str

    def nearest_pairs(self, description: str, categories: set[int]) -> dict[int, tuple[int, int, float, float]]:
        """Найти по одному соседу каждой метки, исключив точное совпадение."""
        query = self.vectorizer.transform([description])
        similarities = (self.matrix @ query.T).toarray().ravel()
        normalized = normalize_description(description)
        eligible = [
            i for i, reference in enumerate(self.references)
            if reference.normalized != normalized
        ]
        pairs: dict[int, tuple[int, int, float, float]] = {}
        for category in sorted(categories):
            positives = [i for i in eligible if category in self.references[i].labels]
            negatives = [i for i in eligible if category not in self.references[i].labels]
            if not positives or not negatives:
                continue
            positive = max(positives, key=lambda i: (similarities[i], -self.references[i].row))
            negative = max(negatives, key=lambda i: (similarities[i], -self.references[i].row))
            pairs[category] = (
                positive, negative, float(similarities[positive]), float(similarities[negative]),
            )
        return pairs


@lru_cache(maxsize=4)
def load_reference_index(path: str, mtime_ns: int, max_chars: int) -> ReferenceIndex:
    """Построить индекс один раз на путь, версию файла и лимит длины."""
    source = Path(path)
    if source.stat().st_mtime_ns != mtime_ns:
        raise ValueError("Файл train изменился до построения retrieval-индекса")
    raw = source.read_bytes()
    references: list[Reference] = []
    for row_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"Train, строка {row_number}: ожидается JSON-объект")
        description = record.get("description")
        labels = record.get("found_microcategories")
        source_category = record.get("mcId", REFERENCE_SOURCE_MC_ID)
        if type(source_category) is not int or source_category != REFERENCE_SOURCE_MC_ID:
            raise ValueError(f"Train, строка {row_number}: ожидается исходная категория 101")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Train, строка {row_number}: отсутствует полное описание")
        if not isinstance(labels, list):
            raise ValueError(f"Train, строка {row_number}: требуется found_microcategories")
        category_ids: set[int] = set()
        for label in labels:
            category = label.get("mcId") if isinstance(label, dict) else None
            if type(category) is not int or category not in MICROCATEGORIES or category == source_category:
                raise ValueError(f"Train, строка {row_number}: некорректный дополнительный mcId")
            category_ids.add(category)
        if len(description) <= max_chars:
            references.append(Reference(
                row=row_number,
                description=description,
                normalized=normalize_description(description),
                labels=frozenset(category_ids),
            ))
    if source.stat().st_mtime_ns != mtime_ns:
        raise ValueError("Файл train изменился при построении retrieval-индекса")
    if not references:
        raise ValueError("Нет полных train-описаний в пределах JEV_REFERENCE_MAX_CHARS")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40000, norm="l2",
    )
    matrix = vectorizer.fit_transform(reference.description for reference in references).tocsr()
    return ReferenceIndex(tuple(references), vectorizer, matrix, hashlib.sha256(raw).hexdigest())
