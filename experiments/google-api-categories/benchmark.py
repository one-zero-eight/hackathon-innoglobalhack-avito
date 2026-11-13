"""Сравнение промптов Gemini Decision 1 на объединённом train.csv."""

import argparse
import csv
import json
import logging
import os
from collections import deque
from time import monotonic, sleep
from pathlib import Path
from random import Random
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASET = ROOT / "local-notes/joined-datasets/train.csv"
MODEL = "gemini-decision-1"
THRESHOLD = 0.5
CATEGORIES = {
    101: ("Ремонт под ключ", "комплексный ремонт помещения в несколько этапов"),
    102: ("Штукатурные работы", "штукатурка, выравнивание штукатуркой, штукатурка откосов"),
    103: ("Плиточные работы", "укладка плитки, керамогранита или мозаики, запил и затирка швов"),
    104: ("Сантехника", "водоснабжение, канализация, сантехприборы, водяное отопление"),
    105: ("Электрика", "проводка, розетки, щиты, светильники, электрический тёплый пол"),
    106: ("Поклейка обоев и малярные работы", "обои, окраска, подготовка под обои или краску"),
}
logger = logging.getLogger(__name__)


def sample_rows(dataset: Path, size: int, seed: int) -> list[dict[str, str]]:
    """Выбрать строки с разными описаниями по group_id."""
    with dataset.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    Random(seed).shuffle(rows)
    groups: dict[str, dict[str, str]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], row)
    if not 1 <= size <= len(groups):
        raise ValueError(f"Размер выборки должен быть от 1 до {len(groups)}")
    return list(groups.values())[:size]


class TokenBudget:
    """Ограничить входные токены в скользящем окне."""

    limit = 45_000

    def __init__(self) -> None:
        self.last_actual: int | None = None
        self.requests: deque[tuple[float, int]] = deque()

    def reserve(self, estimate: int) -> None:
        if estimate > self.limit:
            raise ValueError("Один запрос превышает бюджет входных токенов")
        while True:
            now = monotonic()
            while self.requests and now - self.requests[0][0] >= 60:
                self.requests.popleft()
            if sum(tokens for _, tokens in self.requests) + estimate <= self.limit:
                self.requests.append((now, estimate))
                return
            sleep(max(0, 60 - (now - self.requests[0][0])))

    def reconcile(self, actual: int) -> None:
        started, _ = self.requests.pop()
        self.requests.append((started, actual))


def predict(api_key: str, prompt: str, description: str, source: int,
            budget: TokenBudget, categories: set[int] | None = None) -> dict[int, float]:
    """Получить вероятность для каждой категории, кроме исходной."""
    questions = {
        f"mc{mcid}": {
            "type": "noul",
            "instructions": prompt.format(category=category, examples=examples),
        }
        for mcid, (category, examples) in CATEGORIES.items()
        if mcid != source and (categories is None or mcid in categories)
    }
    payload = json.dumps({
        "model": MODEL,
        "input": [{"type": "text", "text": description}],
        "questions": questions,
    }, ensure_ascii=False).encode()
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/decisions",
        data=payload,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    for attempt in range(3):
        budget.reserve(len(payload) + 256)
        try:
            with urlopen(request, timeout=30) as response:
                result = json.load(response)
            break
        except HTTPError as exc:
            if exc.code not in (429, 503) or attempt == 2:
                raise
            sleep(60)
    actual = result.get("usage", {}).get("total_input_tokens")
    budget.last_actual = actual if isinstance(actual, int) and actual > 0 else None
    if budget.last_actual is not None:
        budget.reconcile(budget.last_actual)
    answers = result["answers"]

    if set(answers) != set(questions):
        raise ValueError("Набор ответов Gemini не совпадает с вопросами")
    probabilities = {}
    for name, answer in answers.items():
        probability = answer["noul"]
        if type(probability) not in (float, int) or not 0 <= probability <= 1:
            raise ValueError(f"Некорректная вероятность для {name}")
        probabilities[int(name[2:])] = probability
    return probabilities


def score(expected: list[set[int]], predicted: list[set[int]]) -> dict:
    """Посчитать micro-F1, баллы и ошибки по категориям."""
    encoder = MultiLabelBinarizer(classes=tuple(CATEGORIES))
    truth = encoder.fit_transform(expected).astype(bool)
    output = encoder.transform(predicted).astype(bool)
    micro_f1 = float(f1_score(truth, output, average="micro", zero_division=0))
    counts = {"tp": truth & output, "fp": ~truth & output, "fn": truth & ~output}
    return {
        "micro_f1": micro_f1,
        "points": 5 * micro_f1,
        **{name: int(matrix.sum()) for name, matrix in counts.items()},
        "by_category": {
            mcid: {name: int(matrix[:, index].sum()) for name, matrix in counts.items()}
            for index, mcid in enumerate(CATEGORIES)
        },
    }


def main() -> None:
    """Оценить оба промпта на одной выборке."""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output", type=Path, default=HERE / "out/decision-1.json")
    args = parser.parse_args()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        parser.error("Укажите GOOGLE_API_KEY в окружении или .env")

    rows = sample_rows(args.dataset, args.sample_size, args.seed)
    sources = [int(row["source_mc_id"]) for row in rows]
    expected = [set(json.loads(row["target_mc_ids"])) for row in rows]
    for source, targets in zip(sources, expected, strict=True):
        if source not in CATEGORIES or source in targets or targets - CATEGORIES.keys():
            raise ValueError("Некорректные категории в train.csv")

    report = {
        "model": MODEL,
        "seed": args.seed,
        "threshold": THRESHOLD,
        "sample": [
            {"group_id": row["group_id"], "source_mc_id": source, "expected": sorted(targets)}
            for row, source, targets in zip(rows, sources, expected, strict=True)
        ],
        "results": {},
    }
    budget = TokenBudget()
    for variant in ("baseline", "guided"):
        prompt = (HERE / "prompts" / f"google_decision_{variant}.md").read_text(encoding="utf-8")
        probabilities = []
        for index, (row, source) in enumerate(zip(rows, sources, strict=True), start=1):
            try:
                probabilities.append(predict(api_key, prompt, row["description"], source, budget))
            except HTTPError as exc:
                raise SystemExit(f"{variant} {index}/{len(rows)}: HTTP {exc.code}") from None
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise SystemExit(f"{variant} {index}/{len(rows)}: {type(exc).__name__}") from None
        predicted = [{mcid for mcid, p in scores.items() if p >= THRESHOLD} for scores in probabilities]
        metrics = score(expected, predicted)
        logger.info("%s micro-F1 %.4f, TP %d, FP %d, FN %d",
                    variant, metrics["micro_f1"], metrics["tp"], metrics["fp"], metrics["fn"])
        report["results"][variant] = {
            **metrics, "predicted": [sorted(ids) for ids in predicted], "probabilities": probabilities
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
