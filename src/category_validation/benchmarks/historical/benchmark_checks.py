from benchmark import sample_rows, score
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import logging


def check_micro_f1() -> None:
    expected = [{102, 104}, set(), {103, 104, 105}]
    predicted = [{102, 105}, {101}, {103, 104, 105}]

    metrics = score(expected, predicted)

    assert metrics["micro_f1"] == 8 / 11
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (4, 2, 1)
    assert metrics["by_category"][104] == {"tp": 1, "fp": 0, "fn": 1}


def check_sampling(directory: Path) -> None:
    dataset = directory / "train.csv"
    dataset.write_text(
        'source_mc_id,target_mc_ids,group_id,description\n'
        '101,"[104,105]",same,"first\nmultiline"\n'
        '104,[],same,"first\nmultiline"\n'
        '103,[],other,second\n',
        encoding="utf-8",
    )

    rows = sample_rows(dataset, 2, 43)

    assert rows == sample_rows(dataset, 2, 43)
    assert {row["group_id"] for row in rows} == {"same", "other"}


def check_rolling_budget() -> None:
    from subcategory_extraction.google_decision import client

    now = 0.0
    waits = []

    def wait(seconds):
        nonlocal now
        waits.append(seconds)
        now += seconds

    with patch.object(client, "monotonic", lambda: now), patch.object(client, "sleep", wait):
        budget = client.TokenBudget()
        budget.reserve(30_000)
        budget.reconcile(20_000)
        now = 10.0
        budget.reserve(25_000)
        budget.reserve(1)
    assert waits == [50.0]


def check_routing() -> None:
    from routing_benchmark import routed

    row = {"tfidf_gated_regex": [102, 103, 105], "regex_gate": [103, 104]}
    answers = {"102": 0.4, "104": 0.8, "105": 0.2}

    assert routed(row, answers, None) == {102, 103, 105}
    assert routed(row, answers, 0.0) == {103, 104}
    assert routed(row, answers, 0.2) == {102, 103, 104}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_micro_f1()
    with TemporaryDirectory() as directory:
        check_sampling(Path(directory))
    check_rolling_budget()
    check_routing()
    logging.info("4/4 проверок исторических метрик, выборки, бюджета и маршрутизации")
