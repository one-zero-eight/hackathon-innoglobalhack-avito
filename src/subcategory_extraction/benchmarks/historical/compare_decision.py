"""Сравнить Decision 1 и заранее заданную комбинацию на свежей выборке."""

import csv
import json
import os
import logging

from dotenv import load_dotenv

from benchmark import DATASET, HERE, PROMPT_DIR, ROOT, THRESHOLD, TokenBudget, predict, score


def main() -> None:
    """Сохранить ответы по одному и оценить комбинацию с TF-IDF."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GOOGLE_API_KEY")
    sample = json.loads((HERE / "2026-11-13_local-comparison.json").read_text())["fresh_32"]
    with DATASET.open(encoding="utf-8", newline="") as stream:
        descriptions = {
            (row["group_id"], int(row["source_mc_id"])): row["description"]
            for row in csv.DictReader(stream)
        }
    prompt = (PROMPT_DIR / "google_decision_guided.md").read_text(encoding="utf-8")
    output = HERE / "2026-11-13_decision-1_fresh32.json"
    report = json.loads(output.read_text()) if output.exists() else {"rows": []}
    budget = TokenBudget()

    for index, row in enumerate(sample):
        key = row["group_id"], row["source_mc_id"]
        if index < len(report["rows"]):
            saved = report["rows"][index]
            if (saved["group_id"], saved["source_mc_id"]) != key:
                raise ValueError("Сохранённые ответы относятся к другой выборке")
            continue
        if not api_key:
            raise SystemExit("Укажите GOOGLE_API_KEY для незавершённой выборки")
        probabilities = predict(api_key, prompt, descriptions[key], key[1], budget)
        google = {mcid for mcid, p in probabilities.items() if p >= THRESHOLD}
        local = set(row["local"]["tfidf"])
        report["rows"].append({
            **row, "probabilities": probabilities, "decision_1": sorted(google),
            "intersection": sorted(google & local), "union": sorted(google | local),
        })
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        logging.info("Decision 1 %d/%d", index + 1, len(sample))

    expected = [set(row["expected"]) for row in report["rows"]]
    # Retain confident Google decisions and use the local model near the cutoff.
    for row in report["rows"]:
        local = set(row["local"]["tfidf_gated_regex"])
        row["confidence_hybrid"] = sorted(
            int(mcid) for mcid, probability in row["probabilities"].items()
            if probability >= 0.65 or (probability >= 0.35 and int(mcid) in local)
        )
    report["scores"] = {
        name: score(expected, [set(row[name]) for row in report["rows"]])
        for name in ("decision_1", "intersection", "union", "confidence_hybrid")
    }
    for method in ("regex", "regex_gate", "tfidf", "tfidf_regex", "tfidf_gated_regex"):
        report["scores"][method] = score(
            expected, [set(row["local"][method]) for row in report["rows"]]
        )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
