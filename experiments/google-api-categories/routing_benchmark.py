"""Проверить Gemini Decision 1 только на расхождениях локальных моделей."""

import csv
import hashlib
import json
import os

from dotenv import load_dotenv

from benchmark import DATASET, HERE, MODEL, ROOT, TokenBudget, predict, score

MARGINS = (None, 0.1, 0.2, 0.3, 0.0)
OUTPUT = HERE / "2026-11-13_routing-results.json"


def local_sets(row):
    return set(row["tfidf_gated_regex"]), set(row["regex_gate"])


def routed(row, answers, margin):
    baseline, alternative = local_sets(row)
    if margin is None:
        return baseline
    for category in baseline ^ alternative:
        probability = answers[str(category)]
        if probability >= 0.5 + margin:
            baseline.add(category)
        elif probability < 0.5 - margin:
            baseline.discard(category)
    return baseline


def measure(rows, cache, margin):
    return score(
        [set(row["expected"]) for row in rows],
        [routed(row, cache.get(row["group_id"], {}).get("probabilities", {}), margin) for row in rows],
    )


def main():
    load_dotenv(ROOT / ".env")
    source = (HERE / "2026-11-13_routing-sample.json").read_bytes()
    sample = json.loads(source)
    prompt = (HERE / "prompts/google_decision_guided.md").read_text(encoding="utf-8")
    fingerprint = hashlib.sha256(source + prompt.encode()).hexdigest()
    report = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {
        "model": MODEL, "fingerprint": fingerprint, "responses": {},
    }
    if report["fingerprint"] != fingerprint:
        raise ValueError("Выборка или промпт изменились")
    cache = report["responses"]
    with DATASET.open(encoding="utf-8", newline="") as stream:
        descriptions = {
            (row["group_id"], int(row["source_mc_id"])): row["description"]
            for row in csv.DictReader(stream)
        }
    budget = TokenBudget()

    def save():
        temp = OUTPUT.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(OUTPUT)

    for split, rows in sample.items():
        for row in rows:
            group, source_id = row["group_id"], row["source_mc_id"]
            baseline, alternative = local_sets(row)
            disputed = baseline ^ alternative
            if not disputed or group in cache:
                continue
            probabilities = predict(os.environ["GOOGLE_API_KEY"], prompt,
                                    descriptions[group, source_id], source_id, budget, disputed)
            cache[group] = {"probabilities": {str(k): v for k, v in probabilities.items()}, "input_tokens": budget.last_actual}
            save()
            print(f"{split}: {len(cache)} calls saved", flush=True)

        if split == "validation":
            trials = [(margin, measure(rows, cache, margin)) for margin in MARGINS]
            report["validation"] = trials
            report["chosen_margin"] = max(trials, key=lambda trial: trial[1]["micro_f1"])[0]
        else:
            report["evaluation"] = {}
            for name, subset in {
                "pooled": rows,
                "innoglobalhack": [row for row in rows if row["origins"] == ["new_train"]],
                "purple": [row for row in rows if "new_train" not in row["origins"]],
            }.items():
                used = [cache[row["group_id"]] for row in subset if row["group_id"] in cache]
                report["evaluation"][name] = {
                    "ads": len(subset), "calls": len(used),
                    "questions": sum(len(r["probabilities"]) for r in used),
                    "input_tokens": sum(r["input_tokens"] or 0 for r in used),
                    "tfidf_gated_regex": measure(subset, cache, None),
                    "regex_gate": score([set(r["expected"]) for r in subset],
                                        [set(r["regex_gate"]) for r in subset]),
                    "selected_rule": measure(subset, cache, report["chosen_margin"]),
                    "all_disagreements": measure(subset, cache, 0.0),
                }
        save()


if __name__ == "__main__":
    main()
