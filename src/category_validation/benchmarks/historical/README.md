# Historical category benchmarks

Архив экспериментов до модульной интеграции. Результаты не являются
оценкой новых модулей. Скрипты используют локальный объединённый CSV и
внешний SplitShift; для новых оценок используйте официальный InnoGlobalHack.

Офлайн-проверки без pytest и внешних данных:
`uv run python src/category_validation/benchmarks/historical/benchmark_checks.py`.
Google-клиент, промпты и функция арбитража теперь импортируются из
`subcategory_extraction/google_decision`, метрики — из общих benchmarks.

Run from the repository root with `GOOGLE_API_KEY` in the environment or the
root `.env`.

```sh
uv run python src/category_validation/benchmarks/historical/benchmark.py
```

The runner uses `local-notes/joined-datasets/train.csv` by default. This local
CSV is git-ignored. Use `--dataset` to point to another copy with the same columns.

The default sample is 32 rows with distinct `group_id` values, selected after
shuffling with seed 43. This sample has no description overlap with the earlier
Flash-Lite sample. Each prompt sees the same rows. Each request sends one ad
description and five independent yes/no questions, omitting the row's
`source_mc_id`. The guided prompt adds category examples and criteria for
explicitly offered services.

A category is predicted at `P(yes) >= 0.5`. The reference is the CSV
`target_mc_ids` column. Micro-F1 uses the same six binary columns for
predictions and references, and points are five times micro-F1.
`out/decision-1.json` in this directory records group IDs, source and target
categories, per-category probabilities, and TP, FP and FN.

[Google Decisions API announcement and request example](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/)

## Results on 2026-11-13

| Questions | Micro-F1 | Points / 5 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 0.6777 | 3.3884 | 41 | 38 | 1 |
| Guided | 0.8632 | 4.3158 | 41 | 12 | 1 |

The sample had 42 target labels and 17 ads with empty target lists.

See the [experiment log](EXPERIMENTS.md) for the saved run, quota observations,
and benchmark plan.

Run the grouped local holdout and the paired Decision 1 follow-up from the
repository root. The local benchmark imports SplitShift from `../IGH cases`.

```sh
uv run python src/category_validation/benchmarks/historical/local_benchmark.py
uv run python src/category_validation/benchmarks/historical/compare_decision.py
```

The local runner also freezes the 64/128 group-disjoint routing samples.
Run the disagreement-only API experiment with:

```sh
uv run python src/category_validation/benchmarks/historical/routing_benchmark.py
```

The routing runner checkpoints each API response and resumes matching samples.
