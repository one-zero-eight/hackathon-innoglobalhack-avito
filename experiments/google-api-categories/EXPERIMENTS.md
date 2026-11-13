# Category benchmark log

## 2026-11-13 · First Decision 1 run

Saved the [full results](2026-11-13_decision-1_seed43.json), including the
ordered sample, labels, probabilities and predictions. The run used the merged
`local-notes/joined-datasets/train.csv` (9,295 rows, SHA-256
`3874316216c6c717e9e172e9e77bbbe752e6b25f3218d8c028b4a9a892ec7fc8`).
Seed 43 selected 32 distinct description groups, with 42 target labels and 17
empty target sets. Both prompts used a fixed 0.5 threshold.

| Prompt | Micro-F1 | Points / 5 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short | 0.6777 | 3.3884 | 41 | 38 | 1 |
| Guided | 0.8632 | 4.3158 | 41 | 12 | 1 |

The [Flash-Lite run](2026-11-13_flash-lite.md) used a different dataset and
sample. Use seed 43 for paired diagnostics with the saved Decision 1 predictions.

## Google quota

The screenshot showed **49.86K / 50K tokens per minute**, about 99.7% of the
limit, with requests at 49 / 200 per minute. For future runs, start with one
request in flight and a rolling 60-second input budget of **45K tokens**, counting
ad text and all five questions. Reconcile estimates with API usage. Save each
completed ad and prompt variant so a 429 can pause the run and resume from the
next unfinished request. Keep prompt wording fixed when comparing model quality.

## Benchmark plan

The knowledge map is in `backlog/avito-services-knowledge-map.html`. Local
checkouts are under `../IGH cases`.

| Order | Candidate | What to compare |
| --- | --- | --- |
| 1 | Always empty, all five other categories, and SplitShift regex | Local controls. The regex detector in `splitshift/src/splitshift/detector.py` gives candidate coverage. Score it as predictions too, with the mention-only behavior labeled. |
| 2 | SplitShift TF-IDF gate and a fresh TF-IDF category baseline | Compare regex alone with regex plus the binary split gate. Fit the gate and the per-category classifier on the same grouped training split. |
| 3 | Local InnoGlobalHack `innoglobalhack/apps/ml` | Closest case match. Start from the classification stage behind `scripts/eval.py`. Verify taxonomy, source-category handling, provider setup and draft cap before reusing its evaluator. |
| Later | `avito-purple-2026` learned candidate gate and `hack-mfti` retrieval | Compare candidate recall separately from final category F1. Verify artifact provenance and model dependencies first. Treat `avito_hack_solution` as the same lineage as SplitShift. |

1. Replay the saved 32 rows for diagnostics. Freeze a fresh, group-disjoint
   evaluation slice before selecting prompts or tuning thresholds. Keep all rows
   with the same `group_id` on one side of the split. Join the companion
   `train.jsonl` on `(group_id, source_mc_id)` for source provenance and report
   the InnoGlobalHack and Purple slices separately alongside the pooled result.
2. Use identical evaluation rows, descriptions, and source IDs across methods.
   For Purple predictors map old IDs to the current taxonomy, drop unsupported
   IDs, deduplicate, and remove the source. If a native Purple predictor requires
   a legacy source ID, use provenance where it exists. Current 106 has multiple
   legacy IDs, so report adapter coverage and use the same covered subset for
   paired comparisons. Purple labels target independently offered services.
3. Fit learned baselines using only the training groups. Check legacy training
   overlap before using pretrained artifacts for headline scores. Compare
   category micro-F1 and 5 × micro-F1, per-category TP/FP/FN, and false positives
   on empty-target ads. For retrieval stages also report candidate recall.
4. Run local baselines first, then send the frozen slice to Decision 1 under the
   token budget. Record dataset and sample IDs, repo commits, prompt version,
   threshold, API usage, results and next decision in this log after each run.

## 2026-11-13 · Local baselines and combinations

[Local report](2026-11-13_local-comparison.json). SplitShift checkout
`239c1dc`. Excluded all groups from the first API run, then split the rest
by description group into 7,410 training, 926 validation, and 927 test rows.
The test includes 477 InnoGlobalHack-only rows, 448 Purple-only rows, and two
with mixed provenance. Selected TF-IDF thresholds and regex bonuses on validation.
SplitShift's gate is refit on the merged training split. Its mention-count
features include all legacy categories because the mapped source can be
ambiguous. Final predictions always exclude the current source category.

| Method | Pooled micro-F1 | InnoGlobalHack | Purple | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empty | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 1450 |
| All other categories | 0.4766 | 0.6672 | 0.2051 | 1450 | 3185 | 0 |
| SplitShift regex | 0.8145 | 0.9388 | 0.4854 | 1420 | 617 | 30 |
| SplitShift regex + split gate | 0.8716 | 0.9326 | 0.6365 | 1310 | 246 | 140 |
| TF-IDF | 0.8549 | 0.9191 | 0.5621 | 1243 | 215 | 207 |
| TF-IDF + regex bonus | 0.8736 | 0.9404 | 0.6054 | 1310 | 239 | 140 |
| TF-IDF + gated regex bonus | **0.8782** | 0.9357 | 0.6262 | 1284 | 190 | 166 |

Regex candidate recall was 1420/1450 (97.9%). The gated regex bonus improved
pooled F1 by 2.33 points over TF-IDF alone. The best method differed by source.
The neighboring InnoGlobalHack classifier at `95b6b60` remains unscored. Its
default provider is mock and its catalog combines painting and plastering,
which are separate targets here. A real provider and a way to resolve that
category by the underlying work are needed for a comparable score.

The API runner now reserves conservatively from payload size and reconciles
against Google input-token usage under a 45K per-process rolling budget.

### Fresh 32-row Decision 1 comparison

[Paired report](2026-11-13_decision-1_fresh32.json). Seed 46 sampled 32 distinct
groups from the 927-row holdout. This slice has 59 positive labels. The same
guided questions and 0.5 cutoff were used as in the first API run.

| Method | Micro-F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| Decision 1 | 0.8000 | 54 | 22 | 5 |
| Decision 1 ∩ TF-IDF | 0.8448 | 49 | 8 | 10 |
| Decision 1 ∪ TF-IDF | 0.8143 | 57 | 24 | 2 |
| Decision 1 confidence hybrid | 0.8209 | 55 | 20 | 4 |
| TF-IDF | 0.8595 | 52 | 10 | 7 |
| TF-IDF + gated regex bonus | 0.8689 | 53 | 10 | 6 |
| SplitShift regex + split gate | **0.8800** | 55 | 11 | 4 |

The confidence hybrid accepts Decision 1 at 0.65 and uses the local gated-regex
TF-IDF predictions for probabilities in [0.35, 0.65). The intersection removed
14 false positives but lost five true positives. The union recovered three
true positives and added two false positives. Neither beat local TF-IDF on
these same rows. Use the local gated-regex combination as the pooled baseline.
The next question is whether Decision 1 helps on a narrowly defined set of
local disagreements. Use a separate validation slice to design that routing.

## 2026-11-13 · Decision 1 as a tie-breaker

[Sample](2026-11-13_routing-sample.json) · [Responses and scores](2026-11-13_routing-results.json)

Seeds 47 and 48 selected 64 routing-validation and 128 routing-evaluation
ads from the local holdout, excluding both earlier API samples. The two
sets are group-disjoint. The local models were already fitted and tuned on
separate groups. Google saw only the categories on which the local methods
disagreed. On validation, a 0.20 confidence margin and routing every
disagreement tied at 0.9194, versus 0.9082 for the TF-IDF baseline.
The tie favored the larger margin. It accepts probabilities at or above
0.70 and rejects below 0.30, retaining the TF-IDF decision in between.

| Evaluation method | Micro-F1 | InnoGlobalHack | Purple | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + gated regex | 0.8612 | 0.9249 | 0.5823 | 183 | 29 | 30 |
| SplitShift regex + gate | 0.8767 | 0.9249 | 0.6957 | 192 | 33 | 21 |
| Frozen 0.20 margin | **0.8782** | 0.9364 | 0.6517 | 191 | 31 | 22 |
| Route every disagreement (comparison) | 0.8858 | 0.9395 | 0.6813 | 194 | 31 | 19 |

Evaluation used 25 requests for 128 ads (19.5%), 33 category questions,
and 15,388 input tokens. Validation and evaluation together used 37
requests and 23,851 input tokens under the 45K rolling budget. The frozen
rule beat the strongest local method by 0.00145 F1 pooled. Against that
method it gained two correct labels and removed two false positives on
InnoGlobalHack, while losing three correct labels on Purple. Treat
routing every disagreement as a candidate for the next fresh evaluation.

## Dataset policy for new experiments

Use only the official InnoGlobalHack files for future training, validation,
and evaluation. Earlier results on the merged CSV are historical.

- `docs/innoglobalhack_avito/public-train-annotated.jsonl` contains 10,000
  automatically labeled ads. The source is always 101 and the targets are
  `found_microcategories`. Use description-group splits for model fitting,
  threshold selection, and internal holdout checks. Audit duplicate groups
  and conflicting labels before fitting.
- `docs/innoglobalhack_avito/public-test-dataset.parquet` contains 208 ads
  with category names. Reserve it for evaluation after freezing the model,
  prompts, and thresholds. Verify the label mapping and audit overlap with
  training descriptions before reporting F1.
- Purple Hack and the merged CSV encode a different task in part. Purple
  marks separately offered services. InnoGlobalHack can count a specific
  service the provider offers as a stage of a renovation. Keep prior scores
  for reference and use the original InnoGlobalHack files for new runs.

Candidate gate baseline: L2-regularized logistic regression on one
(ad, candidate category) pair per non-source category. Compare whole-ad
TF-IDF plus category ID against the same inputs with category-local snippets
and match features. Use the text and source category available at inference.
Keep ads without a keyword match, and tune a single threshold on grouped
validation micro-F1. Any pretrained whole-ad score used as a feature needs
out-of-fold predictions for gate training.

Initial shortcut check on the official training JSONL. Normalize descriptions
with Unicode NFKC, case folding, ё → е, and whitespace collapsing. The
10,000 rows contain 4,854 distinct descriptions. 5,559 rows belong to 413
repeated-description groups, the largest with 94 rows. Repeated descriptions
have identical target sets. On a deliberately naive 80/20 row split
(Python shuffle, seed 42), 1,129 of 2,000 holdout descriptions also appear
in training. Exact-description lookup alone scored 0.7256 micro-F1
(2,961 TP, 0 FP, 2,240 FN). Use this as a leakage diagnostic.

A clean-context Terra 6.1 audit independently found 413 repeated groups
under simpler lowercase/trim normalization. Ignoring punctuation and numbers
increased the count of rows in repeated groups to 5,867. Five simple category
regexes reached 0.9189 micro-F1 over the training annotations, versus 0.6699
for always predicting all five. This is an in-sample annotation diagnostic.
Check that category rules transfer to unseen templates before interpreting
their scores as generalization. Train/public-test overlap remains unverified.
