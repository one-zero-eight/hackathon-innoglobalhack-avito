# Category benchmark log

## 2026-11-14 · Youla → InnoGlobalHack transfer

This run uses the requested `out/youla-300.jsonl` snapshot for training. It
contains 100 ads. `sample_mc_id` is Youla's provisional source mapping.
To get multilabel targets comparable to InnoGlobalHack, Decision 1 answered
the existing guided questions for categories 102–106 for every Youla ad,
using source 101 and a 0.50 cutoff. These are model-generated training labels.

The evaluation uses the existing grouped holdout from the official
InnoGlobalHack training JSONL. Its automatic labels were used only for scoring.
Normalized descriptions were checked against Youla before evaluation. All
learned models use a fixed 0.50 cutoff. The separate public-test Parquet
remains reserved.

Reproduce from the repository root with `PYTHONPATH=src uv run python
out/youla-train-avito.py`. The local `out` directory contains the labeling
script, cached probabilities, training script, and full metrics.

| Method | Micro-F1 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: |
| SplitShift regex | 0.9412 | 1216 | 148 | 4 |
| Regex + Youla-trained SplitShift gate | 0.9419 | 1215 | 145 | 5 |
| Youla-trained per-category gate, whole ad | 0.7040 | 861 | 365 | 359 |
| Youla-trained per-category gate, local context | 0.9168 | 1129 | 114 | 91 |
| Youla-trained per-category gate, whole ad + local context | 0.9211 | 1133 | 107 | 87 |

No normalized overlap was found with the 485 evaluation descriptions.
The same holdout previously yielded 0.9686 with an InnoGlobalHack-trained
context gate using 3,875 ads and a validation-selected cutoff. Training size,
domain, label source, and threshold differ, so the gap cannot be attributed
to the domain alone.

## 2026-11-14 · Expanded Youla learning curve

The 1,382-ad file supplied 400 labeled ads for nested comparisons. At a
fixed 0.50 cutoff, Youla-learned keyword micro-F1 rose from 0.8111 at 100
ads to 0.8917 at 200 and 0.9024 at 400. All variants used the same
485 InnoGlobalHack holdout ads. At 400, adding whole-ad or local TF-IDF
did not improve over keywords alone. See the current
[benchmark](../youla-learned-keywords.md).

The 800-ad extension reached 0.9168 micro-F1 with one Youla-keyword gate
per category, versus 0.9024 at 400 ads, on the same 485-row organizer-train
holdout. See `out/youla-curve-800.json` for per-category counts.
