# Data

Large files are **not** stored in git. Download them from the anonymous link below and unpack them into `data/`.

> **Download:** TODO — anonymous link (e.g. an anonymous Zenodo / OSF / HF dataset) — added for review

Expected layout after download:

```
data/
├── raw/titles/<lang>_en_titles_urls.csv      Wikipedia title pairs (Stage 1 input)
├── parallel/<lang>/parallel_<lang>_en.jsonl  EN–Indic sentence pairs (Stage 1 output)
├── projected/<lang>/
│   ├── alignment_results_<lang>.jsonl        word alignments + scores
│   ├── silver_projected_<lang>.jsonl         projected NER (before filtering)
│   └── silver_filtered_<lang>.jsonl          projected NER (after filtering)
├── naamapadam/<lang>/{train,dev,test}.jsonl  gold (created by scripts/00_prepare_naamapadam.py)
└── augmented/<lang>/{train_aug,dev_aug,test}_<lang>.jsonl + stats_<lang>.json
```

## Formats

All files are UTF-8 JSONL (one JSON object per line). `data/sample/` has a tiny example of each format.

**Parallel pairs**
```json
{"lang": "mr", "tgt_title": "...", "en_title": "...", "tgt_url": "...", "en_url": "...",
 "tgt_sentence": "...", "en_sentence": "...", "cosine_similarity": 0.83}
```

**NER (gold, silver, merged)**
```json
{"id": "mr_17", "tokens": ["...", "..."], "ner_tags": ["B-PER", "O"]}
```
Tag set: `O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC`.

**Alignments**
```json
{"en_sentence": "...", "tgt_sentence": "...",
 "alignments": [{"src_idx": 0, "tgt_idx": 3, "score": 0.91}]}
```
`src_idx` indexes the whitespace-tokenised English sentence. `tgt_idx` indexes the indic-nlp-tokenised target sentence.

## Statistics

Sentence counts (lines in each JSONL file). Merged train = gold train + 99.5% of the
filtered silver; merged dev = gold dev + the other 0.5%. The test set is the unchanged gold test.

| Lang | Parallel pairs (cos ≥ 0.7) | Silver (after filter) | Gold train | Gold dev | Gold test | Merged train | Merged dev |
|---|---|---|---|---|---|---|---|
| as | 1,350 | 632 | 10,266 | 52 | 51 | 10,895 | 55 |
| gu | 75,017 | 43,972 | 472,845 | 2,389 | 1,076 | 516,597 | 2,609 |
| kn | 105,550 | 48,911 | 471,763 | 2,381 | 1,019 | 520,429 | 2,626 |
| ml | 240,000 | 131,224 | 716,652 | 3,618 | 974 | 847,220 | 4,274 |
| mr | 82,776 | 44,415 | 455,248 | 2,300 | 1,080 | 499,441 | 2,522 |
| or | 23,010 | 11,122 | 196,793 | 993 | 994 | 207,859 | 1,049 |
| pa | 128,322 | 81,409 | 463,534 | 2,340 | 993 | 544,536 | 2,747 |
| ta | 299,637 | 172,150 | 497,882 | 2,795 | 758 | 669,171 | 3,656 |
| te | 360,231 | 233,892 | 507,741 | 2,700 | 847 | 740,464 | 3,869 |

## Sources and licences

| Artifact | Source | Licence |
|---|---|---|
| Wikipedia text | {as,gu,kn,ml,mr,or,pa,ta,te,en}.wikipedia.org | CC BY-SA 4.0 |
| Naamapadam | `ai4bharat/naamapadam` (HF Hub) | CC0 1.0 (per the dataset card) |
| Released parallel + silver data | this work | CC BY-SA 4.0 (inherits from Wikipedia) |

Silver labels are produced automatically and **contain noise**. Do not use them as evaluation data.
