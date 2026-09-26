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

<!-- TODO: fill from data/augmented/<lang>/stats_<lang>.json -->

| Lang | Parallel pairs | Silver (after filter) | Gold train | Gold dev | Gold test |
|---|---|---|---|---|---|
| as | | | | | |
| gu | | | | | |
| kn | | | | | |
| ml | | | | | |
| mr | | | | | |
| or | | | | | |
| pa | | | | | |
| ta | | | | | |
| te | | | | | |

## Sources and licences

| Artifact | Source | Licence |
|---|---|---|
| Wikipedia text | {as,gu,kn,ml,mr,or,pa,ta,te,en}.wikipedia.org | CC BY-SA 4.0 |
| Naamapadam | `ai4bharat/naamapadam` (HF Hub) | TODO: confirm from the dataset card |
| Released parallel + silver data | this work | CC BY-SA 4.0 (inherits from Wikipedia) |

Silver labels are produced automatically and **contain noise**. Do not use them as evaluation data.
