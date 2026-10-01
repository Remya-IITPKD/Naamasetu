# Data

## Released in this repository

The parallel sentence pairs, the two baseline word alignments, the silver data and
the merged training sets used for all reported results are included here, gzipped
(each file < 100 MB):

| Path | Contents |
|---|---|
| `data/parallel/<lang>/parallel_<lang>_en.jsonl.gz` | EN–Indic sentence pairs mined from Wikipedia (Stage 1 output) |
| `data/parallel/<lang>/simalign_<lang>.jsonl.gz` | SimAlign word alignments of the pairs (baseline aligner) |
| `data/parallel/<lang>/awesome_<lang>.jsonl.gz` | awesome-align word alignments of the pairs (baseline aligner) |
| `data/projected/<lang>/alignment_results_<lang>.jsonl.gz` | word alignments from the Naamasetu hybrid aligner, with scores (Stage 2) |
| `data/projected/<lang>/silver_filtered_<lang>.jsonl.gz` | Naamasetu silver NER data after filtering (the "Silver (after filter)" column below) |
| `data/augmented/<lang>/train_aug_<lang>.jsonl.gz` | merged training set: Naamapadam gold train + 99.5% of the silver data |

Unpack with `gunzip -k data/*/*/*.jsonl.gz` (or read them directly with Python's
`gzip.open`). Larger files in `data/parallel/` and `data/projected/` are split at line
boundaries into `<name>.part1.jsonl.gz`, `<name>.part2.jsonl.gz`, …; join them in
order, e.g. `cat data/parallel/ml/simalign_ml.part{1..7}.jsonl.gz | gunzip > simalign_ml.jsonl`. The merged dev set is the Naamapadam gold dev split plus the 0.5% of
silver sentences that are not in `train_aug_<lang>`; the gold splits come from
`scripts/00_prepare_naamapadam.py`. The Wikipedia title lists (Stage 1 input) and
the unfiltered projections (`silver_projected_<lang>.jsonl`) are not included in
this release.

Full layout when all stages are run:

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

### Parallel pairs and baseline alignments

These are the files as produced by Stage 1 and by the two baseline aligners, released
unchanged (only gzipped). Line counts:

| Lang | Parallel pairs | SimAlign | awesome-align |
|---|---|---|---|
| as | 1,350 | 1,350 | 1,350 |
| gu | 75,005 | 37,383 | 75,005 |
| kn | 105,592 | 59,361 | 105,592 |
| ml | 240,163 | 240,163 | 240,163 |
| mr | 82,776 | 82,776 | 82,776 |
| or | 23,010 | 23,010 | 23,010 |
| pa | 128,364 | 71,047 | 128,364 |
| ta | 299,637 | 79,400 | 299,637 |
| te | 360,231 | 83,141 | 360,231 |

Notes on these files:

* Field names follow the target language code (`kn_sentence`, `kn_title`, …), as in
  the format below. Exceptions: the Assamese pairs use `english_*` / `assamese_*`
  (e.g. `english_sentence`, `assamese_sentence`), and the Gujarati awesome-align file
  stores the Gujarati sentence under the key `ml_sentence`. The Malayalam pairs have
  no URL fields.
* SimAlign and awesome-align files have `en_sentence`, `<lang>_sentence` and an
  `alignments` list of aligned word pairs (`{"<lang>_word": ..., "en_word": ...}`);
  the SimAlign files also repeat the titles, URLs and cosine similarity of the pair.
  The Malayalam SimAlign file uses a different layout: `sentence_id`, `english`,
  `malayalam` and `aligned_words` (`{"en", "ml", "similarity"}`).
* For gu, kn, pa, ta and te, SimAlign was run on a subset of the pairs (the counts above).
* The pair counts differ slightly from the "Parallel pairs" column of the table below
  for gu, kn, ml and pa (the counts there were taken from the extraction logs).
* `scripts/02_project_labels.py` reads `en_sentence` and `<lang>_sentence`, so the
  pair files can be fed to Stage 2 after `gunzip`; for Assamese, rename
  `english_sentence` → `en_sentence` and `assamese_sentence` → `as_sentence` first.

### Hybrid alignments

`alignment_results_<lang>` holds the output of the Naamasetu hybrid aligner
(semantic + phonetic + romanised similarity, Algorithm 1), released unchanged
(only gzipped). Each line has `en_sentence`, the target sentence under
`<lang>_sentence`, and `alignments` as in the format below
(`{"src_idx", "tgt_idx", "score"}`). Line counts and the run each file comes from:

| Lang | Lines | Source file |
|---|---|---|
| as | 1,289 | `alignment_results_as_2_new.jsonl` |
| gu | 74,964 | `alignment_results_gu_2_new.jsonl` |
| kn | 105,550 | `alignment_results_kn.jsonl` |
| ml | 240,102 | `alignment_results_ml.jsonl` |
| mr | 82,973 | `alignment_results_mr_2_new.jsonl` |
| or | 22,986 | `alignment_results_or_2_new.jsonl` |
| pa | 128,322 | `alignment_results_pa.jsonl` |
| ta | 299,342 | `alignment_results_ta_2_new.jsonl` |
| te | 360,202 | `alignment_results_te_2_new.jsonl` |

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
