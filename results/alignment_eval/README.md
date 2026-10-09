# Alignment evaluation (paper Tables 2, 3 and 7)

Intrinsic comparison of the Naamasetu hybrid aligner with SimAlign and awesome-align,
computed entirely from the released alignment files. The tables are in
[`tables.md`](tables.md); every number in them comes from [`scores.csv`](scores.csv).

## What is measured

All three aligners are scored on the **same 200 sentence pairs per language**, drawn at
random (seed 42) from the pairs that all three aligned (SimAlign did not finish for gu,
kn, pa, ta and te, so the pool there is smaller; see `data/README.md`).

| Measure | Definition | Paper |
|---|---|---|
| Source coverage | fraction of English words with at least one alignment link | Table 2 |
| Target coverage | fraction of target-language words with at least one alignment link | Table 2 |
| Average similarity on entity-bearing words | average similarity of the links on English entity words: multilingual-embedding similarity for SimAlign and awesome-align, the combined semantic + phonetic + romanisation score for the hybrid aligner | Table 3 |
| Entity-word coverage | fraction of English **entity** words with at least one link; entity words are tagged with spaCy (`en_core_web_sm`, PERSON→PER, GPE/LOC→LOC, ORG→ORG), exactly as in `scripts/02_project_labels.py` | Table 7 |

Words are whitespace-separated words of each sentence, the same for every aligner, so
the three aligners are compared on identical denominators.

Coverage counts links, not correct links: there are no gold word alignments, so higher
coverage does not by itself show higher alignment precision.

## Results in short

* **Source coverage:** hybrid best in 8 of 9 languages (0.79–1.00); Malayalam is the
  exception (SimAlign 0.832 vs. hybrid 0.665). Largest gain on Odia (0.309 → 0.812).
* **Target coverage:** hybrid best in all 9 languages.
* **Entity-word coverage:** hybrid best in 8 of 9 languages (0.75–0.99); Malayalam is
  again the exception. Largest gain on Odia (0.462 → 0.845).
* **Average similarity on entity-bearing words (Table 3):** SimAlign 0.53–0.69 and
  awesome-align 0.66–0.76 (embedding similarity); hybrid 0.78–0.84 (combined semantic +
  phonetic + romanisation score).
* **Averages over the nine languages** (source / target / entity-word coverage): hybrid
  0.836 / 0.877 / 0.889, SimAlign 0.682 / 0.705 / 0.783, awesome-align 0.363 / 0.447 / 0.497.

## Additional columns in `scores.csv`

`scores.csv` also reports, per language and aligner:

| Column | Meaning |
|---|---|
| `cov_PER`, `cov_LOC`, `cov_ORG` (`n_*` = number of words) | entity-word coverage per type; the hybrid aligner covers the most Person words in all 9 languages |
| `span_any`, `span_all` | fraction of English entity spans (BIO) with at least one / all words linked |
| `ent_cos`, `span_cos`, `cos_PER/LOC/ORG` | mean cosine similarity of the links on entity words (or spans), computed for **every** aligner with the same encoder: mBERT (`bert-base-multilingual-cased`), layer 8, word vector = mean of its sub-word states, as in the hybrid aligner |
| `hybrid_H` | mean of the hybrid score `H` stored in the hybrid files on entity-word links |

Table 3 uses `ent_cos` for SimAlign and awesome-align and `hybrid_H` for the hybrid
aligner. `hybrid_H` combines semantic, phonetic and romanisation scores, adds the entity
bonus λ and is normalised per sentence, so it is a different measure from the cosine columns.

## How the released files are read

`scripts/analysis/build_alignment_sample.py` maps every aligner's links onto the
whitespace words of the pair:

* **SimAlign / awesome-align** files store links as word strings. Each string is mapped to
  the first not-yet-linked word with that string (exact match, then ignoring punctuation,
  then the word containing the token; the Malayalam SimAlign run split e.g. "India's" into
  "India" + "'s").
* **Hybrid** files store word indices, **with the two fields swapped** relative to their
  names: `src_idx` indexes the Indic-NLP tokens of the (normalised) target sentence and
  `tgt_idx` indexes `en_sentence.split()`. Each target token is mapped to the whitespace
  word it lies in.

Fewer than 0.3% of links cannot be mapped (counts are printed when the sample is built).

## Files

| File | Contents |
|---|---|
| `sample_<lang>.jsonl` | the 200 sampled pairs: English and target words, English BIO tags, and the links of all three aligners as `[english_word_index, target_word_index]` |
| `scores.csv` | all measures above, per language and aligner |
| `tables.md` | Tables 2, 3 and 7 of the paper |

## Reproduce

```bash
pip install spacy indic-nlp-library && python -m spacy download en_core_web_sm
python scripts/analysis/build_alignment_sample.py --n 200 --seed 42 --out-dir results/alignment_eval
# coverage only (no GPU or model needed):
python scripts/analysis/alignment_eval_scores.py --sample-dir results/alignment_eval --out results/alignment_eval/scores.csv
# with the cosine columns (downloads mBERT; a GPU helps but is not required):
python scripts/analysis/alignment_eval_scores.py --sample-dir results/alignment_eval \
    --model bert-base-multilingual-cased --layer 8 --out results/alignment_eval/scores.csv
python scripts/analysis/alignment_tables.py
```
