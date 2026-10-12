# Dataset comparison (paper Table 1)

Sentences per language for the nine languages studied, in existing Indic NER datasets that cover at least one of them, grouped by construction method and ordered by total size. Datasets covering only Hindi/Bengali (e.g. HiNER, MultiCoNER) are not listed. FIRE-2014 and the CUSAT corpus are cited in the paper but omitted here because no per-language sentence counts are published. Machine-readable version: [`dataset_comparison.csv`](dataset_comparison.csv), with the source of every number.

| Dataset | Construction | as | gu | kn | ml | mr | or | pa | ta | te |
|---|---|---|---|---|---|---|---|---|---|---|
| Naamapadam (Mhaske et al. (2023)) | automatic | 10,369 | 476,310 | 475,163 | 721,244 | 458,628 | 198,780 | 466,867 | 501,435 | 511,288 |
| Naamasetu (ours) (this work) | automatic | 1,350 | 75,005 | 105,592 | 240,153 | 82,776 | 23,010 | 128,364 | 299,637 | 360,231 |
| CLASSER (Kaushik and Anand (2025)) | automatic | 156,842 | – | – | – | 680,892 | – | – | – | – |
| WikiAnn (Pan et al. (2017)) | automatic | 300 | 300 | 300 | 12,000 | 7,000 | 300 | 300 | 17,000 | 3,000 |
| L3Cube-MahaNER (Litake et al. (2022)) | manual | – | – | – | – | 25,000 | – | – | – | – |
| AsNER (Pathak et al. (2022)) | manual | 24,040 | – | – | – | – | – | – | – | – |
| Bahad et al. (Bahad et al. (2024)) | manual | – | – | – | – | – | 15,145 | – | – | 3,761 |
| L3Cube-MahaSocialNER (Chaudhari et al. (2024)) | manual | – | – | – | – | 18,000 | – | – | – | – |

Sources of the counts:

* **Naamapadam**: train+dev+test sentences of the released Naamapadam splits (as used in this work; data/naamapadam/).
* **Naamasetu (ours)**: all generated sentence pairs, context similarity >= 0.7 (data/parallel/); 767,727 carry at least one entity.
* **CLASSER**: train+dev+test sentences, CLASSER paper Table 1 (as: 140,257+15,585+1,000; mr: 611,902+67,990+1,000).
* **WikiAnn**: train+validation+test rows of the Hugging Face unimelb-nlp/wikiann splits (Rahimi et al., 2019).
* **L3Cube-MahaNER**: 21,500 train + 1,500 validation + 2,000 test sentences (paper Table 1).
* **AsNER**: 21,475 train + 767 dev + 1,798 test sentences (paper Table 2).
* **Bahad et al.**: Odia 12,109+1,519+1,517; Telugu 2,993+384+384 sentences (paper Table 3).
* **L3Cube-MahaSocialNER**: 18,000 sentences (paper abstract).

Only Naamapadam, WikiAnn and Naamasetu cover all nine languages; every other corpus covers at most two.
