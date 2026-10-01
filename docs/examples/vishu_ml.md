# Example: the festival *Vishu* (വിഷു) in Malayalam NER data

PER/LOC/ORG has no class for festivals, so the correct label for *Vishu* is `O`
(other). This file shows how the word is labelled in the Naamapadam Malayalam
training data and in the Naamasetu silver data.

## A Naamapadam training sentence

From the Naamapadam Malayalam **training split** (`ai4bharat/naamapadam`,
`data/naamapadam/ml/train.jsonl` as written by `scripts/00_prepare_naamapadam.py`;
line 3,386 of 716,652). The sentence is from Naamapadam, not from our projection.

> തിരുവനന്തപുരംഃ വിഷു ദിനത്തിൽ ഗാന്ധാരിയമ്മൻ കോവിലിൽ വച്ച് തുലാഭാരം നടത്തുന്നതിനിടെ ത്രാസ് പൊട്ടി വീണ് തലയ്ക്ക് പരിക്കേറ്റ തിരുവനന്തപുരം ലോക്സഭാ മണ്ഡലത്തിലെ യുഡിഎഫ് സ്ഥാനാർത്ഥി ശശി തരൂർ ആശുപത്രി വിട്ടു .
>
> "Thiruvananthapuram: UDF candidate for the Thiruvananthapuram Lok Sabha
> constituency Shashi Tharoor, who was injured on the head when the scale broke
> during a thulabharam at the Gandhari Amman Kovil on Vishu day, left hospital."

```
Tokens : ['തിരുവനന്തപുരംഃ', 'വിഷു', 'ദിനത്തിൽ', 'ഗാന്ധാരിയമ്മൻ', 'കോവിലിൽ', 'വച്ച്', 'തുലാഭാരം', 'നടത്തുന്നതിനിടെ', 'ത്രാസ്', 'പൊട്ടി', 'വീണ്', 'തലയ്ക്ക്', 'പരിക്കേറ്റ', 'തിരുവനന്തപുരം', 'ലോക്സഭാ', 'മണ്ഡലത്തിലെ', 'യുഡിഎഫ്', 'സ്ഥാനാർത്ഥി', 'ശശി', 'തരൂർ', 'ആശുപത്രി', 'വിട്ടു', '.']
Tags   : ['B-LOC', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O', 'O', 'B-ORG', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O']
ENTITY : തിരുവനന്തപുരംഃ -> B-LOC     (Thiruvananthapuram)
ENTITY : ഗാന്ധാരിയമ്മൻ -> B-LOC      (Gandhari Amman)
ENTITY : കോവിലിൽ -> I-LOC            (Kovil, "temple")
ENTITY : തിരുവനന്തപുരം -> B-LOC      (Thiruvananthapuram)
ENTITY : യുഡിഎഫ് -> B-ORG            (UDF)
ENTITY : ശശി -> B-PER                (Shashi)
ENTITY : തരൂർ -> I-PER               (Tharoor)
other  : വിഷു -> O                   (Vishu, the festival)
```

The place, the party and the person are tagged, and *Vishu* is `O`.

## How *Vishu* is labelled across the data

Counts for the exact token വിഷു (inflected forms such as വിഷുവിന് are not
included; വിഷുവം "equinox" is a different word):

| Data | Occurrences | O | LOC | PER | ORG |
|---|---|---|---|---|---|
| Naamapadam ml train (gold) | 45 | 21 | 14 | 9 | 1 |
| Naamapadam ml dev (gold) | 1 | 0 | 1 | 0 | 0 |
| Naamasetu ml silver (`data/projected/ml/silver_filtered_ml.jsonl.gz`) | 3 | 3 | 0 | 0 | 0 |

In the Naamapadam training data the festival is labelled `O` in 21 of 45
occurrences and as an entity in the other 24 (most often LOC or PER). In the
Naamasetu silver data all three occurrences are `O`. The silver counts are small.

## Reproduce

```bash
python scripts/00_prepare_naamapadam.py --langs ml        # gold splits -> data/naamapadam/ml/
python scripts/analysis/token_tag_counts.py data/naamapadam/ml/train.jsonl വിഷു
python scripts/analysis/token_tag_counts.py data/naamapadam/ml/dev.jsonl വിഷു
python scripts/analysis/token_tag_counts.py data/projected/ml/silver_filtered_ml.jsonl.gz വിഷു
```

The script prints every tag count by token form (the table above uses the
`വിഷു` row) and the first matching sentences.
