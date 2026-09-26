#!/usr/bin/env python3
"""Stage 3 — Filter projected (silver) data before merging with gold.

Keeps a sentence only if:
  * tokens and tags are non-empty and of equal length,
  * every tag is in the PER/LOC/ORG BIO schema,
  * it contains at least one entity (all-O sentences are dropped),
  * (optional) it has between --min-tokens and --max-tokens tokens.

With --dedupe, exact duplicates (same token sequence) are also removed.

Usage:
    python scripts/03_filter_silver.py \
        --input  data/projected/mr/silver_projected_mr.jsonl \
        --output data/projected/mr/silver_filtered_mr.jsonl
"""

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naamasetu.data import read_jsonl, write_jsonl  # noqa: E402
from naamasetu.labels import TAG2ID, decode_tags  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-tokens", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=10_000)
    ap.add_argument("--dedupe", action="store_true", help="drop exact duplicate token sequences")
    args = ap.parse_args()

    reasons, kept, seen = Counter(), [], set()
    for r in read_jsonl(args.input):
        toks, tags = r.get("tokens") or [], decode_tags(r.get("ner_tags") or [])
        if not toks or len(toks) != len(tags):
            reasons["length_mismatch_or_empty"] += 1; continue
        if any(t not in TAG2ID for t in tags):
            reasons["unknown_tag"] += 1; continue
        if all(t == "O" for t in tags):
            reasons["all_O"] += 1; continue
        if not (args.min_tokens <= len(toks) <= args.max_tokens):
            reasons["length_out_of_range"] += 1; continue
        key = tuple(toks)
        if args.dedupe and key in seen:
            reasons["duplicate"] += 1; continue
        seen.add(key)
        kept.append({"id": r.get("id"), "tokens": toks, "ner_tags": tags})

    write_jsonl(kept, args.output)
    total = len(kept) + sum(reasons.values())
    print(f"Input     : {total:,}")
    for k, v in reasons.most_common():
        print(f"  dropped {k:<26}: {v:,}")
    print(f"Kept      : {len(kept):,}  -> {args.output}")


if __name__ == "__main__":
    main()
