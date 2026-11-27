"""Paired comparison of two models from their out-of-fold predictions (project step 3 rule).

Both models must have OOF files processed/oof/<name>.parquet with SK_ID_CURR and columns s0..s4 (one per seed of
processed/folds.parquet). For every seed and fold the AUC of A and B is computed on the same validation rows.

Reported for all loans and for cash loans only (homecredit.evaluate.paired_compare):
  - mean AUC of A and B over the 25 folds, the mean gain B - A, and in how many folds B wins;
  - the bootstrap standard error of the gain (200 resamples of applicants, seed-0 OOF predictions);
  - the verdict of the adoption rule: gain > 2 x SE and B wins at least 18 of 25 folds.

Usage:
    .venv/bin/python -m experiments.compare A B [--seeds 0,2,3,4] [--out-root DIR]
Writes analysis_output/comparisons/<A>_vs_<B>.csv (a new file per comparison; with --seeds: ..._seeds<list>.csv).
--seeds compares only those seeds (e.g. leaving out the seed a setting was tuned on).
"""

import argparse

import pandas as pd

from homecredit.data import load_folds, load_target, scopes
from homecredit.evaluate import load_oof, paired_compare
from homecredit.paths import Paths, add_out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--seeds", default=None, help="compare only these seeds, e.g. 0,2,3,4")
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)

    tr = load_target(P)
    folds = load_folds(P, tr.SK_ID_CURR)
    oa = load_oof(P, args.a, tr.SK_ID_CURR)
    ob = load_oof(P, args.b, tr.SK_ID_CURR)
    only = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    rows = paired_compare(tr.TARGET.values, oa, ob, folds, scopes(tr.NAME_CONTRACT_TYPE), only)
    res = pd.DataFrame([{"a": args.a, "b": args.b, **r} for r in rows])
    sfx = f"_seeds{args.seeds.replace(',', '')}" if args.seeds else ""
    res.to_csv(P.out("comparisons") / f"{args.a}_vs_{args.b}{sfx}.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
