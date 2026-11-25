"""Reproducibility check: compare every output file of two runs (e.g. the saved results vs a fresh run of
run_all.sh into another folder).

For each file under processed/ and analysis_output/ of either root:
  - identical bytes -> "identical";
  - otherwise CSV / Parquet tables are compared cell by cell: the columns whose values differ are listed with the
    largest absolute difference (numbers) or the number of differing cells (text);
  - text files (the LightGBM model) are compared line by line.
Files present in only one root are listed as such.

Usage:
    .venv/bin/python -m experiments.check_reproduced NEW_ROOT [--reference REPO_ROOT] [--name LABEL]
Writes analysis_output/reproducibility/<LABEL>.csv in the reference root.
"""

import argparse
import filecmp
from pathlib import Path

import numpy as np
import pandas as pd

from homecredit.paths import Paths

SUBDIRS = ["processed", "analysis_output"]


def table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)


def compare_tables(a: pd.DataFrame, b: pd.DataFrame) -> str:
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return f"rows differ: {len(a)} vs {len(b)}"
    notes = []
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            both_nan = np.isnan(xv) & np.isnan(yv)
            d = np.where(both_nan, 0.0, np.abs(xv - yv))
            if np.isnan(d).any():
                notes.append(f"{c}: missing values differ")
            elif d.max() > 0:
                notes.append(f"{c}: max abs diff {d.max():.3g}")
        else:
            n = int((x.astype(str) != y.astype(str)).sum())
            if n:
                notes.append(f"{c}: {n} cells differ")
    return "; ".join(notes) if notes else "same values (bytes differ)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("new_root")
    ap.add_argument("--reference", default=None, help="root of the saved results (default: the repository)")
    ap.add_argument("--name", default="check")
    args = ap.parse_args()
    ref, new = Paths(args.reference).root, Path(args.new_root).resolve()

    files = set()
    for root in (ref, new):
        for sub in SUBDIRS:
            files |= {p.relative_to(root) for p in (root / sub).rglob("*")
                      if p.is_file() and "reproducibility" not in p.parts and p.name != ".DS_Store"}
    rows = []
    for rel in sorted(files):
        a, b = ref / rel, new / rel
        if not a.exists() or not b.exists():
            status, detail = ("only in new run", "") if b.exists() else ("only in reference", "")
        elif filecmp.cmp(a, b, shallow=False):
            status, detail = "identical", ""
        elif a.suffix in (".csv", ".parquet"):
            status, detail = "differs", compare_tables(table(a), table(b))
        else:
            la, lb = a.read_text().splitlines(), b.read_text().splitlines()
            n = sum(x != y for x, y in zip(la, lb)) + abs(len(la) - len(lb))
            status, detail = "differs", f"{n} of {max(len(la), len(lb))} lines differ"
        rows.append({"file": str(rel), "status": status, "detail": detail})
    res = pd.DataFrame(rows)
    out = Paths(args.reference).out("reproducibility")
    res.to_csv(out / f"{args.name}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 140)
    print(res.status.value_counts().to_string(), "\n")
    print(res[res.status != "identical"].to_string(index=False))


if __name__ == "__main__":
    main()
