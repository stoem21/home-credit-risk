"""Convert the raw Kaggle CSVs to Parquet, unchanged, so later steps load them quickly.

No cleaning happens here: values, columns and row order are exactly as in the CSV. Column types are pandas'
own inference on the whole file (low_memory=False, so a column never gets mixed types from chunked reads).

Usage:
    .venv/bin/python -m pipeline.make_parquet [--out-root DIR]

Writes one .parquet per CSV, plus processed/raw/manifest.csv with rows, columns and the line count of each CSV,
so the row count can be checked against the file (a CSV row could span lines if a quoted field held a newline).
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from homecredit.paths import Paths, add_out_root

FILES = [
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "credit_card_balance",
    "installments_payments",
    "sample_submission",
    "HomeCredit_columns_description",
]


def count_data_lines(path: Path) -> int:
    """Number of lines minus the header line (what `wc -l` reports, minus one)."""
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    raw, out = P.raw_csv, P.raw
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in FILES:
        src = raw / f"{name}.csv"
        t0 = time.time()
        # The column description file is not UTF-8 everywhere; latin-1 reads any byte without failing.
        enc = "latin-1" if name == "HomeCredit_columns_description" else "utf-8"
        df = pd.read_csv(src, low_memory=False, encoding=enc)
        df.to_parquet(out / f"{name}.parquet", index=False)
        lines = count_data_lines(src)
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": df.shape[1],
                "csv_lines_minus_header": lines,
                "rows_match_lines": len(df) == lines,
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(rows[-1], flush=True)

    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)


if __name__ == "__main__":
    main()
