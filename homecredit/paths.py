"""All file locations in one place.

Inputs that come from Kaggle (data/raw/*.csv) are always read from the repository. Everything the project writes
(processed/ and analysis_output/) goes under an output root: the repository by default, or another folder given
with --out-root. A smoke test or a reproducibility check therefore never writes next to the real results:
    .venv/bin/python -m pipeline.make_folds --out-root /tmp/hc_check
"""

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


class Paths:
    def __init__(self, out_root: str | Path | None = None):
        self.root = Path(out_root).resolve() if out_root else REPO
        self.raw_csv = REPO / "data" / "raw"  # the Kaggle download (never written)
        # typed by hand from the Kaggle submissions page; an input, so always read from the repository
        self.kaggle_scores = REPO / "analysis_output" / "kaggle" / "kaggle_scores.csv"
        self.processed = self.root / "processed"
        self.raw = self.processed / "raw"  # Parquet copies of the CSVs
        self.folds = self.processed / "folds.parquet"
        self.oof_dir = self.processed / "oof"  # out-of-fold predictions, one column per seed
        self.preds_dir = self.processed / "preds"  # test-set predictions
        self.results = self.root / "analysis_output"

    def features(self, version: str = "v1") -> Path:
        return self.processed / f"features_{version}.parquet"

    def oof(self, name: str) -> Path:
        return self.oof_dir / f"{name}.parquet"

    def test_preds(self, name: str) -> Path:
        return self.preds_dir / f"{name}_test.parquet"

    def model_file(self, name: str) -> Path:
        return self.processed / f"{name}_final.txt"

    def out(self, sub: str) -> Path:
        """analysis_output/<sub>, created if needed."""
        p = self.results / sub
        p.mkdir(parents=True, exist_ok=True)
        return p


def add_out_root(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--out-root", default=None,
                    help="write processed/ and analysis_output/ under this folder instead of the repository "
                         "(smoke tests, reproducibility checks)")
