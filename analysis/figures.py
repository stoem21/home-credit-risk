"""Save the README charts (homecredit/figures.py) as PNG files in figures/.

Usage:
    .venv/bin/python -m analysis.figures [--out-root DIR]
Output: figures/<name>.png (in the repository, or under DIR)
"""

import argparse

from homecredit.figures import ALL
from homecredit.paths import Paths, add_out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    add_out_root(ap)
    args = ap.parse_args()
    P = Paths(args.out_root)
    out = P.root / "figures"
    out.mkdir(exist_ok=True)
    for name, fn in ALL.items():
        fig = fn(P)
        fig.savefig(out / f"{name}.png", facecolor=fig.get_facecolor())
        print(f"figures/{name}.png")


if __name__ == "__main__":
    main()
