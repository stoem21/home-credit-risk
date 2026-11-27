"""Save the README charts (homecredit/figures.py) as PNG files in figures/.

Usage:
    .venv/bin/python -m analysis.figures [--version v2|v3] [--out-root DIR]
Output: figures/<name>.png for v2, figures/<name>_v3.png for v3 (in the repository, or under DIR)
"""

import argparse

from homecredit import figures
from homecredit.paths import Paths, add_out_root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=sorted(figures.VERSIONS))
    ap.add_argument("--only", default="", help="comma list of chart names (default: all)")
    add_out_root(ap)
    args = ap.parse_args()
    figures.use(args.version)
    suffix = "" if args.version == "v2" else f"_{args.version}"
    only = [n for n in args.only.split(",") if n]
    P = Paths(args.out_root)
    out = P.root / "figures"
    out.mkdir(exist_ok=True)
    for name, fn in figures.ALL.items():
        if only and name not in only:
            continue
        fig = fn(P)
        fig.savefig(out / f"{name}{suffix}.png", facecolor=fig.get_facecolor())
        print(f"figures/{name}{suffix}.png")


if __name__ == "__main__":
    main()
