"""Charts for the README and the notebook, drawn only from saved result files (no model is trained here).

One function per chart; each returns a matplotlib Figure. analysis/figures.py saves them as PNG in figures/;
the notebook calls the same functions, so both always show the same numbers.

Style: one light surface, thin marks, recessive grid, direct labels instead of legends where there is room,
colours from a validated categorical palette (first three slots: blue, orange, aqua), used in that fixed order.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from homecredit.paths import Paths  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9893"
CHOSEN = "lgbm_v2_combo_lr02_x3"
LABEL = {"rule": "Rule (external scores)", "logreg_baseline": "Logistic regression",
         "scorecard_full_no_gender": "Scorecard, no gender", "scorecard_full": "Scorecard",
         "lgbm_v1": "LightGBM v1", "lgbm_v1_no_gender_family": "LightGBM v1, no gender/family",
         CHOSEN: "LightGBM v2, no gender (chosen)"}


def _base(figsize=(7.2, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _title(ax, title, subtitle):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=22, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def auc_by_model(P: Paths):
    s = pd.read_csv(P.results / "model_summary.csv").set_index("model")
    order = ["rule", "logreg_baseline", "scorecard_full_no_gender", "lgbm_v1", CHOSEN]
    v = s.loc[order, "cv_auc_all"]
    fig, ax = _base((7.2, 3.6))
    y = range(len(order))
    ax.barh(list(y), v.values - 0.70, left=0.70, height=0.55, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for i, val in zip(y, v.values):
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [LABEL[m] for m in order], color=INK)
    ax.set_xlim(0.70, 0.805)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AUC (axis starts at 0.70)", color=INK2, fontsize=9)
    _title(ax, "Each model ranks applicants better than the last",
           "Cross-validated AUC, mean of 25 folds, all loans")
    fig.tight_layout()
    return fig


def cv_vs_kaggle(P: Paths):
    t = pd.read_csv(P.results / "kaggle" / "kaggle_vs_cv.csv")
    fig, ax = _base((5.6, 5.0))
    lo, hi = 0.775, 0.791
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=1, linestyle="--")
    ax.text(hi - 0.0003, hi - 0.0025, "Kaggle = validation", color=INK2, fontsize=8, ha="right", rotation=45,
            rotation_mode="anchor")
    ax.scatter(t.cv_auc_cash, t.private_auc, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    c = t[t.model == CHOSEN].iloc[0]
    ax.annotate("chosen model", (c.cv_auc_cash, c.private_auc), xytext=(-70, 12), textcoords="offset points",
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation AUC (cash loans)", color=INK2, fontsize=9)
    ax.set_ylabel("Kaggle private AUC (hidden test set)", color=INK2, fontsize=9)
    _title(ax, "Validation predicted the hidden test score", f"{len(t)} LightGBM models submitted to Kaggle")
    fig.tight_layout()
    return fig


def calibration(P: Paths, model: str = CHOSEN):
    r = pd.read_csv(P.results / "calibration" / f"{model}_reliability.csv")
    r = r[r.scope == "all"]
    fig, ax = _base((5.6, 5.0))
    top = max(r.mean_pred.max(), r.observed.max()) * 100 * 1.05
    ax.plot([0, top], [0, top], color=GRAY, linewidth=1, linestyle="--")
    d = r[r.version == "raw"]
    ax.plot(d.mean_pred * 100, d.observed * 100, color=BLUE, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Predicted risk (%), 20 equal-size groups", color=INK2, fontsize=9)
    ax.set_ylabel("Observed default rate (%)", color=INK2, fontsize=9)
    _title(ax, "Predicted risk matches observed defaults",
           "LightGBM v2, raw out-of-fold predictions, seed 0; dashed line = perfect")
    fig.tight_layout()
    return fig


def profit_curve(P: Paths):
    c = pd.read_csv(P.results / "business_v2" / "approval_curve.csv")
    base = c[c.approval_pct == 100].profit_per_applicant.iloc[0]
    fig, ax = _base((7.2, 4.4))
    for m, col in [(CHOSEN, BLUE), ("scorecard_full_no_gender", ORANGE)]:
        d = c[c.model == m]
        ax.plot(d.actual_approval_pct, d.profit_per_applicant / 1000, color=col, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, label=LABEL[m])
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.axhline(base / 1000, color=GRAY, linewidth=1, linestyle="--")
    ax.text(51, base / 1000 + 0.6, "approve everyone", color=INK2, fontsize=8)
    ax.set_xlabel("Share of applicants approved (%)", color=INK2, fontsize=9)
    ax.set_ylabel("Profit per applicant (thousands, data's currency)", color=INK2, fontsize=9)
    _title(ax, "Declining the riskiest few percent pays; declining more does not",
           "Back-test on past (approved) applicants; assumed margin 20%, loss 60% of the credit")
    fig.tight_layout()
    return fig


def reason_counterfactual(P: Paths, folder: str = "reasons_v2_cut06"):
    s = pd.read_csv(P.results / folder / "counterfactual_summary.csv").set_index("rank")
    order = ["top1", "top2", "top3", "top4", "control"]
    names = ["1st reason", "2nd reason", "3rd reason", "4th reason", "a random\nlow-impact group"]
    fig, ax = _base((7.0, 3.8))
    vals = s.loc[order, "flip_to_approve_pct"].values
    cols = [BLUE, BLUE, BLUE, GRAY, GRAY]
    ax.bar(names, vals, color=cols, width=0.55, edgecolor=SURFACE, linewidth=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 95)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Declines that flip to approve (%)", color=INK2, fontsize=9)
    _title(ax, "The stated reasons really drive the decision",
           "Declines at the proposed cut-off; replace one reason group with an approved applicant's values")
    fig.tight_layout()
    return fig


def approval_by_group(P: Paths, model: str = CHOSEN):
    g = pd.read_csv(P.results / "business_v2" / "policy_groups.csv")
    g = g[g.model == model]
    order = [("gender", "F", "Women"), ("gender", "M", "Men"), ("age band", "<25", "Under 25"),
             ("age band", "25-34", "25-34"), ("age band", "35-44", "35-44"), ("age band", "45-54", "45-54"),
             ("age band", "55-64", "55-64"), ("age band", "65+", "65+")]
    vals = [g[(g.attribute == a) & (g.group == k)].approval_pct.iloc[0] for a, k, _ in order]
    rates = [g[(g.attribute == a) & (g.group == k)].default_rate_pct.iloc[0] for a, k, _ in order]
    fig, ax = _base((7.2, 3.8))
    x = list(range(len(order)))
    x = [0, 1] + [i + 0.6 for i in range(2, len(order))]  # a gap between gender and age
    ax.bar(x, vals, color=BLUE, width=0.6, edgecolor=SURFACE, linewidth=2)
    for i, v in zip(x, vals):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{n}\n({r:.1f}%)" for (_, _, n), r in zip(order, rates)], color=INK, fontsize=8.5)
    ax.set_ylim(80, 101)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Approved (%), axis starts at 80", color=INK2, fontsize=9)
    _title(ax, "Who is approved under the proposed cut-off",
           "LightGBM v2, break-even rule (approve if risk < 25%); default rate of each group in brackets")
    fig.tight_layout()
    return fig


ALL = {"auc_by_model": auc_by_model, "cv_vs_kaggle": cv_vs_kaggle, "calibration": calibration,
       "profit_curve": profit_curve, "reason_counterfactual": reason_counterfactual,
       "approval_by_group": approval_by_group}
