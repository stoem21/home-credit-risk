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
# The charts exist for two versions of the chosen model. v2 is the default (its PNGs keep their names);
# use("v3") switches every chart to the v3 result files (analysis/figures.py --version v3, PNGs named *_v3.png).
VERSIONS = {
    "v2": {"main": "lgbm_v2_combo_lr02_x3", "name": "LightGBM v2", "summary": "model_summary.csv",
           "kaggle": "kaggle_vs_cv_v2.csv", "business": "business_v2", "reasons": "reasons_v2_cut06",
           "shap": "global_shap_v2", "credit": "credit_v2"},
    "v3": {"main": "lgbm_v3b_sh90_lr02_x3", "name": "LightGBM v3", "summary": "model_summary_v3.csv",
           "kaggle": "kaggle_vs_cv_v3.csv", "business": "business_v3", "reasons": "reasons_v3_cut06",
           "shap": "global_shap_v3", "credit": "credit_v3"},
    "v3t": {"main": "lgbm_v3b_sh90_lr02_x3_tuned", "name": "LightGBM v3 (tuned)", "summary": "model_summary_v3t.csv",
            "kaggle": "kaggle_vs_cv_v3t.csv", "business": "business_v3t", "reasons": "reasons_v3t_cut06",
            "shap": "global_shap_v3t", "credit": "credit_v3t"},
    "v3m": {"main": "lgbm_v3c_sh90_lr02_x3_tuned_mono", "name": "LightGBM v3 (final)", "summary": "model_summary_v3m.csv",
            "kaggle": "kaggle_vs_cv_v3m.csv", "business": "business_v3m", "reasons": "reasons_v3m_cut06",
            "shap": "global_shap_v3m", "credit": "credit_v3m"},
}
V = VERSIONS["v2"]
CHOSEN = V["main"]
LABEL = {"rule": "Rule (external scores)", "logreg_baseline": "Logistic regression",
         "scorecard_full_no_gender": "Scorecard, no gender", "scorecard_full": "Scorecard",
         "lgbm_v1": "LightGBM v1", "lgbm_v1_no_gender_family": "LightGBM v1, no gender/family",
         "lgbm_v2_combo_lr02_x3": "LightGBM v2, no gender (chosen)",
         "lgbm_v3b_sh90_lr02_x3": "LightGBM v3, no gender (chosen)",
         "lgbm_v3b_sh90_lr02_x3_tuned": "LightGBM v3 tuned, no gender (chosen)",
         "lgbm_v3c_sh90_lr02_x3_tuned_mono": "LightGBM v3 final, no gender (chosen)"}


def use(version: str) -> None:
    """Point every chart at one version's result files."""
    global V, CHOSEN
    V = VERSIONS[version]
    CHOSEN = V["main"]
    if version != "v2":
        LABEL["lgbm_v2_combo_lr02_x3"] = "LightGBM v2, no gender"
    if version in ("v3t", "v3m"):
        LABEL["lgbm_v3b_sh90_lr02_x3"] = "LightGBM v3, no gender"
    if version == "v3m":
        LABEL["lgbm_v1_no_gender_family"] = "LightGBM v1, no gender or family"
        LABEL["lgbm_v3b_sh90_lr02_x3_tuned"] = "LightGBM v3 tuned, no gender"


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
    s = pd.read_csv(P.results / V["summary"]).set_index("model")
    if V is VERSIONS["v3m"]:  # the same models as the README table
        order = ["rule", "logreg_baseline", "scorecard_full_no_gender", "lgbm_v1_no_gender_family",
                 "lgbm_v2_combo_lr02_x3", "lgbm_v3b_sh90_lr02_x3_tuned", CHOSEN]
    else:
        order = ["rule", "logreg_baseline", "scorecard_full_no_gender", "lgbm_v1"]
        order += [m for m in ["lgbm_v2_combo_lr02_x3", "lgbm_v3b_sh90_lr02_x3"] if m in s.index and m != CHOSEN]
        order += [CHOSEN]
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
    _title(ax, "Each model ranks applicants better",
           "Cross-validated AUC, mean of 25 folds, all loans")
    fig.tight_layout()
    return fig


def cv_vs_kaggle(P: Paths):
    t = pd.read_csv(P.results / "kaggle" / V["kaggle"])
    fig, ax = _base((5.6, 5.0))
    lo, hi = 0.775, max(0.791, round(max(t.cv_auc_cash.max(), t.private_auc.max()) + 0.002, 3))
    ax.plot([lo, hi], [lo, hi], color=GRAY, linewidth=1, linestyle="--")
    # label the diagonal at its top end; with the v3 models up there, move it below the line in the middle
    tx = hi - 0.0003 if hi <= 0.791 else lo + 0.45 * (hi - lo)
    ax.text(tx, tx - 0.0022, "Kaggle = validation", color=INK2, fontsize=8, ha="right", rotation=45,
            rotation_mode="anchor")
    ax.scatter(t.cv_auc_cash, t.private_auc, s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    c = t[(t.model == CHOSEN) & t.private_auc.notna()]
    if len(c):  # the chosen model's Kaggle score may not be typed in yet
        c = c.iloc[0]
        ax.annotate("chosen model", (c.cv_auc_cash, c.private_auc), xytext=(-70, 12) if hi <= 0.791 else (-40, -40),
                    textcoords="offset points",
                    fontsize=9, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation AUC (cash loans)", color=INK2, fontsize=9)
    ax.set_ylabel("Kaggle private AUC (hidden test set)", color=INK2, fontsize=9)
    n = int((t.private_auc.notna() & t.cv_auc_cash.notna()).sum())  # models with both scores (the plotted points)
    _title(ax, "Validation and the hidden test rank models alike", f"{n} LightGBM models with both scores")
    fig.tight_layout()
    return fig


def calibration(P: Paths, model: str | None = None):
    r = pd.read_csv(P.results / "calibration" / f"{model or CHOSEN}_reliability.csv")
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
    ax.set_ylabel("Observed rate of payment difficulties (%)", color=INK2, fontsize=9)
    _title(ax, "Predicted risk matches what happened",
           f"{V['name']}, raw out-of-fold predictions, seed 0; dashed line = perfect")
    fig.tight_layout()
    return fig


def profit_curve(P: Paths):
    c = pd.read_csv(P.results / V["business"] / "approval_curve.csv")
    base = c[c.approval_pct == 100].profit_per_applicant.iloc[0]
    fig, ax = _base((7.2, 4.4))
    for m, col in [(CHOSEN, BLUE), ("scorecard_full_no_gender", ORANGE)]:
        d = c[c.model == m]
        ax.plot(d.actual_approval_pct, d.profit_per_applicant / 1000, color=col, linewidth=2, marker="o", markersize=4,
                markeredgecolor=SURFACE, label="My final model" if m == CHOSEN else "Classic scorecard")
    pol = pd.read_csv(P.results / V["business"] / "policies.csv")
    pick = pol[(pol.model == CHOSEN) & (pol.policy == "break-even rule")].iloc[0]
    ax.plot([pick.approval_pct], [pick.profit_per_applicant / 1000], marker="o", markersize=9, color=BLUE,
            markeredgecolor=INK, markeredgewidth=1.2, linestyle="none")
    ax.annotate(f"my proposed cut-off:\napprove {pick.approval_pct:.1f}%", fontsize=8.5, color=INK,
                xy=(pick.approval_pct, pick.profit_per_applicant / 1000), xytext=(62, 85),
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.axhline(base / 1000, color=GRAY, linewidth=1, linestyle="--")
    ax.text(51, base / 1000 + 0.6, "approve everyone", color=INK2, fontsize=8)
    ax.set_xlabel("Share of applicants approved (%)", color=INK2, fontsize=9)
    ax.set_ylabel("Profit per applicant (thousands, data's currency;\naxis does not start at zero)", color=INK2,
                  fontsize=9)
    _title(ax, "Declining the riskiest few percent pays; declining more does not",
           "Back-test on past (approved) applicants; assumed margin 20%, loss 60% of the credit")
    fig.tight_layout()
    return fig


def reason_counterfactual(P: Paths, folder: str | None = None):
    s = pd.read_csv(P.results / (folder or V["reasons"]) / "counterfactual_summary.csv").set_index("rank")
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


def approval_by_group(P: Paths, model: str | None = None):
    g = pd.read_csv(P.results / V["business"] / "policy_groups.csv")
    g = g[g.model == (model or CHOSEN)]
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
    # four-fifths rule of thumb: 80% of the highest approval rate within each attribute
    for xs, vs in [(x[:2], vals[:2]), (x[2:], vals[2:])]:
        ax.hlines(0.8 * max(vs), xs[0] - 0.4, xs[-1] + 0.4, color=INK2, linewidth=1, linestyles="--")
    ax.set_xlabel("Dashed lines: four-fifths of the highest approval rate in each set (a common rule of thumb)",
                  color=INK2, fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Approved (%)", color=INK2, fontsize=9)
    _title(ax, "Who is approved under the proposed cut-off",
           "Approve if predicted risk < 25%; in brackets: share of the group that had trouble repaying")
    fig.tight_layout()
    return fig


def shap_groups(P: Paths, top: int = 12):
    g = pd.read_csv(P.results / V["shap"] / "groups.csv")
    head = g.head(top)
    rest = g.iloc[top:]
    names = list(head.group) + ([f"Other ({len(rest)} groups)"] if len(rest) else [])
    vals = list(head.share_pct) + ([round(rest.share_pct.sum(), 1)] if len(rest) else [])
    fig, ax = _base((8.4, 0.32 * len(names) + 1.6))
    y = list(range(len(names)))[::-1]
    ax.barh(y, vals, height=0.6, color=[BLUE] * len(head) + [GRAY] * (len(names) - len(head)),
            edgecolor=SURFACE, linewidth=2)
    for i, v in zip(y, vals):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, names, color=INK, fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of the model's total effect on predictions (mean |SHAP|, %)", color=INK2, fontsize=9)
    _title(ax, "What the model relies on overall",
           "My final model, 20,000 training applicants; groups as in the reason codes")
    fig.subplots_adjust(top=0.86)
    fig.tight_layout()
    return fig


def shap_dependence(P: Paths):
    d = pd.read_csv(P.results / V["shap"] / "dependence.csv")
    cols = list(dict.fromkeys(d.column))
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, c in zip(axes.flat, cols):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        x = d[(d.column == c) & (d.decile != "missing")]
        ax.axhline(0, color=GRAY, linewidth=1)
        ax.plot(x.x_mid, x.mean_shap, color=BLUE, linewidth=2, marker="o", markersize=4, markeredgecolor=SURFACE)
        ax.set_title(c, fontsize=9, color=INK, loc="left")
    for ax in axes.flat[len(cols):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Effect on risk score\n(log-odds; above 0 = riskier)", color=INK2, fontsize=8.5)
    fig.suptitle("How the six largest inputs move the predicted risk", x=0.01, ha="left", fontsize=12,
                 fontweight="bold", color=INK)
    fig.text(0.01, 0.935, f"{V['name']}: mean SHAP value per tenth of each input (x = the tenth's median value)",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def risk_by_band(P: Paths):
    """Share with payment difficulties in each tenth of predicted risk (final model), against the overall rate."""
    b = pd.read_csv(P.results / V["credit"] / "score_bands.csv")
    d = b[b.model == CHOSEN].sort_values("band")
    avg = (d.applicants * d.difficulty_rate_pct).sum() / d.applicants.sum()
    fig, ax = _base((7.2, 4.2))
    ax.bar(d.band, d.difficulty_rate_pct, width=0.62, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for xi, yi in zip(d.band, d.difficulty_rate_pct):
        ax.text(xi, yi + 0.5, f"{yi:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.axhline(avg, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.6, avg + 0.6, f"average of everyone: {avg:.1f}%\n(each bar would sit here if the model knew nothing)",
            fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(range(1, 11), ["1\nsafest"] + [str(i) for i in range(2, 10)] + ["10\nriskiest"])
    ax.set_ylim(0, 35)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Past applicants sorted by the model's risk score, in ten equal groups", color=INK2, fontsize=9)
    ax.set_ylabel("Had trouble repaying (%)", color=INK2, fontsize=9)
    _title(ax, "Most trouble sits in the groups the model calls risky",
           "Tested on past applicants the model had not seen")
    fig.tight_layout()
    return fig


def declined_vs_caught(P: Paths):
    """Share of all applicants who later struggled that a cut-off turns away, against the share declined."""
    c = pd.read_csv(P.results / V["business"] / "approval_curve.csv")
    fig, ax = _base((7.2, 4.4))
    ax.plot([0, 50], [0, 50], color=GRAY, linewidth=1, linestyle="--")
    ax.text(40, 35, "picking at random", color=INK2, fontsize=8)
    for m, col in [(CHOSEN, BLUE), ("scorecard_full_no_gender", ORANGE)]:
        d = c[c.model == m].sort_values("actual_approval_pct", ascending=False)
        ax.plot(100 - d.actual_approval_pct, d.pct_defaulters_declined, color=col, linewidth=2, marker="o",
                markersize=4, markeredgecolor=SURFACE,
                label="My final model" if m == CHOSEN else "Classic scorecard")
    d = c[(c.model == CHOSEN) & (c.approval_pct == 90)].iloc[0]
    ax.annotate(f"turn away 1 in 10 applicants:\n{d.pct_defaulters_declined:.0f}% of those who later struggled",
                xy=(100 - d.actual_approval_pct, d.pct_defaulters_declined), xytext=(2, 72), fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
    ax.set_xlim(0, 51)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlabel("Share of applicants turned away, riskiest first (%)", color=INK2, fontsize=9)
    ax.set_ylabel("Share of later problem loans turned away (%)", color=INK2, fontsize=9)
    _title(ax, "A few refusals catch many of the later problem loans",
           "Back-test on past (approved) applicants")
    fig.tight_layout()
    return fig


def caught_per_100(P: Paths, approval: int = 90):
    """Of every 100 applicants who later struggled, how many a cut-off turns away when 1 in 10 is declined."""
    c0 = pd.read_csv(P.results / V["business"] / "approval_curve.csv")
    c = c0[c0.approval_pct == approval]
    rows = [("Pick at random", 100 - c[c.model == CHOSEN].actual_approval_pct.iloc[0], GRAY),
            ("Classic scorecard", c[c.model == "scorecard_full_no_gender"].pct_defaulters_declined.iloc[0], ORANGE),
            ("My final model", c[c.model == CHOSEN].pct_defaulters_declined.iloc[0], BLUE)]
    fig, ax = _base((7.2, 3.4))
    y = list(range(len(rows)))
    ax.barh(y, [v for _, v, _ in rows], height=0.55, color=[col for _, _, col in rows], edgecolor=SURFACE,
            linewidth=2)
    for i, (_, v, _) in zip(y, rows):
        ax.text(v + 0.8, i, f"{v:.0f} of 100", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, [n for n, _, _ in rows], color=INK)
    ax.set_xlim(0, 50)
    ax.grid(axis="y", visible=False)
    # of the applicants refused, the share who would have repaid
    total_bad = c0.loc[c0.approval_pct == 100, "bad_rate_approved_pct"].iloc[0]
    m = c[c.model == CHOSEN].iloc[0]
    refused_bad = m.pct_defaulters_declined * total_bad / (100 - m.actual_approval_pct)
    ax.set_xlabel(f"People refused, out of every 100 who later had trouble repaying\n"
                  f"(the price: about {round((100 - refused_bad) / 10)} in 10 people my model refuses would have "
                  f"repaid)", color=INK2, fontsize=9)
    _title(ax, f"Refuse the riskiest 1 in {round(100 / (100 - approval))}",
           "Of every 100 past applicants who later had trouble repaying, how many are refused?")
    fig.tight_layout()
    return fig


def profit_gain(P: Paths):
    """Extra profit per applicant against approving everyone, at each model's break-even cut-off."""
    pol = pd.read_csv(P.results / V["business"] / "policies.csv")
    pol = pol[pol.policy == "break-even rule"].set_index("model")
    rows = [("Classic scorecard", "scorecard_full_no_gender", ORANGE), ("My final model", CHOSEN, BLUE)]
    base = pd.read_csv(P.results / V["business"] / "policies.csv").query("policy == 'approve everyone'")
    base = base.profit_per_applicant.iloc[0]
    fig, ax = _base((7.2, 3.0))
    y = list(range(len(rows)))
    vals = [pol.loc[m, "profit_vs_approve_all"] for _, m, _ in rows]
    ax.barh(y, vals, height=0.55, color=[col for _, _, col in rows], edgecolor=SURFACE, linewidth=2)
    for i, v, (_, m, _) in zip(y, vals, rows):
        pct = 100 * v / base
        ax.text(v + 40, i, f"+{v:,.0f} per applicant, +{pct:.1f}% (refuses {100 - pol.loc[m, 'approval_pct']:.1f}%)",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [n for n, _, _ in rows], color=INK)
    ax.set_xlim(0, max(vals) * 1.95)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Extra profit per applicant compared with approving everyone (data's currency)", color=INK2,
                  fontsize=9)
    _title(ax, "Extra profit, if my assumptions are right",
           "Assumed: a repaid loan earns 20%, a loan with trouble loses 60% (past applicants)")
    fig.tight_layout()
    return fig


ALL = {"auc_by_model": auc_by_model, "cv_vs_kaggle": cv_vs_kaggle, "calibration": calibration,
       "profit_curve": profit_curve, "reason_counterfactual": reason_counterfactual,
       "approval_by_group": approval_by_group, "shap_groups": shap_groups, "shap_dependence": shap_dependence,
       "risk_by_band": risk_by_band, "declined_vs_caught": declined_vs_caught,
       "caught_per_100": caught_per_100, "profit_gain": profit_gain}
