from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / ".mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS


DATA_DIR = ROOT / "OHIE_Data"
OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "figures"


PAPER_LATE = {
    "ever_medicaid_full_sample": 0.25,
    "doc_num_mod_12m": 1.083,
    "cost_any_oop_12m": -0.200,
    "health_genflip_bin_12m": 0.130,
    "nodep_screen_12m": 0.078,
}

LABELS = {
    "ever_medicaid_full_sample": "Ever on Medicaid",
    "doc_num_mod_12m": "Outpatient visits, last 6m",
    "cost_any_oop_12m": "Any out-of-pocket spending",
    "health_genflip_bin_12m": "Good/very good/excellent health",
    "nodep_screen_12m": "Negative depression screen",
    "notbaddays_ment_12m": "Good mental-health days",
}


@dataclass
class Estimate:
    outcome: str
    sample: str
    coef: float
    se: float
    pvalue: float
    control_mean: float
    n: int
    model: str


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    mask = values.notna()
    if weights is None:
        return float(values.loc[mask].mean())
    wmask = mask & weights.notna() & (weights > 0)
    return float(np.average(values.loc[wmask], weights=weights.loc[wmask]))


def recode_flip(series: pd.Series) -> pd.Series:
    out = series.copy()
    return out.map({1: 0, 0: 1}).astype(float)


def build_analysis_data() -> pd.DataFrame:
    desc = pd.read_stata(
        DATA_DIR / "oregonhie_descriptive_vars.dta", convert_categoricals=False
    )
    survey0 = pd.read_stata(
        DATA_DIR / "oregonhie_survey0m_vars.dta", convert_categoricals=False
    )
    survey12 = pd.read_stata(
        DATA_DIR / "oregonhie_survey12m_vars.dta", convert_categoricals=False
    )
    state = pd.read_stata(
        DATA_DIR / "oregonhie_stateprograms_vars.dta", convert_categoricals=False
    )

    df = (
        desc.merge(survey0, on="person_id", how="left", suffixes=("", "_0mdup"))
        .merge(survey12, on="person_id", how="left", suffixes=("", "_12mdup"))
        .merge(state, on="person_id", how="left")
    )

    df["draw_survey_12m"] = df["wave_survey12m"]
    df["ohp_all_ever_admin"] = df["ohp_all_ever_matchn_30sep2009"]
    df["ohp_all_ever_survey"] = df["ohp_all_ever_firstn_30sep2009"]
    df["sample_12m_resp"] = (
        (df["returned_12m"] == 1) & df["weight_12m"].notna() & (df["weight_12m"] > 0)
    ).astype(float)

    for wave in ["0m", "12m"]:
        df[f"health_genflip_bin_{wave}"] = recode_flip(df[f"health_gen_bin_{wave}"])
        df[f"health_chgflip_bin_{wave}"] = recode_flip(df[f"health_chg_bin_{wave}"])
        if wave == "12m":
            poor = (df[f"health_gen_{wave}"] == 1).astype(float)
            poor[df[f"health_gen_{wave}"].isna()] = np.nan
            df["health_notpoor_12m"] = recode_flip(poor)
        for stub in ["tot", "phys", "ment"]:
            df[f"notbaddays_{stub}_{wave}"] = 30 - df[f"baddays_{stub}_{wave}"]

    dep_screen_12m = (
        (df["dep_interest_12m"] + df["dep_sad_12m"]) >= 5
    ).astype(float)
    dep_screen_12m[
        df[["dep_interest_12m", "dep_sad_12m"]].isna().any(axis=1)
    ] = np.nan
    df["nodep_screen_12m"] = recode_flip(dep_screen_12m)

    numeric_cols = [
        "treatment",
        "numhh_list",
        "draw_lottery",
        "draw_survey_12m",
        "weight_12m",
        "household_id",
        "returned_0m",
        "returned_12m",
        "ohp_all_ever_admin",
        "ohp_all_ever_survey",
        "doc_num_mod_0m",
        "doc_num_mod_12m",
        "cost_any_oop_0m",
        "cost_any_oop_12m",
        "health_genflip_bin_0m",
        "health_genflip_bin_12m",
        "notbaddays_ment_0m",
        "notbaddays_ment_12m",
        "nodep_screen_12m",
    ]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def survey_controls() -> str:
    # A saturated wave-by-household-size control set spans the same space as
    # wave FE, hh-size FE, and their interaction, while avoiding collinearity
    # issues in the IV design matrix.
    return "C(draw_survey_12m):C(numhh_list)"


def full_sample_controls() -> str:
    return "C(draw_lottery) + C(numhh_list)"


def fit_itt(
    df: pd.DataFrame,
    outcome: str,
    controls: str,
    sample_name: str,
    weight_col: str | None,
    extra_rhs: str = "",
) -> Estimate:
    needed = [outcome, "treatment", "household_id"]
    if weight_col:
        needed.append(weight_col)
    if "draw_survey_12m" in controls:
        needed += ["draw_survey_12m", "numhh_list"]
    if "draw_lottery" in controls:
        needed += ["draw_lottery", "numhh_list"]
    if extra_rhs:
        needed += [c.strip() for c in extra_rhs.split("+")]
    dat = df[needed].copy().dropna()
    if weight_col:
        dat = dat.loc[dat[weight_col] > 0].copy()
    rhs = f"treatment + {controls}"
    if extra_rhs:
        rhs = f"{rhs} + {extra_rhs}"
    formula = f"{outcome} ~ {rhs}"
    if weight_col:
        model = smf.wls(formula, data=dat, weights=dat[weight_col])
    else:
        model = smf.ols(formula, data=dat)
    res = model.fit(
        cov_type="cluster", cov_kwds={"groups": dat["household_id"]}, use_t=True
    )
    control = weighted_mean(
        dat.loc[dat["treatment"] == 0, outcome],
        dat.loc[dat["treatment"] == 0, weight_col] if weight_col else None,
    )
    return Estimate(
        outcome=outcome,
        sample=sample_name,
        coef=float(res.params["treatment"]),
        se=float(res.bse["treatment"]),
        pvalue=float(res.pvalues["treatment"]),
        control_mean=control,
        n=int(res.nobs),
        model="itt",
    )


def fit_iv(
    df: pd.DataFrame,
    outcome: str,
    insurance: str,
    controls: str,
    sample_name: str,
    weight_col: str | None,
    extra_rhs: str = "",
) -> Estimate:
    needed = [outcome, insurance, "treatment", "household_id"]
    if weight_col:
        needed.append(weight_col)
    if "draw_survey_12m" in controls:
        needed += ["draw_survey_12m", "numhh_list"]
    if "draw_lottery" in controls:
        needed += ["draw_lottery", "numhh_list"]
    if extra_rhs:
        needed += [c.strip() for c in extra_rhs.split("+")]
    dat = df[needed].copy().dropna()
    if weight_col:
        dat = dat.loc[dat[weight_col] > 0].copy()
        weights = dat[weight_col]
    else:
        weights = None
    rhs = f"1 + {controls}"
    if extra_rhs:
        rhs = f"{rhs} + {extra_rhs}"

    exog = patsy.dmatrix(rhs, dat, return_type="dataframe")
    keep_cols: list[str] = []
    current = np.empty((len(exog), 0))
    for col in exog.columns:
        candidate = np.column_stack([current, exog[[col]].to_numpy()])
        if np.linalg.matrix_rank(candidate) > current.shape[1]:
            keep_cols.append(col)
            current = candidate
    exog = exog[keep_cols]

    res = IV2SLS(
        dependent=dat[outcome],
        exog=exog,
        endog=dat[[insurance]],
        instruments=dat[["treatment"]],
        weights=weights,
    ).fit(
        cov_type="clustered", clusters=dat["household_id"]
    )
    control = weighted_mean(
        dat.loc[dat["treatment"] == 0, outcome],
        dat.loc[dat["treatment"] == 0, weight_col] if weight_col else None,
    )
    return Estimate(
        outcome=outcome,
        sample=sample_name,
        coef=float(res.params[insurance]),
        se=float(res.std_errors[insurance]),
        pvalue=float(res.pvalues[insurance]),
        control_mean=control,
        n=int(res.nobs),
        model="late",
    )


def fit_did(
    df: pd.DataFrame,
    y0: str,
    y12: str,
    sample_name: str,
    weight_col: str | None,
) -> Estimate:
    needed = [y0, y12, "treatment", "household_id"]
    if weight_col:
        needed.append(weight_col)
    dat = df[needed].copy().dropna()
    if weight_col:
        dat = dat.loc[dat[weight_col] > 0].copy()
    dat["delta"] = dat[y12] - dat[y0]
    formula = "delta ~ treatment"
    if weight_col:
        model = smf.wls(formula, data=dat, weights=dat[weight_col])
    else:
        model = smf.ols(formula, data=dat)
    res = model.fit(
        cov_type="cluster", cov_kwds={"groups": dat["household_id"]}, use_t=True
    )
    control = weighted_mean(
        dat.loc[dat["treatment"] == 0, "delta"],
        dat.loc[dat["treatment"] == 0, weight_col] if weight_col else None,
    )
    return Estimate(
        outcome=y12,
        sample=sample_name,
        coef=float(res.params["treatment"]),
        se=float(res.bse["treatment"]),
        pvalue=float(res.pvalues["treatment"]),
        control_mean=control,
        n=int(res.nobs),
        model="did_itt",
    )


def run_replication(df: pd.DataFrame) -> pd.DataFrame:
    survey = df.loc[df["sample_12m_resp"] == 1].copy()

    results: list[Estimate] = []
    results.append(
        fit_itt(
            df=df,
            outcome="ohp_all_ever_admin",
            controls=full_sample_controls(),
            sample_name="full_sample",
            weight_col=None,
        )
    )
    for outcome in [
        "doc_num_mod_12m",
        "cost_any_oop_12m",
        "health_genflip_bin_12m",
        "nodep_screen_12m",
    ]:
        results.append(
            fit_itt(
                df=survey,
                outcome=outcome,
                controls=survey_controls(),
                sample_name="survey_12m",
                weight_col="weight_12m",
            )
        )
        results.append(
            fit_iv(
                df=survey,
                outcome=outcome,
                insurance="ohp_all_ever_survey",
                controls=survey_controls(),
                sample_name="survey_12m",
                weight_col="weight_12m",
            )
        )

    out = pd.DataFrame([vars(r) for r in results])
    out["paper_late"] = out["outcome"].map(PAPER_LATE)
    out.loc[out["outcome"] == "ohp_all_ever_admin", "paper_late"] = PAPER_LATE[
        "ever_medicaid_full_sample"
    ]
    return out


def run_reanalysis(df: pd.DataFrame) -> pd.DataFrame:
    balanced = df.loc[(df["returned_0m"] == 1) & (df["sample_12m_resp"] == 1)].copy()

    specs = [
        ("doc_num_mod_0m", "doc_num_mod_12m"),
        ("cost_any_oop_0m", "cost_any_oop_12m"),
        ("health_genflip_bin_0m", "health_genflip_bin_12m"),
        ("notbaddays_ment_0m", "notbaddays_ment_12m"),
    ]

    rows: list[dict[str, float | str | int]] = []
    for y0, y12 in specs:
        original_itt = fit_itt(
            df=df.loc[df["sample_12m_resp"] == 1].copy(),
            outcome=y12,
            controls=survey_controls(),
            sample_name="original_12m",
            weight_col="weight_12m",
        )
        original_iv = fit_iv(
            df=df.loc[df["sample_12m_resp"] == 1].copy(),
            outcome=y12,
            insurance="ohp_all_ever_survey",
            controls=survey_controls(),
            sample_name="original_12m",
            weight_col="weight_12m",
        )
        adj_itt = fit_itt(
            df=balanced,
            outcome=y12,
            controls=survey_controls(),
            sample_name="balanced_0m_12m",
            weight_col="weight_12m",
            extra_rhs=y0,
        )
        adj_iv = fit_iv(
            df=balanced,
            outcome=y12,
            insurance="ohp_all_ever_survey",
            controls=survey_controls(),
            sample_name="balanced_0m_12m",
            weight_col="weight_12m",
            extra_rhs=y0,
        )
        did = fit_did(
            df=balanced,
            y0=y0,
            y12=y12,
            sample_name="balanced_0m_12m",
            weight_col="weight_12m",
        )
        rows.append(
            {
                "outcome": y12,
                "label": LABELS[y12],
                "original_itt": original_itt.coef,
                "original_itt_se": original_itt.se,
                "original_iv": original_iv.coef,
                "original_iv_se": original_iv.se,
                "adj_itt": adj_itt.coef,
                "adj_itt_se": adj_itt.se,
                "adj_iv": adj_iv.coef,
                "adj_iv_se": adj_iv.se,
                "did_itt": did.coef,
                "did_itt_se": did.se,
                "n_balanced": did.n,
            }
        )
    return pd.DataFrame(rows)


def make_replication_plot(replication: pd.DataFrame) -> None:
    plot_df = replication.loc[
        (replication["model"] == "late")
        & (replication["outcome"].isin(["doc_num_mod_12m", "cost_any_oop_12m"]))
    ].copy()
    plot_df["paper_late"] = plot_df["outcome"].map(PAPER_LATE)
    plot_df["label"] = plot_df["outcome"].map(LABELS)
    plot_df = plot_df.sort_values("coef")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["coef"],
        y,
        xerr=1.96 * plot_df["se"],
        fmt="o",
        color="#1f4e79",
        label="Replication",
        capsize=4,
    )
    ax.scatter(
        plot_df["paper_late"],
        y,
        marker="D",
        color="#c0504d",
        s=48,
        label="Paper",
        zorder=3,
    )
    ax.axvline(0, color="0.75", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("LATE estimate")
    ax.set_title("Replication lines up with published magnitudes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "replication_plot.pdf")
    fig.savefig(FIG_DIR / "replication_plot.png", dpi=220)
    plt.close(fig)


def make_reanalysis_plot(reanalysis: pd.DataFrame) -> None:
    plot_df = reanalysis.loc[
        reanalysis["outcome"].isin(
            ["doc_num_mod_12m", "cost_any_oop_12m", "health_genflip_bin_12m"]
        )
    ].copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    y = np.arange(len(plot_df))
    offsets = [-0.18, 0.0, 0.18]
    series = [
        ("original_iv", "original_iv_se", "Original IV", "#1f4e79"),
        ("adj_iv", "adj_iv_se", "Baseline-adjusted IV", "#7a9e1f"),
        ("did_itt", "did_itt_se", "0m→12m DiD ITT", "#c0504d"),
    ]
    for offset, (coef_col, se_col, label, color) in zip(offsets, series):
        ax.errorbar(
            plot_df[coef_col],
            y + offset,
            xerr=1.96 * plot_df[se_col],
            fmt="o",
            color=color,
            label=label,
            capsize=4,
        )
    ax.axvline(0, color="0.75", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Estimate")
    ax.set_title("Reanalysis of original IV, adjusted IV, and DiD estimates")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reanalysis_plot.pdf")
    fig.savefig(FIG_DIR / "reanalysis_plot.png", dpi=220)
    plt.close(fig)


def make_summary(replication: pd.DataFrame, reanalysis: pd.DataFrame, df: pd.DataFrame) -> None:
    summary = {
        "sample_sizes": {
            "full_sample": int(len(df)),
            "survey_12m_responders": int((df["sample_12m_resp"] == 1).sum()),
            "balanced_0m_12m": int(
                ((df["returned_0m"] == 1) & (df["sample_12m_resp"] == 1)).sum()
            ),
        },
        "replication": {
            row["outcome"]: {
                "coef": row["coef"],
                "se": row["se"],
                "paper_late": row["paper_late"],
            }
            for _, row in replication.loc[replication["model"] == "late"].iterrows()
        },
        "reanalysis": {
            row["outcome"]: {
                "original_iv": row["original_iv"],
                "adj_iv": row["adj_iv"],
                "did_itt": row["did_itt"],
            }
            for _, row in reanalysis.iterrows()
        },
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = build_analysis_data()
    replication = run_replication(df)
    reanalysis = run_reanalysis(df)

    replication.to_csv(OUT_DIR / "replication_results.csv", index=False)
    reanalysis.to_csv(OUT_DIR / "reanalysis_results.csv", index=False)

    make_replication_plot(replication)
    make_reanalysis_plot(reanalysis)
    make_summary(replication, reanalysis, df)

    print("Wrote:")
    print(f"  {OUT_DIR / 'replication_results.csv'}")
    print(f"  {OUT_DIR / 'reanalysis_results.csv'}")
    print(f"  {OUT_DIR / 'summary.json'}")
    print(f"  {FIG_DIR / 'replication_plot.pdf'}")
    print(f"  {FIG_DIR / 'reanalysis_plot.pdf'}")


if __name__ == "__main__":
    main()
