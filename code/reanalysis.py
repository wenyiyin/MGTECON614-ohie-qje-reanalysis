"""Reanalysis of OHIE weighting and panel estimands.

We compute:
  (1) OLS ex-post weights C_i for the ITT regression:
      beta_hat_OLS = sum_i C_i * beta_i, C_i = D_i^perp D_i / sum_j (D_j^perp)^2.
  (2) 2SLS ex-post weights and the BBMT'22 decomposition:
      beta_2SLS = [C(W) tau(W) + C_a(W) tau_a(W)] / E[C(W)+C_a(W)],
      where C(W) = s * E[Z|W](1-L[Z|W]) pi(W) is the complier / "good" weight
      and C_a(W) = (E[Z|W] - L[Z|W]) * pi_a(W) is the always-taker / "bad" weight.
      With saturated stratum FEs, E[Z|W] = L[Z|W] => C_a(W) = 0.
      We also compare unconditional-complier weights P(W) pi(W) with
      covariate-adjusted 2SLS weights P(W) Var(Z|W) pi(W).
  (3) Two-period DiD weights on the balanced panel: trivial under
      random assignment but we report them for completeness.

Outputs: output/reanalysis_decomposition.csv, output/reanalysis_weights.json,
output/late_weighting_diagnostics.csv, plus histograms under figures/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(ROOT / ".mplconfig")
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy

from analysis import build_analysis_data, survey_controls, full_sample_controls


OUT_DIR = ROOT / "output"
FIG_DIR = ROOT / "figures"

OUTCOMES = [
    ("doc_num_mod_12m", "Outpatient visits, last 6m"),
    ("cost_any_oop_12m", "Any out-of-pocket spending"),
    ("health_genflip_bin_12m", "Good/very good/excellent health"),
    ("nodep_screen_12m", "Negative depression screen"),
]


def _drop_rank_deficient(X: np.ndarray, cols: list[str]) -> tuple[np.ndarray, list[str]]:
    keep: list[int] = []
    current = np.empty((X.shape[0], 0))
    for j in range(X.shape[1]):
        cand = np.column_stack([current, X[:, j : j + 1]])
        if np.linalg.matrix_rank(cand) > current.shape[1]:
            keep.append(j)
            current = cand
    return X[:, keep], [cols[j] for j in keep]


def residualize(y: np.ndarray, X: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Return y - X @ beta where beta solves the (weighted) OLS of y on X."""
    if w is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        sw = np.sqrt(w)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return y - X @ beta


def build_design(df: pd.DataFrame, rhs: str) -> tuple[np.ndarray, list[str]]:
    mat = patsy.dmatrix(rhs, df, return_type="dataframe")
    X, cols = _drop_rank_deficient(mat.to_numpy(), list(mat.columns))
    return X, cols


# ---------- OLS ex-post weights ----------

def ols_expost_weights(
    df: pd.DataFrame, outcome: str, treatment: str, controls: str,
    weight_col: str | None,
) -> dict:
    cols_needed = [outcome, treatment]
    cols_needed += [c.strip() for c in controls.replace(":", "+").replace("C(", "").replace(")", "").split("+") if c.strip()]
    if weight_col:
        cols_needed.append(weight_col)
    dat = df[list(dict.fromkeys(cols_needed))].dropna().copy()
    if weight_col:
        dat = dat.loc[dat[weight_col] > 0].copy()
        w = dat[weight_col].to_numpy()
    else:
        w = None

    W, _ = build_design(dat, "1 + " + controls)
    D = dat[treatment].to_numpy().astype(float)
    Y = dat[outcome].to_numpy().astype(float)

    D_perp = residualize(D, W, w)
    # ex-post weights for the (weighted) OLS coefficient on D
        # C_i: weights on individual beta_i under the heterogeneous-effect model.
    if w is None:
        denom_sq = float(np.sum(D_perp ** 2))
        C = D_perp * D / denom_sq
        beta_hat = float(np.sum(D_perp * Y) / denom_sq)
    else:
        denom_sq = float(np.sum(w * D_perp ** 2))
        C = w * D_perp * D / denom_sq
        beta_hat = float(np.sum(w * D_perp * Y) / denom_sq)

    return {
        "n": int(len(dat)),
        "beta": beta_hat,
        "weights": C,
        "D_perp": D_perp,
        "D": D,
        "Y": Y,
        "num_negative": int((C < 0).sum()),
        "share_negative": float((C < 0).mean()),
        "min_weight_x_n": float(C.min() * len(dat)),
        "max_weight_x_n": float(C.max() * len(dat)),
        "effective_n": float(1.0 / np.sum(C ** 2)),
    }


# ---------- 2SLS / BBMT decomposition ----------

def iv_expost_weights(
    df: pd.DataFrame, outcome: str, endog: str, instrument: str,
    controls: str, weight_col: str | None,
) -> dict:
    cols_needed = list(dict.fromkeys([outcome, endog, instrument] + (
        [weight_col] if weight_col else []
    ) + ["draw_survey_12m", "numhh_list"]))
    dat = df[[c for c in cols_needed if c in df.columns]].dropna().copy()
    if weight_col:
        dat = dat.loc[dat[weight_col] > 0].copy()
        w = dat[weight_col].to_numpy()
    else:
        w = None

    W, _ = build_design(dat, "1 + " + controls)
    Z = dat[instrument].to_numpy().astype(float)
    D = dat[endog].to_numpy().astype(float)
    Y = dat[outcome].to_numpy().astype(float)

    Z_perp = residualize(Z, W, w)

    # just-identified 2SLS ex-post weight on individual Delta Y:
    # beta_2SLS = sum_i (w_i Z_i^perp) / E[Z^perp D] * DeltaY_i approximately
    # using the representation beta_2SLS = E[Z^perp Y] / E[Z^perp D]
    if w is None:
        denom = float(np.sum(Z_perp * D))
        weight_on_Y = Z_perp / denom
    else:
        denom = float(np.sum(w * Z_perp * D))
        weight_on_Y = w * Z_perp / denom

    beta = float(np.sum(weight_on_Y * Y))

    return {
        "n": int(len(dat)),
        "beta": beta,
        "Z_perp": Z_perp,
        "Z": Z,
        "D": D,
        "Y": Y,
        "weight_on_Y": weight_on_Y,
        "denominator": denom,
    }


def bbmt_decomposition(
    df: pd.DataFrame, outcome: str, endog: str, instrument: str,
    weight_col: str,
) -> dict:
    """Decompose beta_2SLS into complier and always-taker terms.

    beta_2SLS = E[Z^perp * D * DeltaY] / E[Z^perp D]
              = good + bad,
    good = E[Z^perp * Z * DeltaD * DeltaY] / E[Z^perp D],
    bad  = E[Z^perp * D(0) * DeltaY]       / E[Z^perp D].

    Empirically we cannot observe D(0) for Z=1 units, but under A2 (random Z)
    within strata, we can identify the *weight* on tau_a via:
        C_a(W) = (E[Z|W] - L[Z|W]) * P(G=always | W).
    With saturated W, E[Z|W] = L[Z|W] so C_a = 0 exactly.
    We compute the share of IV numerator coming from D=1 units with Z=0
    (always-takers) vs D=1 units with Z=1 (compliers+always-takers) to
    visualize contamination.
    """
    cols_needed = list(dict.fromkeys([outcome, endog, instrument, weight_col,
                                      "draw_survey_12m", "numhh_list",
                                      "birthyear_list", "female_list"]))
    dat = df[[c for c in cols_needed if c in df.columns]].dropna().copy()
    dat = dat.loc[dat[weight_col] > 0].copy()
    w = dat[weight_col].to_numpy()
    Z = dat[instrument].to_numpy().astype(float)
    D = dat[endog].to_numpy().astype(float)
    Y = dat[outcome].to_numpy().astype(float)
    # center age to avoid scale issues
    dat["age_c"] = dat["birthyear_list"] - dat["birthyear_list"].mean()

    out = {}
    for label, rhs in [
        ("saturated", "C(draw_survey_12m):C(numhh_list)"),
        ("numhh_only", "C(numhh_list)"),  # still saturates the assignment strata
        ("linear_age", "C(numhh_list) + age_c + female_list"),  # deliberately non-saturated
    ]:
        W, _ = build_design(dat, "1 + " + rhs)
        Z_perp = residualize(Z, W, w)
        # projection L[Z|W]:
        L_Z = Z - Z_perp
        # E[Z|W]: within-stratum mean
        # build stratum id
        if label == "saturated":
            strata = (dat["draw_survey_12m"].astype(int).astype(str) + "_"
                      + dat["numhh_list"].astype(int).astype(str))
        else:
            strata = dat["numhh_list"].astype(int).astype(str)
        strata = strata.to_numpy()
        EZ_given_W = np.zeros_like(Z)
        for s in pd.unique(pd.Series(strata)):
            mask = (strata == s)
            ww = w[mask]
            EZ_given_W[mask] = np.sum(ww * Z[mask]) / np.sum(ww)

        # Max discrepancy between E[Z|W] (nonparametric) and L[Z|W] (linear proj)
        disc = float(np.max(np.abs(EZ_given_W - L_Z)))

        # Decompose IV numerator:
        # numer_total = sum w_i Z_i^perp Y_i
        # numer_good  = sum w_i Z_i^perp Z_i DeltaY_i  (compliers+always+never with Z=1)
        # Approx "always-taker share" in numerator:
        #   always-takers contribute via D(0)=1 units, i.e., D=1 with Z=0.
        wZperp = w * Z_perp
        numer_total = float(np.sum(wZperp * Y))
        denom = float(np.sum(wZperp * D))
        # share of denominator from Z=0, D=1 units (always-takers observed side):
        mask_at = (Z == 0) & (D == 1)
        mask_ct = (Z == 1) & (D == 1)  # compliers + always-takers (treated side)
        share_at_denom = float(np.sum(wZperp[mask_at] * D[mask_at]) / denom)
        share_ct_denom = float(np.sum(wZperp[mask_ct] * D[mask_ct]) / denom)

        # Contribution to beta from each cell:
        contrib = {}
        for name, mask in [
            ("Z0_D0", (Z == 0) & (D == 0)),
            ("Z0_D1", (Z == 0) & (D == 1)),  # observed always-takers
            ("Z1_D0", (Z == 1) & (D == 0)),  # never-takers
            ("Z1_D1", (Z == 1) & (D == 1)),  # compliers + always-takers
        ]:
            contrib[name] = float(np.sum(wZperp[mask] * Y[mask]) / denom)

        out[label] = {
            "max_abs_EZ_minus_LZ": disc,
            "beta_2SLS": numer_total / denom,
            "share_denom_always_taker_side": share_at_denom,
            "share_denom_treated_side": share_ct_denom,
            "contrib_to_beta": contrib,
            "pi_a_hat": float(np.average(D[Z == 0], weights=w[Z == 0])),  # P(D=1|Z=0)
            "first_stage": float(np.average(D[Z == 1], weights=w[Z == 1])
                                 - np.average(D[Z == 0], weights=w[Z == 0])),
        }
    return out


def late_weighting_diagnostics(
    df: pd.DataFrame,
    endog: str,
    instrument: str,
    strata_cols: list[str],
    weight_col: str | None,
    scope: str,
) -> tuple[pd.DataFrame, dict]:
    """Compare unconditional-complier and covariate-adjusted 2SLS weights.

    Under monotonicity and saturated controls W, the unconditional LATE weights
    strata by P(W) * pi(W), where pi(W) is the first stage. Covariate-adjusted
    2SLS instead weights strata by P(W) * Var(Z|W) * pi(W). These coincide
    only when Var(Z|W) is constant across strata or tau(W) is homogeneous.
    """
    cols = [endog, instrument] + strata_cols
    if weight_col:
        cols.append(weight_col)
    dat = df[cols].dropna().copy()
    if weight_col:
        dat = dat.loc[dat[weight_col] > 0].copy()
        dat["_analysis_weight"] = dat[weight_col].astype(float)
    else:
        dat["_analysis_weight"] = 1.0

    rows: list[dict] = []
    total_weight = float(dat["_analysis_weight"].sum())
    group_key = strata_cols[0] if len(strata_cols) == 1 else strata_cols
    for key, g in dat.groupby(group_key, dropna=True):
        w = g["_analysis_weight"].to_numpy()
        z = g[instrument].to_numpy().astype(float)
        d = g[endog].to_numpy().astype(float)
        z1 = z == 1
        z0 = z == 0
        if not z1.any() or not z0.any():
            continue

        p_w = float(w.sum() / total_weight)
        p_z = float(np.average(z, weights=w))
        var_z = p_z * (1.0 - p_z)
        first_stage = float(
            np.average(d[z1], weights=w[z1]) - np.average(d[z0], weights=w[z0])
        )
        if isinstance(key, tuple):
            stratum = " x ".join(str(k) for k in key)
        else:
            stratum = str(key)
        rows.append(
            {
                "scope": scope,
                "stratum": stratum,
                "p_w": p_w,
                "p_lottery_given_w": p_z,
                "var_lottery_given_w": var_z,
                "first_stage_pi_w": first_stage,
                "unconditional_mass": p_w * first_stage,
                "covadj_2sls_mass": p_w * var_z * first_stage,
                "n": int(len(g)),
            }
        )

    out = pd.DataFrame(rows)
    out["unconditional_late_weight"] = (
        out["unconditional_mass"] / out["unconditional_mass"].sum()
    )
    out["covadj_2sls_weight"] = out["covadj_2sls_mass"] / out["covadj_2sls_mass"].sum()
    out["weight_difference"] = out["covadj_2sls_weight"] - out["unconditional_late_weight"]
    summary = {
        "n_strata": int(len(out)),
        "total_variation_distance": float(0.5 * out["weight_difference"].abs().sum()),
        "max_abs_weight_difference": float(out["weight_difference"].abs().max()),
        "min_p_lottery_given_w": float(out["p_lottery_given_w"].min()),
        "max_p_lottery_given_w": float(out["p_lottery_given_w"].max()),
    }
    return out, summary


# ---------- Two-period DiD weights ----------

def did_expost_weights(df: pd.DataFrame, y0: str, y12: str,
                       weight_col: str = "weight_12m") -> dict:
    dat = df[[y0, y12, "treatment", weight_col]].dropna().copy()
    dat = dat.loc[dat[weight_col] > 0].copy()
    w = dat[weight_col].to_numpy()
    D = dat["treatment"].to_numpy().astype(float)
    # In a balanced 2-period panel with no dynamics, TWFE reduces to
    # Delta-on-D regression; ex-post weights on individual Delta_i are
    # w_i (D_i - Dbar) / sum w_j (D_j - Dbar)^2 * (D_i - 0) style.
    Dbar = float(np.average(D, weights=w))
    D_tilde = D - Dbar
    denom = float(np.sum(w * D_tilde ** 2))
    C = w * D_tilde * D / denom
    return {
        "n": int(len(dat)),
        "weights": C,
        "num_negative": int((C < 0).sum()),
        "min_weight_x_n": float(C.min() * len(dat)),
        "max_weight_x_n": float(C.max() * len(dat)),
        "effective_n": float(1.0 / np.sum(C ** 2)),
    }


# ---------- Plotting ----------

def plot_weight_histogram(weights: np.ndarray, title: str, path: Path) -> None:
    n = len(weights)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(weights * n, bins=60, color="#1f4e79", edgecolor="white")
    ax.axvline(1.0, color="#c0504d", lw=1.4, label="1/n weight")
    ax.set_xlabel("Ex post weight × sample size")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------- Main ----------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = build_analysis_data()
    survey = df.loc[df["sample_12m_resp"] == 1].copy()

    results: list[dict] = []
    weight_diag: dict[str, dict] = {}

    # (1) OLS ex-post weights on the first-stage regression (ever-Medicaid on lottery, full sample)
    ols_fs = ols_expost_weights(
        df, outcome="ohp_all_ever_admin", treatment="treatment",
        controls="C(draw_lottery) + C(numhh_list)", weight_col=None,
    )
    weight_diag["ols_first_stage"] = {
        k: v for k, v in ols_fs.items() if k not in {"weights", "D_perp", "D", "Y"}
    }
    plot_weight_histogram(
        ols_fs["weights"],
        "Ex-post OLS weights: first stage (ever-Medicaid on lottery)",
        FIG_DIR / "weights_ols_first_stage.pdf",
    )

    # (2) OLS ex-post weights on ITT for each survey outcome
    for outcome, label in OUTCOMES:
        w_diag = ols_expost_weights(
            survey, outcome=outcome, treatment="treatment",
            controls="C(draw_survey_12m):C(numhh_list)", weight_col="weight_12m",
        )
        weight_diag[f"ols_itt_{outcome}"] = {
            k: v for k, v in w_diag.items() if k not in {"weights", "D_perp", "D", "Y"}
        }
        if outcome == "doc_num_mod_12m":  # pick one for the plot
            plot_weight_histogram(
                w_diag["weights"],
                f"Ex-post OLS weights: ITT for {label}",
                FIG_DIR / "weights_ols_itt_docvisits.pdf",
            )

    # (3) 2SLS ex-post weights + BBMT decomposition
    for outcome, label in OUTCOMES:
        iv_w = iv_expost_weights(
            survey, outcome=outcome, endog="ohp_all_ever_survey",
            instrument="treatment", controls="C(draw_survey_12m):C(numhh_list)",
            weight_col="weight_12m",
        )
        bbmt = bbmt_decomposition(
            survey, outcome=outcome, endog="ohp_all_ever_survey",
            instrument="treatment",
            weight_col="weight_12m",
        )
        weight_diag[f"iv_{outcome}"] = {
            "n": iv_w["n"],
            "beta_2SLS": iv_w["beta"],
            "pi_a_hat": bbmt["saturated"]["pi_a_hat"],
            "first_stage": bbmt["saturated"]["first_stage"],
            "bbmt_saturated": bbmt["saturated"],
            "bbmt_numhh_only": bbmt["numhh_only"],
            "bbmt_linear_age": bbmt["linear_age"],
        }
        if outcome == "doc_num_mod_12m":
            plot_weight_histogram(
                iv_w["weight_on_Y"],
                f"2SLS ex-post weights on Y: {label}",
                FIG_DIR / "weights_iv_docvisits.pdf",
            )

    # (4) DiD weights on balanced panel
    balanced = df.loc[(df["returned_0m"] == 1) & (df["sample_12m_resp"] == 1)].copy()
    for y0, y12 in [
        ("doc_num_mod_0m", "doc_num_mod_12m"),
        ("cost_any_oop_0m", "cost_any_oop_12m"),
        ("health_genflip_bin_0m", "health_genflip_bin_12m"),
        ("notbaddays_ment_0m", "notbaddays_ment_12m"),
    ]:
        dd = did_expost_weights(balanced, y0, y12)
        weight_diag[f"did_{y12}"] = {
            k: v for k, v in dd.items() if k != "weights"
        }

    # (5) Covariate-adjusted 2SLS is not the unconditional LATE.
    full_late_weights, full_late_summary = late_weighting_diagnostics(
        df=df,
        endog="ohp_all_ever_admin",
        instrument="treatment",
        strata_cols=["numhh_list"],
        weight_col=None,
        scope="full_by_household_size",
    )
    survey_late_weights, survey_late_summary = late_weighting_diagnostics(
        df=survey,
        endog="ohp_all_ever_survey",
        instrument="treatment",
        strata_cols=["draw_survey_12m", "numhh_list"],
        weight_col="weight_12m",
        scope="survey_by_wave_household_size",
    )
    pd.concat([full_late_weights, survey_late_weights], ignore_index=True).to_csv(
        OUT_DIR / "late_weighting_diagnostics.csv", index=False
    )
    weight_diag["late_weighting"] = {
        "full_by_household_size": full_late_summary,
        "survey_by_wave_household_size": survey_late_summary,
    }

    # Persist
    for key, d in weight_diag.items():
        d_clean = {}
        for k, v in d.items():
            if isinstance(v, dict):
                d_clean[k] = {kk: (float(vv) if isinstance(vv, (np.floating, np.integer))
                                    else vv if not isinstance(vv, dict)
                                    else {kkk: float(vvv) for kkk, vvv in vv.items()})
                               for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                d_clean[k] = float(v)
            else:
                d_clean[k] = v
        weight_diag[key] = d_clean

    with open(OUT_DIR / "reanalysis_weights.json", "w", encoding="utf-8") as f:
        json.dump(weight_diag, f, indent=2, default=float)

    # Compact CSV of BBMT decomposition per outcome
    rows = []
    for outcome, label in OUTCOMES:
        d = weight_diag[f"iv_{outcome}"]
        for spec in ["saturated", "numhh_only", "linear_age"]:
            b = d[f"bbmt_{spec}"]
            rows.append({
                "outcome": outcome,
                "label": label,
                "spec": spec,
                "beta_2SLS": b["beta_2SLS"],
                "max_abs_EZ_minus_LZ": b["max_abs_EZ_minus_LZ"],
                "pi_a_hat_P_D1_given_Z0": b["pi_a_hat"],
                "first_stage": b["first_stage"],
                "share_denom_always_taker_side": b["share_denom_always_taker_side"],
                "share_denom_treated_side": b["share_denom_treated_side"],
                "contrib_Z0_D1": b["contrib_to_beta"]["Z0_D1"],
                "contrib_Z1_D1": b["contrib_to_beta"]["Z1_D1"],
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "reanalysis_decomposition.csv", index=False)

    print("Wrote:")
    print(f"  {OUT_DIR / 'reanalysis_decomposition.csv'}")
    print(f"  {OUT_DIR / 'reanalysis_weights.json'}")
    print(f"  {OUT_DIR / 'late_weighting_diagnostics.csv'}")
    print(f"  {FIG_DIR / 'weights_ols_first_stage.pdf'}")
    print(f"  {FIG_DIR / 'weights_ols_itt_docvisits.pdf'}")
    print(f"  {FIG_DIR / 'weights_iv_docvisits.pdf'}")

    # Print a human summary
    print("\n=== BBMT sanity check (saturated strata) ===")
    for outcome, label in OUTCOMES:
        d = weight_diag[f"iv_{outcome}"]["bbmt_saturated"]
        print(f"  {label}:")
        print(f"    beta_2SLS            = {d['beta_2SLS']:+.4f}")
        print(f"    max|E[Z|W]-L[Z|W]|   = {d['max_abs_EZ_minus_LZ']:.2e}  -> C_a(W)=0")
        print(f"    P(D=1|Z=0)           = {d['pi_a_hat']:.3f}")
        print(f"    first stage          = {d['first_stage']:.3f}")

    print("\n=== BBMT bad-term diagnostic (linear age+female, non-saturated) ===")
    for outcome, label in OUTCOMES:
        d = weight_diag[f"iv_{outcome}"]["bbmt_linear_age"]
        print(f"  {label}: beta={d['beta_2SLS']:+.4f}  "
              f"max|E[Z|W]-L[Z|W]|={d['max_abs_EZ_minus_LZ']:.2e}  "
              f"alwaystaker_denom_share={d['share_denom_always_taker_side']:+.4f}")

    print("\n=== LATE weighting diagnostic ===")
    print("  full sample by household size:")
    print(f"    TV distance = {full_late_summary['total_variation_distance']:.3f}, "
          f"max stratum difference = {full_late_summary['max_abs_weight_difference']:.3f}")
    print("  survey sample by wave x household size:")
    print(f"    TV distance = {survey_late_summary['total_variation_distance']:.3f}, "
          f"max stratum difference = {survey_late_summary['max_abs_weight_difference']:.3f}")


if __name__ == "__main__":
    main()
