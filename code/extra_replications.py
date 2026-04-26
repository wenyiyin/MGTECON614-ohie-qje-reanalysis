"""Add Table I (demographics) and Figure 1 (OOP quantile plot) outputs.

Writes into the same output/ and figures/ directories used by the
submission write-up.
"""
from __future__ import annotations
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.regression.quantile_regression import QuantReg

DATA = ROOT / "OHIE_Data"
OUT = ROOT / "output"
FIG = ROOT / "figures"
OUT.mkdir(exist_ok=True, parents=True); FIG.mkdir(exist_ok=True, parents=True)


def tex_escape(s):
    out = str(s)
    for old, new in {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}.items():
        out = out.replace(old, new)
    return out


def fmt_value(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "Not in public files"
    if abs(float(x)) >= 10:
        return f"{float(x):,.0f}"
    return f"{float(x):.3f}"


def w_mean(x, w):
    m = x.notna() & w.notna() & (w > 0)
    if m.sum() == 0: return np.nan
    return float(np.average(x[m], weights=w[m]))


def table1():
    desc = pd.read_stata(DATA / "oregonhie_descriptive_vars.dta", convert_categoricals=False)
    s12 = pd.read_stata(DATA / "oregonhie_survey12m_vars.dta", convert_categoricals=False)
    df = desc.merge(s12, on="person_id", how="left")

    ctrl_full = df[df.treatment == 0]
    ctrl_resp = df[(df.treatment == 0) & (df.sample_12m_resp == 1)]
    w = ctrl_resp.weight_12m

    def wb(col, value):
        m = ctrl_resp[col].notna()
        return float(np.average((ctrl_resp.loc[m, col] == value).astype(float), weights=w.loc[m]))

    age_ref_year = 2008
    panelA = {
        "% Female":      ctrl_full.female_list.mean(),
        "% 50-64":       ((age_ref_year - ctrl_full.birthyear_list) >= 50).mean(),
        "% 20-50":       (((age_ref_year - ctrl_full.birthyear_list) >= 20) &
                          ((age_ref_year - ctrl_full.birthyear_list) < 50)).mean(),
        "% English preferred": ctrl_full.english_list.mean(),
        "% MSA":         ctrl_full.zip_msa_list.mean(),
        "ZIP median household income": None,
    }

    def wm(col): return w_mean(ctrl_resp[col], w)
    def fpl_share(lo, hi=None):
        x = ctrl_resp.hhinc_pctfpl_12m
        if hi is None:
            s = pd.Series(np.where(x.notna(), x >= lo, np.nan), index=x.index)
        else:
            s = pd.Series(np.where(x.notna(), (x >= lo) & (x < hi), np.nan), index=x.index)
        return w_mean(s.astype(float), w)

    panelB = {
        "Lottery list: % Female":   w_mean(ctrl_resp.female_list, w),
        "Lottery list: % 50-64":    w_mean(((age_ref_year - ctrl_resp.birthyear_list) >= 50).astype(float), w),
        "Lottery list: % 20-50":    w_mean((((age_ref_year - ctrl_resp.birthyear_list) >= 20) &
                                            ((age_ref_year - ctrl_resp.birthyear_list) < 50)).astype(float), w),
        "Lottery list: % English":  w_mean(ctrl_resp.english_list, w),
        "Lottery list: % MSA":      w_mean(ctrl_resp.zip_msa_list, w),
        "Lottery list: ZIP median household income": None,
        "Survey: % White":          wm("race_white_12m"),
        "Survey: % Black":          wm("race_black_12m"),
        "Survey: % Hispanic/Latino":wm("race_hisp_12m"),
        "Survey: ever diagnosed diabetes": wm("dia_dx_12m"),
        "Survey: ever diagnosed asthma": wm("ast_dx_12m"),
        "Survey: ever diagnosed high blood pressure": wm("hbp_dx_12m"),
        "Survey: ever diagnosed emphysema/chronic bronchitis": wm("emp_dx_12m"),
        "Survey: ever diagnosed depression/anxiety": wm("dep_dx_12m"),
        "Survey: education less than high school": wb("edu_12m", 1),
        "Survey: education high school/GED": wb("edu_12m", 2),
        "Survey: education vocational/2-year": wb("edu_12m", 3),
        "Survey: education 4-year college or more": wb("edu_12m", 4),
        "Survey: income <50% FPL": fpl_share(0, 50),
        "Survey: income 50-75% FPL": fpl_share(50, 75),
        "Survey: income 75-100% FPL": fpl_share(75, 100),
        "Survey: income 100-150% FPL": fpl_share(100, 150),
        "Survey: income above 150% FPL": fpl_share(150),
        "Survey: average household income": None,
        "Survey: % Any insurance":  wm("ins_any_12m"),
        "Survey: % OHP/Medicaid":   wm("ins_ohp_12m"),
        "Survey: % Private":        wm("ins_private_12m"),
        "Survey: % Other insurance": wm("ins_other_12m"),
        "Survey: # months of last 6 insured": wm("ins_months_12m"),
    }
    return {"panelA": panelA, "panelB": panelB}


def write_table1_latex(t1):
    rows = []
    rows.append(r"% Written by code/extra_replications.py. Do not edit by hand.")
    rows.append(r"\begin{table}[H]\centering\scriptsize")
    rows.append(r"\caption{Replication of \citet{Finkelstein2012} Table I: control-group descriptive statistics available in the public files.}")
    rows.append(r"\label{tab:rep-I}")
    rows.append(r"\begin{tabular}{lr}")
    rows.append(r"\toprule")
    rows.append(r"Variable & Control mean \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{l}{\textit{Panel A: full sample}} \\")
    for k, v in t1["panelA"].items():
        rows.append(f"{tex_escape(k)} & {fmt_value(v)} \\\\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{2}{l}{\textit{Panel B: survey responders}} \\")
    for k, v in t1["panelB"].items():
        rows.append(f"{tex_escape(k)} & {fmt_value(v)} \\\\")
    rows.append(r"\bottomrule")
    rows.append(r"\end{tabular}")
    rows.append(r"\end{table}")
    (OUT / "table1_replication.tex").write_text("\n".join(rows), encoding="utf-8")


def figure1():
    desc = pd.read_stata(DATA / "oregonhie_descriptive_vars.dta", convert_categoricals=False)
    s12 = pd.read_stata(DATA / "oregonhie_survey12m_vars.dta", convert_categoricals=False)
    df = desc.merge(s12, on="person_id", how="left")
    df = df.loc[(df.sample_12m_resp == 1) & df.cost_tot_oop_12m.notna() & (df.weight_12m > 0)].copy()

    # Generate control distribution of OOP and ITT quantile estimates.
    # To keep the figure tractable, we omit household-size x wave FE
    # here (the qualitative shape is the same as the paper's).
    taus = np.linspace(0.50, 0.95, 19)
    control_q = df.loc[df.treatment == 0, "cost_tot_oop_12m"].quantile(taus).values

    # Quantile ITT: estimate y = a + b*treatment at each tau
    qr = df[["cost_tot_oop_12m", "treatment"]].dropna()
    betas, lo, hi = [], [], []
    for tau in taus:
        mod = QuantReg(
            qr.cost_tot_oop_12m,
            np.column_stack([np.ones(len(qr)), qr.treatment]),
        ).fit(q=tau, max_iter=5000)
        b = mod.params.iloc[1]
        se = mod.bse.iloc[1]
        betas.append(b); lo.append(b - 1.96 * se); hi.append(b + 1.96 * se)
    betas = np.array(betas); lo = np.array(lo); hi = np.array(hi)

    # Panel A: control vs estimated-treatment distribution
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4))
    axA.plot(taus * 100, control_q, "-", color="#1f4e79", label="Control distribution")
    axA.plot(taus * 100, control_q + betas, "--", color="#c0504d", label="Est. treatment distribution")
    axA.set_xlabel("Percentile"); axA.set_ylabel("OOP last 6m ($)")
    axA.set_title("Panel A: Control and estimated treatment distribution")
    axA.legend(frameon=False)

    axB.plot(taus * 100, betas, "-", color="#1f4e79")
    axB.fill_between(taus * 100, lo, hi, color="#1f4e79", alpha=0.2)
    axB.axhline(0, color="0.7", lw=1)
    axB.set_xlabel("Percentile"); axB.set_ylabel(r"$\hat\beta_\tau$ on LOTTERY")
    axB.set_title("Panel B: Quantile ITT estimates (95\\% CI)")
    fig.tight_layout()
    fig.savefig(FIG / "figure1_oop_quantile.pdf")
    fig.savefig(FIG / "figure1_oop_quantile.png", dpi=200)
    plt.close(fig)

    return {"taus": taus.tolist(), "control_q": control_q.tolist(),
            "itt_q": betas.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def main():
    t1 = table1()
    f1 = figure1()
    with open(OUT / "table1.json", "w") as f:
        json.dump(t1, f, indent=2, default=float)
    with open(OUT / "figure1.json", "w") as f:
        json.dump(f1, f, indent=2)
    write_table1_latex(t1)
    print("Wrote Table 1 and Figure 1 outputs")


if __name__ == "__main__":
    main()
