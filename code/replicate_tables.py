"""Replicate survey-data tables of Finkelstein et al. (QJE 2012) from the
public-use files. Produces LaTeX-ready numbers.

Tables that CAN be replicated from public data:
  Table 1  (control demographics; see extra_replications.py)
  Table 3  (first stage; full-sample and survey-respondent columns)
  Table 5  (health-care utilization, survey)
  Table 6  (preventive care, survey)
  Table 8  (financial strain, survey)
  Table 9  (mortality panel A and survey panel B)
  Table 10 (access, quality, happiness, survey)
  Table 11 (standardized survey ITT effects at different survey waves)

Tables that CANNOT be replicated (admin/credit/hospital/zip data not public):
  Table 4  (hospital discharge admin)
  Table 7  (credit-report financial strain)
  Table 2  full F-tests (ZIP median income and credit/hospital pre-outcomes missing)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "OHIE_Data"
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)


def tex_escape(value: str) -> str:
    """Escape text used in generated LaTeX tables."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
    }
    out = str(value)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def fmt_num(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_n(value) -> str:
    return f"{int(value):,}".replace(",", r"{,}")


def p_stars(pvalue) -> str:
    if pvalue is None or pd.isna(pvalue):
        return ""
    if pvalue < 0.01:
        return r"\textsuperscript{***}"
    if pvalue < 0.05:
        return r"\textsuperscript{**}"
    if pvalue < 0.10:
        return r"\textsuperscript{*}"
    return ""


def latex_row(row: dict, label: str) -> str:
    late = f"{fmt_num(row['late'])}{p_stars(row['late_p'])} ({fmt_num(row['late_se'])})"
    return (
        f"{tex_escape(label)} & {fmt_num(row['cmean'])} & {fmt_num(row['itt'])} & "
        f"{fmt_num(row['paper_itt'])} & {late} & {fmt_num(row['paper_late'])} & "
        f"{fmt_n(row['n'])} \\\\"
    )


def fmt_coef_se(coef, se, pvalue=None) -> str:
    return f"{fmt_num(coef)}{p_stars(pvalue)} ({fmt_num(se)})"


def fmt_mine_paper(coef, se, paper, pvalue=None) -> str:
    return f"{fmt_coef_se(coef, se, pvalue)} / {fmt_num(paper)}"


def replication_table_block(caption: str, label: str, rows: list[dict]) -> list[str]:
    lines = [
        r"\begin{table}[H]\centering\scriptsize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Panel & Outcome & Control & ITT & ITT & LATE & LATE & \\",
        r" & & mean & (mine) & (paper) & (mine, SE) & (paper) & $N$ \\",
        r"\midrule",
    ]
    for row in rows:
        panel = tex_escape(row.get("panel", ""))
        late = fmt_coef_se(row["late"], row["late_se"], row.get("late_p")) if "late" in row and not pd.isna(row.get("late")) else ""
        lines.append(
            f"{panel} & {tex_escape(row['label'])} & {fmt_num(row['cmean'])} & "
            f"{fmt_num(row['itt'])} & {fmt_num(row['paper_itt'])} & "
            f"{late} & {fmt_num(row.get('paper_late'))} & {fmt_n(row['n'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    return lines


def write_replication_latex_tables(results: dict) -> None:
    """Generate all public-data replication tables included by the main TeX file."""
    lines = [
        r"% Written by code/replicate_tables.py. Do not edit by hand.",
        "",
        r"\paragraph{Table III: first stage and public-program outcomes.}",
        r"Table~\ref{tab:rep-III} reports the first-stage rows that can be constructed from the public files. The Medicaid and self-reported insurance rows reproduce the published values. The TANF/SNAP rows are reported as public-file estimates; these rows do not match the published values as closely, so I do not use them in the reanalysis. The credit-report column in the published table is not reproduced because those data are not public.",
        "",
        r"\begin{table}[H]\centering\scriptsize",
        r"\caption{Replication of \citet{Finkelstein2012} Table III: first-stage estimates in the public files. Each cell reports mine (SE) / paper.}",
        r"\label{tab:rep-III}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Outcome & Full control & Full FS & Survey control & Survey FS \\",
        r"\midrule",
    ]
    labels = list(dict.fromkeys(row["label"] for row in results["table3"]))
    for label in labels:
        full = next((r for r in results["table3"] if r["label"] == label and r["sample"] == "full_sample"), None)
        survey = next((r for r in results["table3"] if r["label"] == label and r["sample"] == "survey_responders"), None)
        full_control = f"{fmt_num(full['cmean'])} / {fmt_num(full['paper_cmean'])}" if full else ""
        full_fs = fmt_mine_paper(full["itt"], full["itt_se"], full["paper_itt"], full["itt_p"]) if full else ""
        survey_control = f"{fmt_num(survey['cmean'])} / {fmt_num(survey['paper_cmean'])}" if survey else ""
        survey_fs = fmt_mine_paper(survey["itt"], survey["itt_se"], survey["paper_itt"], survey["itt_p"]) if survey else ""
        lines.append(
            f"{tex_escape(label)} & {full_control} & {full_fs} & {survey_control} & {survey_fs} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])

    for key, caption, label in [
        ("table5", r"Replication of \citet{Finkelstein2012} Table V: health-care utilization, survey data.", "tab:rep-V"),
        ("table6", r"Replication of \citet{Finkelstein2012} Table VI: compliance with recommended preventive care.", "tab:rep-VI"),
        ("table8", r"Replication of \citet{Finkelstein2012} Table VIII: financial strain, survey data.", "tab:rep-VIII"),
        ("table9", r"Replication of \citet{Finkelstein2012} Table IX: mortality and self-reported health.", "tab:rep-IX"),
        ("table10", r"Replication of \citet{Finkelstein2012} Table X: access, quality, and happiness.", "tab:rep-X"),
    ]:
        rows = []
        for row in results[key]:
            row = dict(row)
            if key == "table9":
                row["panel"] = row.get("panel", "B")
            else:
                row["panel"] = ""
            rows.append(row)
        lines.extend(replication_table_block(caption, label, rows))

    lines.extend(
        [
            r"\paragraph{Table XI: treatment effects at different survey waves.}",
            r"Table~\ref{tab:rep-XI} reports standardized ITT effects at the initial, six-month, and main survey waves. Each cell reports mine (SE) / paper.",
            "",
            r"\begin{table}[H]\centering\scriptsize",
            r"\caption{Replication of \citet{Finkelstein2012} Table XI: estimated effects of lottery at different times.}",
            r"\label{tab:rep-XI}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Domain & Initial survey & Six-month survey & Main survey \\",
            r"\midrule",
        ]
    )
    for domain in list(dict.fromkeys(row["domain"] for row in results["table11"])):
        rows = [r for r in results["table11"] if r["domain"] == domain]
        by_sample = {r["sample"]: r for r in rows}
        cells = []
        for sample in ["initial", "six_month", "main"]:
            r = by_sample[sample]
            cells.append(fmt_mine_paper(r["itt"], r["itt_se"], r["paper_itt"], r["itt_p"]))
        lines.append(f"{tex_escape(rows[0]['label'])} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])

    (OUT / "replication_results_tables.tex").write_text("\n".join(lines), encoding="utf-8")


def load() -> pd.DataFrame:
    desc = pd.read_stata(DATA / "oregonhie_descriptive_vars.dta", convert_categoricals=False)
    s0 = pd.read_stata(DATA / "oregonhie_survey0m_vars.dta", convert_categoricals=False)
    s6 = pd.read_stata(DATA / "oregonhie_survey6m_vars.dta", convert_categoricals=False)
    s12 = pd.read_stata(DATA / "oregonhie_survey12m_vars.dta", convert_categoricals=False)
    state = pd.read_stata(DATA / "oregonhie_stateprograms_vars.dta", convert_categoricals=False)
    df = (
        desc.merge(s0, on="person_id", how="left")
        .merge(s6, on="person_id", how="left")
        .merge(s12, on="person_id", how="left")
        .merge(state, on="person_id", how="left")
    )
    df["ohp_all_ever_admin"] = df["ohp_all_ever_matchn_30sep2009"]
    df["ohp_all_ever_survey"] = df["ohp_all_ever_firstn_30sep2009"]
    df["alive_30sep2009"] = 1 - df["postn_death"]
    df["draw_survey_0m"] = df["wave_survey0m"]
    df["draw_survey_6m"] = df["wave_survey6m"]
    df["draw_survey_12m"] = df["wave_survey12m"]
    df["sample_resp"] = ((df["returned_12m"] == 1) & df["weight_12m"].notna() & (df["weight_12m"] > 0)).astype(int)
    df["sample_resp_0m"] = (df["returned_0m"] == 1).astype(int)
    df["sample_resp_6m"] = ((df["returned_6m"] == 1) & df["weight_6m"].notna() & (df["weight_6m"] > 0)).astype(int)

    # Flipped binaries so higher values follow the paper's favorable-outcome direction.
    def flip(x): return x.map({1: 0, 0: 1}).astype(float)
    for wave in ["0m", "6m", "12m"]:
        df[f"health_genflip_bin_{wave}"] = flip(df[f"health_gen_bin_{wave}"])
        hg = df[f"health_gen_{wave}"]
        df[f"health_notpoor_{wave}"] = np.where(hg.isna(), np.nan, (hg != 1).astype(float))
        df[f"health_chgflip_bin_{wave}"] = flip(df[f"health_chg_bin_{wave}"])
        for s in ["phys", "ment", "tot"]:
            df[f"notbaddays_{s}_{wave}"] = 30 - df[f"baddays_{s}_{wave}"]
        df[f"notnoner_{wave}"] = flip(df[f"er_noner_{wave}"]) if f"er_noner_{wave}" in df else np.nan

    dep = ((df["dep_interest_12m"] + df["dep_sad_12m"]) >= 5).astype(float)
    dep[df[["dep_interest_12m", "dep_sad_12m"]].isna().any(axis=1)] = np.nan
    df["nodep_screen_12m"] = flip(dep)

    # happiness binary: happy or pretty happy (vs not too happy). happiness_12m: 1=very,2=pretty,3=not too
    hp = df["happiness_12m"]
    df["happy_bin_12m"] = np.where(hp.isna(), np.nan, (hp <= 2).astype(float))

    # Preventive care: chk variables are coded 1=within 1yr, 2=1-5 yrs, 3=never (Table 6).
    # Paper uses "ever checked" = (var in {1,2}) for blood cholesterol & diabetes;
    # and "within last 12 months" = (var == 1) for mammogram and pap test.
    for v in ["chl_chk_12m", "dia_chk_12m"]:
        df[v + "_ever"] = np.where(df[v].isna(), np.nan, (df[v] <= 2).astype(float))
    for v in ["mam_chk_12m", "pap_chk_12m"]:
        df[v + "_12mo"] = np.where(df[v].isna(), np.nan, (df[v] == 1).astype(float))

    # annual spending "implied" not computable; skip
    return df


CONTROLS = "C(draw_survey_12m):C(numhh_list)"
FULL_CONTROLS = "C(draw_lottery) + C(numhh_list)"


def _prep(df, cols, weight="weight_12m"):
    keep = list(set(cols + ["treatment", "household_id", "draw_survey_12m", "numhh_list", weight]))
    d = df[keep].dropna()
    d = d.loc[d[weight] > 0].copy()
    return d


def control_cols(controls: str) -> list[str]:
    cols = ["numhh_list"]
    for col in ["draw_lottery", "draw_survey_0m", "draw_survey_6m", "draw_survey_12m"]:
        if col in controls:
            cols.append(col)
    return cols


def prep_generic(df, y, controls, weight=None):
    keep = [y, "treatment", "household_id"] + control_cols(controls)
    if weight:
        keep.append(weight)
    d = df[list(dict.fromkeys(keep))].dropna().copy()
    if weight:
        d = d.loc[d[weight] > 0].copy()
    return d


def itt_generic(df, y, controls, weight=None):
    d = prep_generic(df, y, controls, weight)
    model = smf.wls(f"{y} ~ treatment + {controls}", data=d, weights=d[weight]) if weight else smf.ols(
        f"{y} ~ treatment + {controls}", data=d
    )
    r = model.fit(cov_type="cluster", cov_kwds={"groups": d["household_id"]}, use_t=True)
    cm = np.average(d.loc[d.treatment == 0, y], weights=d.loc[d.treatment == 0, weight]) if weight else d.loc[d.treatment == 0, y].mean()
    return dict(coef=r.params["treatment"], se=r.bse["treatment"], p=r.pvalues["treatment"], cmean=cm, n=int(r.nobs))


def late_generic(df, y, ins, controls, weight=None):
    d = prep_generic(df, y, controls, weight)
    keep = list(dict.fromkeys(list(d.columns) + [ins]))
    d = df[keep].dropna().copy()
    if weight:
        d = d.loc[d[weight] > 0].copy()
    exog = patsy.dmatrix(f"1 + {controls}", d, return_type="dataframe")
    cols, cur = [], np.empty((len(exog), 0))
    for c in exog.columns:
        cand = np.column_stack([cur, exog[[c]].to_numpy()])
        if np.linalg.matrix_rank(cand) > cur.shape[1]:
            cols.append(c)
            cur = cand
    exog = exog[cols]
    r = IV2SLS(
        dependent=d[y],
        exog=exog,
        endog=d[[ins]],
        instruments=d[["treatment"]],
        weights=d[weight] if weight else None,
    ).fit(cov_type="clustered", clusters=d["household_id"])
    cm = np.average(d.loc[d.treatment == 0, y], weights=d.loc[d.treatment == 0, weight]) if weight else d.loc[d.treatment == 0, y].mean()
    return dict(coef=float(r.params[ins]), se=float(r.std_errors[ins]), p=float(r.pvalues[ins]), cmean=cm, n=int(r.nobs))


def itt(df, y, weight="weight_12m"):
    d = _prep(df, [y], weight)
    f = f"{y} ~ treatment + {CONTROLS}"
    r = smf.wls(f, data=d, weights=d[weight]).fit(cov_type="cluster", cov_kwds={"groups": d["household_id"]}, use_t=True)
    cm = np.average(d.loc[d.treatment == 0, y], weights=d.loc[d.treatment == 0, weight])
    return dict(coef=r.params["treatment"], se=r.bse["treatment"], p=r.pvalues["treatment"], cmean=cm, n=int(r.nobs))


def late(df, y, ins="ohp_all_ever_survey", weight="weight_12m"):
    d = _prep(df, [y, ins], weight)
    exog = patsy.dmatrix(f"1 + {CONTROLS}", d, return_type="dataframe")
    # drop collinear
    keep, cur = [], np.empty((len(exog), 0))
    for c in exog.columns:
        cand = np.column_stack([cur, exog[[c]].to_numpy()])
        if np.linalg.matrix_rank(cand) > cur.shape[1]:
            keep.append(c); cur = cand
    exog = exog[keep]
    r = IV2SLS(dependent=d[y], exog=exog, endog=d[[ins]], instruments=d[["treatment"]], weights=d[weight]).fit(
        cov_type="clustered", clusters=d["household_id"])
    cm = np.average(d.loc[d.treatment == 0, y], weights=d.loc[d.treatment == 0, weight])
    return dict(coef=float(r.params[ins]), se=float(r.std_errors[ins]), p=float(r.pvalues[ins]), cmean=cm, n=int(r.nobs))


def row(df_resp, y, label, subset=None, paper_itt=None, paper_late=None, paper_cmean=None):
    d = df_resp if subset is None else df_resp.loc[subset]
    a = itt(d, y); b = late(d, y)
    return dict(outcome=y, label=label, cmean=a["cmean"], itt=a["coef"], itt_se=a["se"], itt_p=a["p"],
                late=b["coef"], late_se=b["se"], late_p=b["p"], n=a["n"],
                paper_cmean=paper_cmean, paper_itt=paper_itt, paper_late=paper_late)


def first_stage_rows(df: pd.DataFrame, resp: pd.DataFrame) -> list[dict]:
    specs = [
        ("ohp_all_ever_matchn_30sep2009", "ohp_all_ever_firstn_30sep2009", "Ever on Medicaid", 0.141, 0.256, 0.135, 0.290),
        ("ohp_std_ever_matchn_30sep2009", "ohp_std_ever_firstn_30sep2009", "Ever on OHP Standard", 0.027, 0.264, 0.026, 0.302),
        ("ohp_all_mo_matchn_30sep2009", "ohp_all_mo_firstn_30sep2009", "# months on Medicaid", 1.408, 3.355, 1.509, 3.943),
        ("ohp_all_end_30sep2009", "ohp_all_end_30sep2009", "On Medicaid, end of study period", 0.106, 0.148, 0.105, 0.189),
        (None, "ins_any_12m", "Currently have any insurance (self-report)", None, None, 0.325, 0.179),
        (None, "ins_private_12m", "Currently have private insurance (self-report)", None, None, 0.128, -0.0076),
        (None, "ins_ohp_12m", "Currently on Medicaid (self-report)", None, None, 0.117, 0.197),
        (None, "ohp_all_at_12m", "Currently on Medicaid", None, None, 0.105, 0.191),
        ("tanf_ever_matchn_30sep2009", "tanf_ever_firstn_survey12m", "Ever on TANF", 0.031, 0.0011, 0.023, 0.0019),
        ("tanf_tot_hh_30sep2009", "tanf_tot_hh_firstn_survey12m", "TANF benefits ($)", 124, -1.659, 100, -4.991),
        ("snap_ever_matchn_30sep2009", "snap_ever_firstn_survey12m", "Ever on food stamps", 0.606, 0.017, 0.622, 0.023),
        ("snap_tot_hh_30sep2009", "snap_tot_hh_firstn_survey12m", "Food stamp benefits ($)", 1776, 61.3, 2202, 122.4),
    ]
    rows: list[dict] = []
    for full_var, survey_var, label, full_cm, full_itt, survey_cm, survey_itt in specs:
        if full_var is not None:
            r = itt_generic(df, full_var, FULL_CONTROLS)
            rows.append(dict(table="table3", sample="full_sample", outcome=full_var, label=label,
                             cmean=r["cmean"], itt=r["coef"], itt_se=r["se"], itt_p=r["p"], n=r["n"],
                             paper_cmean=full_cm, paper_itt=full_itt))
        r = itt_generic(resp, survey_var, CONTROLS, "weight_12m")
        rows.append(dict(table="table3", sample="survey_responders", outcome=survey_var, label=label,
                         cmean=r["cmean"], itt=r["coef"], itt_se=r["se"], itt_p=r["p"], n=r["n"],
                         paper_cmean=survey_cm, paper_itt=survey_itt))
    return rows


def mortality_row(df: pd.DataFrame) -> dict:
    a = itt_generic(df, "alive_30sep2009", FULL_CONTROLS)
    b = late_generic(df, "alive_30sep2009", "ohp_all_ever_admin", FULL_CONTROLS)
    return dict(table="table9", panel="A", outcome="alive_30sep2009", label="Alive through September 2009",
                cmean=a["cmean"], itt=a["coef"], itt_se=a["se"], itt_p=a["p"],
                late=b["coef"], late_se=b["se"], late_p=b["p"], n=a["n"],
                paper_cmean=0.992, paper_itt=0.00032, paper_late=0.0013)


TABLE11_SPECS = {
    "utilization_extensive": {
        "label": "Utilization (extensive margin)",
        "patterns": ["rx_any_{w}", "doc_any_{w}", "er_any_{w}", "hosp_any_{w}"],
        "paper": {"0m": (0.0038, 0.0084, 0.656), "6m": (0.047, 0.020, 0.020), "12m": (0.050, 0.011, 0.0001)},
        "paper_diff_p": {"0m_vs_6m": 0.035, "6m_vs_12m": 0.867, "0m_vs_12m": 0.0001},
    },
    "utilization_total": {
        "label": "Utilization (total)",
        "patterns": ["rx_num_mod_{w}", "doc_num_mod_{w}", "er_num_mod_{w}", "hosp_num_mod_{w}"],
        "paper": {"0m": (-0.00023, 0.0086, 0.978), "6m": (0.027, 0.020, 0.187), "12m": (0.040, 0.011, 0.0003)},
        "paper_diff_p": {"0m_vs_6m": 0.188, "6m_vs_12m": 0.556, "0m_vs_12m": 0.001},
    },
    "financial_strain": {
        "label": "Financial strain",
        "patterns": ["cost_any_oop_{w}", "cost_any_owe_{w}", "cost_borrow_{w}", "cost_refused_{w}"],
        "paper": {"0m": (-0.035, 0.0089, 0.0001), "6m": (-0.099, 0.020, 0.0001), "12m": (-0.089, 0.010, 0.0001)},
        "paper_diff_p": {"0m_vs_6m": 0.002, "6m_vs_12m": 0.613, "0m_vs_12m": 0.0001},
    },
    "health": {
        "label": "Health",
        "patterns": ["health_genflip_bin_{w}", "health_notpoor_{w}", "health_chgflip_bin_{w}", "notbaddays_tot_{w}", "notbaddays_phys_{w}", "notbaddays_ment_{w}"],
        "paper": {"0m": (0.042, 0.010, 0.0001), "6m": (0.097, 0.023, 0.0001), "12m": (0.061, 0.011, 0.0001)},
        "paper_diff_p": {"0m_vs_6m": 0.014, "6m_vs_12m": 0.121, "0m_vs_12m": 0.112},
    },
    "access": {
        "label": "Access",
        "patterns": ["usual_clinic_{w}", "needmet_med_{w}", "needmet_rx_{w}", "notnoner_{w}"],
        "paper": {"0m": (0.047, 0.0078, 0.0001), "6m": (0.075, 0.019, 0.0001), "12m": (0.119, 0.0086, 0.0001)},
        "paper_diff_p": {"0m_vs_6m": 0.163, "6m_vs_12m": 0.026, "0m_vs_12m": 0.0001},
    },
}


def standardized_effect(df: pd.DataFrame, wave: str, outcomes: list[str], weight: str | None) -> dict:
    controls = f"C(draw_survey_{wave}):C(numhh_list)"
    z_coefs = []
    z_ses = []
    n_values = []
    zcols = []
    keep = ["treatment", "household_id", "numhh_list", f"draw_survey_{wave}"] + outcomes
    if weight:
        keep.append(weight)
    sample = df[f"sample_resp_{wave}"] == 1 if wave in {"0m", "6m"} else df["sample_resp"] == 1
    base = df.loc[sample, keep].copy()
    if weight:
        base = base.loc[base[weight] > 0].copy()

    for outcome in outcomes:
        r = itt_generic(base, outcome, controls, weight)
        d = prep_generic(base, outcome, controls, weight)
        ctl = d["treatment"] == 0
        if weight:
            cm = np.average(d.loc[ctl, outcome], weights=d.loc[ctl, weight])
            sd = np.sqrt(np.average((d.loc[ctl, outcome] - cm) ** 2, weights=d.loc[ctl, weight]))
        else:
            cm = d.loc[ctl, outcome].mean()
            sd = d.loc[ctl, outcome].std(ddof=0)
        z_coefs.append(r["coef"] / sd)
        z_ses.append(r["se"] / sd)
        n_values.append(r["n"])

        zcol = f"__z_{outcome}"
        base[zcol] = (base[outcome] - cm) / sd
        zcols.append(zcol)

    complete = base.dropna(subset=zcols).copy()
    complete["__std_index"] = complete[zcols].mean(axis=1)
    model = smf.wls(f"__std_index ~ treatment + {controls}", data=complete, weights=complete[weight]) if weight else smf.ols(
        f"__std_index ~ treatment + {controls}", data=complete
    )
    res = model.fit(cov_type="cluster", cov_kwds={"groups": complete["household_id"]}, use_t=True)
    return {
        "itt": float(np.mean(z_coefs)),
        "itt_se": float(res.bse["treatment"]),
        "itt_p": float(res.pvalues["treatment"]),
        "component_se_mean": float(np.mean(z_ses)),
        "n": int(min(n_values)),
    }


def table11_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    wave_info = [("0m", None, "initial"), ("6m", "weight_6m", "six_month"), ("12m", "weight_12m", "main")]
    for domain, spec in TABLE11_SPECS.items():
        for wave, weight, sample_label in wave_info:
            outcomes = [pattern.format(w=wave) for pattern in spec["patterns"]]
            r = standardized_effect(df, wave, outcomes, weight)
            paper_itt, paper_se, paper_p = spec["paper"][wave]
            rows.append(dict(table="table11", domain=domain, sample=sample_label, outcome=domain,
                             label=spec["label"], itt=r["itt"], itt_se=r["itt_se"], itt_p=r["itt_p"],
                             component_se_mean=r["component_se_mean"], n=r["n"],
                             paper_itt=paper_itt, paper_se=paper_se, paper_p=paper_p,
                             paper_p_initial_vs_six=spec["paper_diff_p"]["0m_vs_6m"],
                             paper_p_six_vs_main=spec["paper_diff_p"]["6m_vs_12m"],
                             paper_p_initial_vs_main=spec["paper_diff_p"]["0m_vs_12m"]))
    return rows


def main():
    df = load()
    resp = df.loc[df.sample_resp == 1].copy()
    # Women subsets for mammogram (age >= 40) and pap (women).
    # The published Table VI row is matched by using the lottery-list year.
    resp["age2008"] = 2008 - resp["birthyear_12m"]
    women = resp["female_12m"] == 1
    women40 = women & (resp["age2008"] >= 40)

    results = {}

    # ---------- Table 3 first stage ----------
    fs_ins = itt(resp, "ohp_all_ever_survey")
    results["firststage_survey"] = dict(
        outcome="ohp_all_ever_survey", label="Ever on Medicaid (survey subsample)",
        itt=fs_ins["coef"], itt_se=fs_ins["se"], cmean=fs_ins["cmean"], n=fs_ins["n"],
        paper_itt=0.290, paper_cmean=0.135)
    results["table3"] = first_stage_rows(df, resp)

    # ---------- Table 5: utilization ----------
    t5 = []
    t5.append(row(resp, "rx_any_12m",       "Prescription drugs currently",         paper_itt=0.025, paper_late=0.088, paper_cmean=0.637))
    t5.append(row(resp, "doc_any_12m",      "Outpatient visits last 6m",            paper_itt=0.062, paper_late=0.212, paper_cmean=0.574))
    t5.append(row(resp, "er_any_12m",       "ER visits last 6m",                    paper_itt=0.0065, paper_late=0.022, paper_cmean=0.261))
    t5.append(row(resp, "hosp_any_12m",     "Inpatient hosp. admissions last 6m",   paper_itt=0.0022, paper_late=0.0077, paper_cmean=0.072))
    t5.append(row(resp, "rx_num_mod_12m",   "# Prescription drugs",                 paper_itt=0.100, paper_late=0.347, paper_cmean=2.318))
    t5.append(row(resp, "doc_num_mod_12m",  "# Outpatient visits",                  paper_itt=0.314, paper_late=1.083, paper_cmean=1.914))
    t5.append(row(resp, "er_num_mod_12m",   "# ER visits",                          paper_itt=0.0074, paper_late=0.026, paper_cmean=0.47))
    t5.append(row(resp, "hosp_num_mod_12m", "# Inpatient admissions",               paper_itt=0.0062, paper_late=0.021, paper_cmean=0.097))
    results["table5"] = t5

    # ---------- Table 6: preventive care ----------
    t6 = []
    t6.append(row(resp, "chl_chk_12m_ever", "Blood cholesterol checked (ever)",       paper_itt=0.033, paper_late=0.114, paper_cmean=0.625))
    t6.append(row(resp, "dia_chk_12m_ever", "Blood tested for high blood sugar/diab.",paper_itt=0.026, paper_late=0.090, paper_cmean=0.604))
    t6.append(row(resp.loc[women40], "mam_chk_12m_12mo", "Mammogram within last 12m (women>=40)",
                  paper_itt=0.055, paper_late=0.187, paper_cmean=0.298))
    t6.append(row(resp.loc[women], "pap_chk_12m_12mo", "Pap test within last 12m (women)",
                  paper_itt=0.051, paper_late=0.183, paper_cmean=0.406))
    results["table6"] = t6

    # ---------- Table 8: financial strain (survey) ----------
    t8 = []
    t8.append(row(resp, "cost_any_oop_12m",  "Any OOP medical expenses, last 6m",   paper_itt=-0.058, paper_late=-0.200, paper_cmean=0.555))
    t8.append(row(resp, "cost_any_owe_12m",  "Owe money for medical expenses currently", paper_itt=-0.052, paper_late=-0.180, paper_cmean=0.597))
    t8.append(row(resp, "cost_borrow_12m",   "Borrowed/skipped bills for medical debt, last 6m", paper_itt=-0.045, paper_late=-0.154, paper_cmean=0.364))
    t8.append(row(resp, "cost_refused_12m",  "Refused treatment because of medical debt, last 6m", paper_itt=-0.011, paper_late=-0.036, paper_cmean=0.081))
    results["table8"] = t8

    # ---------- Table 9: health ----------
    t9 = []
    t9.append(row(resp, "health_genflip_bin_12m", "Self-reported health good/VG/excellent",
                  paper_itt=0.039, paper_late=0.133, paper_cmean=0.548))
    t9.append(row(resp, "health_notpoor_12m",     "Self-reported health not poor",
                  paper_itt=0.029, paper_late=0.099, paper_cmean=0.86))
    t9.append(row(resp, "health_chgflip_bin_12m", "Health same/better last 6m",
                  paper_itt=0.033, paper_late=0.113, paper_cmean=0.714))
    # Note: in public files, 30-baddays_tot corresponds to the paper's
    # "# days physical health good" and 30-baddays_phys to "did not impair".
    t9.append(row(resp, "notbaddays_tot_12m",     "# days physical health good (past 30)",
                  paper_itt=0.381, paper_late=1.317, paper_cmean=21.862))
    t9.append(row(resp, "notbaddays_phys_12m",    "# days not impaired (past 30)",
                  paper_itt=0.459, paper_late=1.585, paper_cmean=20.329))
    t9.append(row(resp, "notbaddays_ment_12m",    "# days mental health good (past 30)",
                  paper_itt=0.603, paper_late=2.082, paper_cmean=18.738))
    t9.append(row(resp, "nodep_screen_12m",       "Did not screen positive for depression",
                  paper_itt=0.023, paper_late=0.078, paper_cmean=0.671))
    t9.insert(0, mortality_row(df))
    results["table9"] = t9

    # ---------- Table 10: access, quality, happiness ----------
    t10 = []
    t10.append(row(resp, "usual_clinic_12m", "Have usual place of clinic-based care",
                   paper_itt=0.099, paper_late=0.339, paper_cmean=0.499))
    t10.append(row(resp, "usual_doc_12m",   "Have personal doctor",
                   paper_itt=0.081, paper_late=0.280, paper_cmean=0.490))
    t10.append(row(resp, "needmet_med_12m", "Got all needed medical care, last 6m",
                   paper_itt=0.069, paper_late=0.239, paper_cmean=0.684))
    t10.append(row(resp, "needmet_rx_12m",  "Got all needed drugs, last 6m",
                   paper_itt=0.056, paper_late=0.195, paper_cmean=0.765))
    t10.append(row(resp, "notnoner_12m",        "Didn't use ER for nonemergency, last 6m",
                   paper_itt=-0.0011, paper_late=-0.0037, paper_cmean=0.916))
    t10.append(row(resp, "med_qual_bin_12m",    "Quality of care good/VG/excellent (conditional)",
                   paper_itt=0.043, paper_late=0.142, paper_cmean=0.708))
    t10.append(row(resp, "happy_bin_12m",       "Very happy or pretty happy",
                   paper_itt=0.056, paper_late=0.191, paper_cmean=0.594))
    results["table10"] = t10

    # ---------- Table 11: standardized survey ITT effects over time ----------
    results["table11"] = table11_rows(df)

    # ---------- Save ----------
    with open(OUT / "replication_all_tables.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    # Also CSV for convenience
    rows = []
    for tbl in ["table3", "table5", "table6", "table8", "table9", "table10", "table11"]:
        for r in results[tbl]:
            r2 = dict(r)
            r2.setdefault("table", tbl)
            rows.append(r2)
    pd.DataFrame(rows).to_csv(OUT / "replication_all_tables.csv", index=False)
    write_replication_latex_tables(results)

    print("Wrote", OUT / "replication_all_tables.json")
    print("Wrote", OUT / "replication_results_tables.tex")
    return results


if __name__ == "__main__":
    main()
