# app.py — JetLearn: MIS + Predictibility + Trend & Analysis + 80-20 (Merged, de-conflicted)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from calendar import monthrange
import re

# ======================
# Page & minimal styling
# ======================
st.set_page_config(page_title="JetLearn – MIS + Predictibility + Trend + 80-20", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .stAltairChart {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 14px;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(15,23,42,.08);
      }
      .legend-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        margin-right: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #111827;
      }
      .pill-total { background: #e5e7eb; }
      .pill-ai    { background: #bfdbfe; }
      .pill-math  { background: #bbf7d0; }

      .kpi-card {
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px 14px;
        background: #fafafa;
      }
      .kpi-title { color:#6b7280; font-size:.9rem; margin-bottom:6px; }
      .kpi-value { font-weight:700; font-size:1.4rem; color:#111827; }
      .kpi-sub   { color:#6b7280; font-size:.85rem; }
      .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: .25rem;
        margin-bottom: .25rem;
      }
      .chip {
        display:inline-block; padding:4px 8px; border-radius:999px;
        background:#f3f4f6; color:#374151; font-size:.8rem; margin-top:.25rem;
      }
      .muted { color:#6b7280; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = {
    "Total": "#6b7280",
    "AI Coding": "#2563eb",
    "Math": "#16a34a",
    "ThresholdLow": "#f3f4f6",
    "ThresholdMid": "#e5e7eb",
    "ThresholdHigh": "#d1d5db",
    "A_actual": "#2563eb",
    "Rem_prev": "#6b7280",
    "Rem_same": "#16a34a",
}

# ======================
# Helpers (shared)
# ======================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NaT, index=series.index if series is not None else None)
    s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    if s.notna().sum() == 0:
        for unit in ["s", "ms"]:
            try:
                s = pd.to_datetime(series, errors="coerce", unit=unit)
                break
            except Exception:
                pass
    return s

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_of_prev = first_this - timedelta(days=1)
    return month_bounds(last_of_prev)

# Invalid deals exclusion
INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal[s]?\s*$", flags=re.IGNORECASE)
def exclude_invalid_deals(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col:
        return df, 0
    col = df[dealstage_col].astype(str)
    mask_keep = ~col.apply(lambda x: bool(INVALID_RE.match(x)))
    removed = int((~mask_keep).sum())
    return df.loc[mask_keep].copy(), removed

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str):
        return "Other"
    v = value.strip().lower()
    if "math" in v: return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

# Key-source mapping (Referral / PM buckets)
def normalize_key_source(val: str) -> str:
    if not isinstance(val, str): return "Other"
    v = val.strip().lower()
    if "referr" in v: return "Referral"
    if "pm" in v and "search" in v: return "PM - Search"
    if "pm" in v and "social" in v: return "PM - Social"
    return "Other"

def assign_src_pick(df: pd.DataFrame, source_col: str | None, use_key: bool) -> pd.DataFrame:
    d = df.copy()
    if source_col and source_col in d.columns:
        if use_key:
            d["_src_pick"] = d[source_col].apply(normalize_key_source)
        else:
            d["_src_pick"] = d[source_col].fillna("Unknown").astype(str)
    else:
        d["_src_pick"] = "Other"
    return d

# ======================
# Load data & global sidebar
# ======================
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"  # point to /mnt/data/Master_sheet-DB.csv if needed

if "data_src" not in st.session_state:
    st.session_state["data_src"] = DEFAULT_DATA_PATH

with st.sidebar:
    st.header("JetLearn • Navigation")
    view = st.radio(
        "Go to",
        ["Dashboard", "MIS", "Predictibility", "AC Wise Detail", "Trend & Analysis", "80-20", "Stuck deals", "Daily business", "Lead Movement"],  # ← add this
        index=0
    )
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)
    st.caption("Use MIS for status; Predictibility for forecast; Trend & Analysis for grouped drilldowns; 80-20 for Pareto & Mix.")


st.title("📊 JetLearn – Unified App")

# Legend pills (for MIS/Trend visuals)
def active_labels(track: str) -> list[str]:
    if track == "AI Coding":
        return ["Total", "AI Coding"]
    if track == "Math":
        return ["Total", "Math"]
    return ["Total", "AI Coding", "Math"]

legend_labels = active_labels(track)
pill_map = {
    "Total": "<span class='legend-pill pill-total'>Total (Both)</span>",
    "AI Coding": "<span class='legend-pill pill-ai'>AI-Coding</span>",
    "Math": "<span class='legend-pill pill-math'>Math</span>",
}
st.markdown("<div>" + "".join(pill_map[l] for l in legend_labels) + "</div>", unsafe_allow_html=True)

# Data load
data_src = st.session_state["data_src"]
with st.expander("Data & Filters (Global for MIS / Predictibility / Trend & Analysis)", expanded=False):
    def _update_data_src():
        st.session_state["data_src"] = st.session_state.get("data_src_input", DEFAULT_DATA_PATH)
        st.rerun()

    st.text_input(
        "Data file path",
        key="data_src_input",
        value=st.session_state.get("data_src", DEFAULT_DATA_PATH),
        help="CSV path (e.g., /mnt/data/Master_sheet-DB.csv).",
        on_change=_update_data_src,
    )

df = load_csv(data_src)

# Column mapping
dealstage_col = find_col(df, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
df, _removed = exclude_invalid_deals(df, dealstage_col)
if dealstage_col:
    st.caption(f"Excluded “1.2 Invalid deal(s)”: *{_removed:,}* rows (column: *{dealstage_col}*).")
else:
    st.info("Deal Stage column not found — cannot auto-exclude “1.2 Invalid deal(s)”. Check your file.")

create_col = find_col(df, ["Create Date","Create date","Create_Date","Created At"])
pay_col    = find_col(df, ["Payment Received Date","Payment Received date","Payment_Received_Date","Payment Date","Paid At"])
pipeline_col = find_col(df, ["Pipeline"])
counsellor_col = find_col(df, ["Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor"])
country_col    = find_col(df, ["Country"])
source_col     = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
first_cal_sched_col = find_col(df, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
cal_resched_col     = find_col(df, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
cal_done_col        = find_col(df, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])
calibration_slot_col = find_col(df, ["Calibration Slot (Deal)", "Calibration Slot", "Cal Slot (Deal)", "Cal Slot"])


if not create_col or not pay_col:
    st.error("Could not find required date columns. Need 'Create Date' and 'Payment Received Date' (or close variants).")
    st.stop()

# Clean invalid Create Date
tmp_create_all = coerce_datetime(df[create_col])
missing_create = int(tmp_create_all.isna().sum())
if missing_create > 0:
    df = df.loc[tmp_create_all.notna()].copy()
    st.caption(f"Removed rows with missing/invalid *Create Date: **{missing_create:,}*")

# Presets
today = date.today()
yday = today - timedelta(days=1)
last_m_start, last_m_end = last_month_bounds(today)
this_m_start, this_m_end = month_bounds(today)
this_m_end_mtd = today

# Global filters for MIS/Pred/Trend
def prep_options(series: pd.Series):
    vals = sorted([str(v) for v in series.dropna().unique()])
    return ["All"] + vals

with st.expander("Filters (apply to MIS / Predictibility / Trend & Analysis)", expanded=False):
    
    if counsellor_col:
        sel_counsellors = st.multiselect("Academic Counsellor", options=prep_options(df[counsellor_col]), default=["All"])
    else:
        sel_counsellors = []
        st.info("Academic Counsellor column not found.")

    if country_col:
        sel_countries = st.multiselect("Country", options=prep_options(df[country_col]), default=["All"])
    else:
        sel_countries = []
        st.info("Country column not found.")

    if source_col:
        sel_sources = st.multiselect("JetLearn Deal Source", options=prep_options(df[source_col]), default=["All"])
    else:
        sel_sources = []
        st.info("JetLearn Deal Source column not found.")

def apply_filters(
    df: pd.DataFrame,
    counsellor_col: str | None,
    country_col: str | None,
    source_col: str | None,
    sel_counsellors: list[str],
    sel_countries: list[str],
    sel_sources: list[str],
) -> pd.DataFrame:
    f = df.copy()
    if counsellor_col and sel_counsellors and "All" not in sel_counsellors:
        f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and sel_countries and "All" not in sel_countries:
        f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and sel_sources and "All" not in sel_sources:
        f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

df_f = apply_filters(df, counsellor_col, country_col, source_col, sel_counsellors, sel_countries, sel_sources)

if track != "Both":
    if pipeline_col and pipeline_col in df_f.columns:
        _norm = df_f[pipeline_col].map(normalize_pipeline).fillna("Other")
        df_f = df_f.loc[_norm == track].copy()
    else:
        st.warning("Pipeline column not found — the Track filter can’t be applied.", icon="⚠")

st.caption(f"Rows in scope after filters: *{len(df_f):,}*")
st.caption(f"Track filter: *{track}*")

# ======================
# Shared functions for MIS / Trend / Predictibility
# ======================
def prepare_counts_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    month_for_mtd: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None
):
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col])
    d["_pay_dt"] = coerce_datetime(d[pay_col])

    in_range_pay = d["_pay_dt"].dt.date.between(start_d, end_d)
    m_start, m_end = month_bounds(month_for_mtd)
    in_month_create = d["_create_dt"].dt.date.between(m_start, m_end)

    cohort_df = d.loc[in_range_pay]
    mtd_df = d.loc[in_range_pay & in_month_create]

    if pipeline_col and pipeline_col in d.columns:
        cohort_split = cohort_df[pipeline_col].map(normalize_pipeline).fillna("Other")
        mtd_split = mtd_df[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        cohort_split = pd.Series([], dtype=object)
        mtd_split = pd.Series([], dtype=object)

    cohort_counts = {
        "Total": int(len(cohort_df)),
        "AI Coding": int((pd.Series(cohort_split) == "AI Coding").sum()),
        "Math": int((pd.Series(cohort_split) == "Math").sum()),
    }
    mtd_counts = {
        "Total": int(len(mtd_df)),
        "AI Coding": int((pd.Series(mtd_split) == "AI Coding").sum()),
        "Math": int((pd.Series(mtd_split) == "Math").sum()),
    }
    return mtd_counts, cohort_counts

def deals_created_mask_range(df: pd.DataFrame, denom_start: date, denom_end: date, create_col: str) -> pd.Series:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    return d["_create_dt"].between(denom_start, denom_end)

def prepare_conversion_for_range(
    df: pd.DataFrame,
    start_d: date,
    end_d: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    *,
    denom_start: date,
    denom_end: date
):
    d = df.copy()
    d["_create_dt"] = coerce_datetime(d[create_col]).dt.date
    d["_pay_dt"] = coerce_datetime(d[pay_col]).dt.date

    denom_mask = deals_created_mask_range(d, denom_start, denom_end, create_col)

    if pipeline_col and pipeline_col in d.columns:
        pl = d[pipeline_col].map(normalize_pipeline).fillna("Other")
    else:
        pl = pd.Series(["Other"] * len(d), index=d.index)

    den_total = int(denom_mask.sum()); den_ai = int((denom_mask & (pl == "AI Coding")).sum()); den_math = int((denom_mask & (pl == "Math")).sum())
    denoms = {"Total": den_total, "AI Coding": den_ai, "Math": den_math}

    pay_mask = d["_pay_dt"].between(start_d, end_d)

    mtd_mask = pay_mask & denom_mask
    mtd_total = int(mtd_mask.sum()); mtd_ai = int((mtd_mask & (pl == "AI Coding")).sum()); mtd_math = int((mtd_mask & (pl == "Math")).sum())

    coh_mask = pay_mask
    coh_total = int(coh_mask.sum()); coh_ai = int((coh_mask & (pl == "AI Coding")).sum()); coh_math = int((coh_mask & (pl == "Math")).sum())

    def pct(n, d):
        if d == 0: return 0.0
        return max(0.0, min(100.0, round(100.0 * n / d, 1)))

    mtd_pct = {"Total": pct(mtd_total, den_total), "AI Coding": pct(mtd_ai, den_ai), "Math": pct(mtd_math, den_math)}
    coh_pct = {"Total": pct(coh_total, den_total), "AI Coding": pct(coh_ai, den_ai), "Math": pct(coh_math, den_math)}
    numerators = {"mtd": {"Total": mtd_total, "AI Coding": mtd_ai, "Math": mtd_math}, "cohort": {"Total": coh_total, "AI Coding": coh_ai, "Math": coh_math}}
    return mtd_pct, coh_pct, denoms, numerators

def bubble_chart_counts(title: str, total: int, ai_cnt: int, math_cnt: int, labels: list[str] = None):
    all_rows = [
        {"Label": "Total",     "Value": total,   "Row": 0, "Col": 0.5},
        {"Label": "AI Coding", "Value": ai_cnt,  "Row": 1, "Col": 0.33},
        {"Label": "Math",      "Value": math_cnt,"Row": 1, "Col": 0.66},
    ]
    if labels is None:
        labels = ["Total", "AI Coding", "Math"]
    data = pd.DataFrame([r for r in all_rows if r["Label"] in labels])

    color_domain = labels
    color_range_map = {"Total": PALETTE["Total"], "AI Coding": PALETTE["AI Coding"], "Math": PALETTE["Math"]}
    color_range = [color_range_map[l] for l in labels]

    base = alt.Chart(data).encode(
        x=alt.X("Col:Q", axis=None, scale=alt.Scale(domain=(0, 1))),
        y=alt.Y("Row:Q", axis=None, scale=alt.Scale(domain=(-0.2, 1.2))),
        tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("Value:Q")],
    )
    circles = base.mark_circle(opacity=0.85).encode(
        size=alt.Size("Value:Q", scale=alt.Scale(range=[400, 8000]), legend=None),
        color=alt.Color("Label:N", scale=alt.Scale(domain=color_domain, range=color_range), legend=None),
    )
    text = base.mark_text(fontWeight="bold", dy=0, color="#111827").encode(text=alt.Text("Value:Q"))
    return (circles + text).properties(height=360, title=title)

def conversion_kpis_only(title: str, pcts: dict, nums: dict, denoms: dict, labels: list[str]):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    order = [l for l in ["Total", "AI Coding", "Math"] if l in labels]
    cols = st.columns(len(order))
    for i, label in enumerate(order):
        color = {"Total":"#111827","AI Coding":PALETTE["AI Coding"],"Math":PALETTE["Math"]}[label]
        with cols[i]:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-title'>{label}</div>"
                f"<div class='kpi-value' style='color:{color}'>{pcts[label]:.1f}%</div>"
                f"<div class='kpi-sub'>Den: {denoms.get(label,0):,} • Num: {nums.get(label,0):,}</div></div>",
                unsafe_allow_html=True,
            )

def trend_timeseries(
    df: pd.DataFrame,
    payments_start: date,
    payments_end: date,
    *,
    denom_start: date,
    denom_end: date,
    create_col: str = "",
    pay_col: str = ""
):
    df = df.copy()
    df["_create_dt"] = coerce_datetime(df[create_col]).dt.date
    df["_pay_dt"] = coerce_datetime(df[pay_col]).dt.date

    base_start = min(payments_start, denom_start)
    base_end = max(payments_end, denom_end)
    denom_mask = df["_create_dt"].between(denom_start, denom_end)

    all_days = pd.date_range(base_start, base_end, freq="D").date

    leads = (
        df.loc[denom_mask]
          .groupby("_create_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("Leads")
    )
    pay_mask = df["_pay_dt"].between(payments_start, payments_end)
    cohort = (
        df.loc[pay_mask]
          .groupby("_pay_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("Cohort")
    )
    mtd = (
        df.loc[pay_mask & denom_mask]
          .groupby("_pay_dt")
          .size()
          .reindex(all_days, fill_value=0)
          .rename("MTD")
    )

    ts = pd.concat([leads, mtd, cohort], axis=1).fillna(0).reset_index()
    ts = ts.rename(columns={"index": "Date"})
    return ts

def trend_chart(ts: pd.DataFrame, title: str):
    base = alt.Chart(ts).encode(x=alt.X("Date:T", axis=alt.Axis(title=None)))
    bars = base.mark_bar(opacity=0.75).encode(
        y=alt.Y("Leads:Q", axis=alt.Axis(title="Leads (deals created)")),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Leads:Q")]
    ).properties(height=260)
    line_mtd = base.mark_line(point=True).encode(
        y=alt.Y("MTD:Q", axis=alt.Axis(title="Enrolments"), scale=alt.Scale(zero=True)),
        color=alt.value(PALETTE["AI Coding"]),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("MTD:Q", title="MTD Enrolments")]
    )
    line_coh = base.mark_line(point=True).encode(
        y=alt.Y("Cohort:Q", axis=alt.Axis(title="Enrolments"), scale=alt.Scale(zero=True)),
        color=alt.value(PALETTE["Math"]),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Cohort:Q", title="Cohort Enrolments")]
    )
    return alt.layer(bars, line_mtd, line_coh).resolve_scale(y='independent').properties(title=title)

# ======================
# MIS rendering
# ======================
def render_period_block(
    df_scope: pd.DataFrame,
    title: str,
    range_start: date,
    range_end: date,
    running_month_anchor: date,
    create_col: str,
    pay_col: str,
    pipeline_col: str | None,
    track: str
):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    labels = active_labels(track)

    # Counts
    mtd_counts, coh_counts = prepare_counts_for_range(
        df_scope, range_start, range_end, running_month_anchor, create_col, pay_col, pipeline_col
    )

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)",
                                            mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"],
                                            labels=labels), use_container_width=True)
    with c2:
        st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)",
                                            coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"],
                                            labels=labels), use_container_width=True)

    # Conversion% (denominator = create dates within selected window) — KPI only
    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
        df_scope, range_start, range_end, create_col, pay_col, pipeline_col,
        denom_start=range_start, denom_end=range_end
    )
    st.caption("Denominators (selected window create dates) — " +
               " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in labels]))

    conversion_kpis_only("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=labels)
    conversion_kpis_only("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=labels)

    # Trend uses SAME population rule
    ts = trend_timeseries(df_scope, range_start, range_end,
                          denom_start=range_start, denom_end=range_end,
                          create_col=create_col, pay_col=pay_col)
    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)

# ======================
# Predictibility helpers
# ======================
def add_month_cols(df: pd.DataFrame, create_col: str, pay_col: str) -> pd.DataFrame:
    d = df.copy()
    d["_create_dt"] = coerce_datetime(df[create_col])
    d["_pay_dt"]    = coerce_datetime(df[pay_col])
    d["_create_m"]  = d["_create_dt"].dt.to_period("M")
    d["_pay_m"]     = d["_pay_dt"].dt.to_period("M")
    d["_same_month"] = (d["_create_m"] == d["_pay_m"])
    return d

def per_source_monthly_counts(d_hist: pd.DataFrame, source_col: str):
    if d_hist.empty:
        return pd.DataFrame(columns=["_pay_m", source_col, "cnt_same", "cnt_prev", "days_in_month"])
    g = d_hist.groupby(["_pay_m", source_col])
    by = g["_same_month"].agg(
        cnt_same=lambda s: int(s.sum()),
        cnt_prev=lambda s: int((~s).sum())
    ).reset_index()
    by["days_in_month"] = by["_pay_m"].apply(lambda p: monthrange(p.year, p.month)[1])
    return by

def daily_rates_from_lookback(d_hist: pd.DataFrame, source_col: str, lookback: int, weighted: bool):
    if d_hist.empty:
        return {}, {}, 0.0, 0.0

    months = sorted(d_hist["_pay_m"].unique())
    months = months[-lookback:] if len(months) > lookback else months
    d_hist = d_hist[d_hist["_pay_m"].isin(months)].copy()

    by = per_source_monthly_counts(d_hist, source_col)
    month_to_w = {m: (i+1 if weighted else 1.0) for i, m in enumerate(sorted(months))}

    rates_same, rates_prev = {}, {}
    for src, sub in by.groupby(source_col):
        w = sub["_pay_m"].map(month_to_w)
        num_same = (sub["cnt_same"] / sub["days_in_month"] * w).sum()
        num_prev = (sub["cnt_prev"] / sub["days_in_month"] * w).sum()
        den = w.sum()
        rates_same[str(src)] = float(num_same/den) if den > 0 else 0.0
        rates_prev[str(src)] = float(num_prev/den) if den > 0 else 0.0

    by_overall = d_hist.groupby("_pay_m")["_same_month"].agg(
        cnt_same=lambda s: int(s.sum()),
        cnt_prev=lambda s: int((~s).sum())
    ).reset_index()
    by_overall["days_in_month"] = by_overall["_pay_m"].apply(lambda p: monthrange(p.year, p.month)[1])
    w_all = by_overall["_pay_m"].map(month_to_w)
    num_same_o = (by_overall["cnt_same"] / by_overall["days_in_month"] * w_all).sum()
    num_prev_o = (by_overall["cnt_prev"] / by_overall["days_in_month"] * w_all).sum()
    den_o = w_all.sum()
    overall_same_rate = float(num_same_o/den_o) if den_o > 0 else 0.0
    overall_prev_rate = float(num_prev_o/den_o) if den_o > 0 else 0.0
    return rates_same, rates_prev, overall_same_rate, overall_prev_rate

def predict_running_month(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                          lookback: int, weighted: bool, today: date):
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()
        # include blank/NaN deal sources as "Unknown" so they are counted
        df_work[source_col] = df_work[source_col].fillna("Unknown").astype(str)

    d = add_month_cols(df_work, create_col, pay_col)

    cur_start, cur_end = month_bounds(today)
    cur_period = pd.Period(today, freq="M")

    d_cur = d[d["_pay_m"] == cur_period].copy()
    if d_cur.empty:
        realized_by_src = pd.DataFrame(columns=[source_col, "A"])
    else:
        # include Unknown deal source in Actual-to-date
        realized_by_src = (
            d_cur.assign({source_col: d_cur[source_col].fillna("Unknown").astype(str)})
                .groupby(source_col).size().rename("A").reset_index()
        )

    d_hist = d[d["_pay_m"] < cur_period].copy()
    rates_same, rates_prev, overall_same_rate, overall_prev_rate = daily_rates_from_lookback(
        d_hist, source_col, lookback, weighted
    )

    elapsed_days = (today - cur_start).days + 1
    total_days   = (cur_end - cur_start).days + 1
    remaining_days = max(0, total_days - elapsed_days)

    src_realized = set(d_cur[source_col].fillna("Unknown").astype(str)) if not d_cur.empty else set()
    src_hist = set(list(rates_same.keys()) + list(rates_prev.keys()))
    all_sources = sorted(src_realized | src_hist | ({"All"} if source_col == "_Source" else set()))

    A_tot = B_tot = C_tot = 0.0
    rows = []
    a_map = dict(zip(realized_by_src[source_col], realized_by_src["A"])) if not realized_by_src.empty else {}

    for src in all_sources:
        a_val = float(a_map.get(src, 0.0))
        rate_same = rates_same.get(src, overall_same_rate)
        rate_prev = rates_prev.get(src, overall_prev_rate)

        b_val = float(rate_same * remaining_days)
        c_val = float(rate_prev * remaining_days)

        rows.append({
            "Source": src,
            "A_Actual_ToDate": a_val,
            "B_Remaining_SameMonth": b_val,
            "C_Remaining_PrevMonths": c_val,
            "Projected_MonthEnd_Total": a_val + b_val + c_val,
            "Rate_Same_Daily": rate_same,
            "Rate_Prev_Daily": rate_prev,
            "Remaining_Days": remaining_days
        })
        A_tot += a_val
        B_tot += b_val
        C_tot += c_val

    tbl = pd.DataFrame(rows).sort_values("Source").reset_index(drop=True)
    totals = {
        "A_Actual_ToDate": A_tot,
        "B_Remaining_SameMonth": B_tot,
        "C_Remaining_PrevMonths": C_tot,
        "Projected_MonthEnd_Total": A_tot + B_tot + C_tot,
        "Remaining_Days": remaining_days
    }
    return tbl, totals



def predict_chart_stacked(tbl: pd.DataFrame):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    melt = tbl.melt(
        id_vars=["Source"],
        value_vars=["A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths"],
        var_name="Component",
        value_name="Value"
    )
    color_map = {"A_Actual_ToDate": PALETTE["A_actual"], "B_Remaining_SameMonth": PALETTE["Rem_same"], "C_Remaining_PrevMonths": PALETTE["Rem_prev"]}
    chart = alt.Chart(melt).mark_bar().encode(
        x=alt.X("Source:N", sort=alt.SortField("Source")),
        y=alt.Y("Value:Q", stack=True),
        color=alt.Color("Component:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="Component", orient="top", labelLimit=240)),
        tooltip=[alt.Tooltip("Source:N"), alt.Tooltip("Component:N"), alt.Tooltip("Value:Q", format=",.1f")]
    ).properties(height=360, title="Predictibility (A + B + C = Projected Month-End)")
    return chart

def month_list_before(period_end: pd.Period, k: int):
    months = []
    p = period_end
    for _ in range(k):
        p = (p - 1)
        months.append(p)
    months.reverse()
    return months

def backtest_accuracy(df_f: pd.DataFrame, create_col: str, pay_col: str, source_col: str,
                      lookback: int, weighted: bool, backtest_months: int, today: date):
    if source_col is None or source_col not in df_f.columns:
        df_work = df_f.copy()
        source_col = "_Source"
        df_work[source_col] = "All"
    else:
        df_work = df_f.copy()

    d = add_month_cols(df_work, create_col, pay_col)
    current_period = pd.Period(today, freq="M")

    months_to_eval = month_list_before(current_period, backtest_months)
    rows = []
    for m in months_to_eval:
        train_months = month_list_before(m, lookback)
        d_train = d[d["_pay_m"].isin(train_months)]
        if d_train.empty:
            same_rates, prev_rates, same_rate_o, prev_rate_o = {}, {}, 0.0, 0.0
        else:
            same_rates, prev_rates, same_rate_o, prev_rate_o = daily_rates_from_lookback(
                d_train, source_col, lookback=len(train_months), weighted=weighted
            )

        d_m = d[d["_pay_m"] == m]
        actual_total = int(len(d_m))
        days_in_m = monthrange(m.year, m.month)[1]

        sources = set(list(same_rates.keys()) + list(prev_rates.keys()))
        if not sources and source_col != "_Source":
            sources = set(d_m[source_col].dropna().astype(str).unique().tolist())
        if not sources:
            sources = {"All"}

        forecast = 0.0
        for src in sources:
            r_same = same_rates.get(src, same_rate_o)
            r_prev = prev_rates.get(src, prev_rate_o)
            forecast += (r_same + r_prev) * days_in_m

        err = forecast - actual_total
        rows.append({
            "Month": str(m), "Days": days_in_m,
            "Forecast": float(forecast), "Actual": float(actual_total),
            "Error": float(err), "AbsError": float(abs(err)),
            "SqError": float(err**2),
            "APE": float(abs(err) / actual_total) if actual_total > 0 else np.nan
        })

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {"MAPE": np.nan, "WAPE": np.nan, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = bt["AbsError"].mean()
    rmse = (bt["SqError"].mean())**0.5
    wape = (bt["AbsError"].sum() / bt["Actual"].sum()) if bt["Actual"].sum() > 0 else np.nan
    mape = bt["APE"].dropna().mean() if bt["APE"].notna().any() else np.nan
    ss_res = ((bt["Actual"] - bt["Forecast"])**2).sum()
    ss_tot = ((bt["Actual"] - bt["Actual"].mean())**2).sum()
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return bt, {"MAPE": mape, "WAPE": wape, "MAE": mae, "RMSE": rmse, "R2": r2}

def accuracy_scatter(bt: pd.DataFrame):
    if bt.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    chart = alt.Chart(bt).mark_circle(size=120, opacity=0.8).encode(
        x=alt.X("Actual:Q", title="Actual (month total)"),
        y=alt.Y("Forecast:Q", title="Forecast (start-of-month)"),
        tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Actual:Q"), alt.Tooltip("Forecast:Q"), alt.Tooltip("Error:Q")],
    ).properties(height=360, title="Forecast vs Actual (by month)")
    line = alt.Chart(pd.DataFrame({"x":[bt["Actual"].min(), bt["Actual"].max()],
                                   "y":[bt["Actual"].min(), bt["Actual"].max()]})).mark_line()
    return chart + line

# ======================
# 80-20 (Pareto + Trajectory + Mix) helpers
# ======================
def build_pareto(df: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    if group_col is None or group_col not in df.columns:
        return pd.DataFrame(columns=[label, "Count", "CumCount", "CumPct", "Tag"])
    tmp = (
        df.assign(_grp=df[group_col].fillna("Unknown").astype(str))
          .groupby("_grp").size().sort_values(ascending=False).rename("Count").reset_index()
          .rename(columns={"_grp": label})
    )
    if tmp.empty:
        return tmp
    tmp["CumCount"] = tmp["Count"].cumsum()
    total = tmp["Count"].sum()
    tmp["CumPct"] = (tmp["CumCount"] / total) * 100.0
    tmp["Tag"] = np.where(tmp["CumPct"] <= 80.0, "Top 80%", "Bottom 20%")
    return tmp

def pareto_chart(tbl: pd.DataFrame, label: str, title: str):
    if tbl.empty:
        return alt.Chart(pd.DataFrame({"x":[],"y":[]}))
    base = alt.Chart(tbl).encode(x=alt.X(f"{label}:N", sort=list(tbl[label])))
    bars = base.mark_bar(opacity=0.85).encode(
        y=alt.Y("Count:Q", axis=alt.Axis(title="Enrollments (count)")),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("Count:Q")]
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("CumPct:Q", axis=alt.Axis(title="Cumulative %", orient="right")),
        color=alt.value("#16a34a"),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("CumPct:Q", format=".1f")]
    )
    rule80 = alt.Chart(pd.DataFrame({"y":[80.0]})).mark_rule(strokeDash=[4,4]).encode(y="y:Q")
    return alt.layer(bars, line, rule80).resolve_scale(y='independent').properties(title=title, height=360)

def months_back_list(end_d: date, k: int):
    p_end = pd.Period(end_d, freq="M")
    return [p_end - i for i in range(k-1, -1, -1)]

# ======================
# RENDER: Views
# ======================
if view == "MIS":
    show_all = st.checkbox("Show all preset periods (Yesterday • Today • Last Month • This Month)", value=False)
    if show_all:
        st.subheader("Preset Periods")
        colA, colB = st.columns(2)
        with colA:
            render_period_block(df_f, "Yesterday", yday, yday, yday, create_col, pay_col, pipeline_col, track)
            st.divider()
            render_period_block(df_f, "Last Month", last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col, track)
        with colB:
            render_period_block(df_f, "Today", today, today, today, create_col, pay_col, pipeline_col, track)
            st.divider()
            render_period_block(df_f, "This Month (MTD)", this_m_start, this_m_end_mtd, this_m_start, create_col, pay_col, pipeline_col, track)
    else:
        tabs = st.tabs(["Yesterday", "Today", "Last Month", "This Month (MTD)", "Custom"])
        with tabs[0]:
            render_period_block(df_f, "Yesterday", yday, yday, yday, create_col, pay_col, pipeline_col, track)
        with tabs[1]:
            render_period_block(df_f, "Today", today, today, today, create_col, pay_col, pipeline_col, track)
        with tabs[2]:
            render_period_block(df_f, "Last Month", last_m_start, last_m_end, last_m_start, create_col, pay_col, pipeline_col, track)
        with tabs[3]:
            render_period_block(df_f, "This Month (MTD)", this_m_start, this_m_end_mtd, this_m_start, create_col, pay_col, pipeline_col, track)
        with tabs[4]:
            st.markdown("Select a *payments period* and choose the *Conversion% denominator* mode.")
            colc1, colc2 = st.columns(2)
            with colc1: custom_start = st.date_input("Payments period start", value=this_m_start, key="mis_cust_pay_start")
            with colc2: custom_end   = st.date_input("Payments period end (inclusive)", value=this_m_end, key="mis_cust_pay_end")
            if custom_end < custom_start:
                st.error("Payments period end cannot be before start.")
            else:
                denom_mode = st.radio("Denominator for Conversion%", ["Anchor month", "Custom range"], index=0, horizontal=True, key="mis_dmode")
                if denom_mode == "Anchor month":
                    anchor = st.date_input("Running-month anchor (denominator month)", value=custom_start, key="mis_anchor")
                    mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor, create_col, pay_col, pipeline_col)
                    c1, c2 = st.columns(2)
                    with c1: st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], labels=active_labels(track)), use_container_width=True)
                    with c2: st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], labels=active_labels(track)), use_container_width=True)

                    mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
                        df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                        denom_start=anchor.replace(day=1),
                        denom_end=month_bounds(anchor)[1]
                    )
                    st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in active_labels(track)]))
                    conversion_kpis_only("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=active_labels(track))
                    conversion_kpis_only("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=active_labels(track))

                    ts = trend_timeseries(df_f, custom_start, custom_end,
                                          denom_start=anchor.replace(day=1), denom_end=month_bounds(anchor)[1],
                                          create_col=create_col, pay_col=pay_col)
                    st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)
                else:
                    cold1, cold2 = st.columns(2)
                    with cold1: denom_start = st.date_input("Denominator start (deals created from)", value=custom_start, key="mis_den_start")
                    with cold2: denom_end   = st.date_input("Denominator end (deals created to)",   value=custom_end,   key="mis_den_end")
                    if denom_end < denom_start:
                        st.error("Denominator end cannot be before start.")
                    else:
                        anchor_for_counts = custom_start
                        mtd_counts, coh_counts = prepare_counts_for_range(df_f, custom_start, custom_end, anchor_for_counts, create_col, pay_col, pipeline_col)
                        c1, c2 = st.columns(2)
                        with c1: st.altair_chart(bubble_chart_counts("MTD Enrolments (counts)", mtd_counts["Total"], mtd_counts["AI Coding"], mtd_counts["Math"], labels=active_labels(track)), use_container_width=True)
                        with c2: st.altair_chart(bubble_chart_counts("Cohort Enrolments (counts)", coh_counts["Total"], coh_counts["AI Coding"], coh_counts["Math"], labels=active_labels(track)), use_container_width=True)

                        mtd_pct, coh_pct, denoms, nums = prepare_conversion_for_range(
                            df_f, custom_start, custom_end, create_col, pay_col, pipeline_col,
                            denom_start=denom_start, denom_end=denom_end
                        )
                        st.caption("Denominators — " + " • ".join([f"{lbl}: {denoms.get(lbl,0):,}" for lbl in active_labels(track)]))
                        conversion_kpis_only("MTD Conversion %", mtd_pct, nums["mtd"], denoms, labels=active_labels(track))
                        conversion_kpis_only("Cohort Conversion %", coh_pct, nums["cohort"], denoms, labels=active_labels(track))

                        ts = trend_timeseries(df_f, custom_start, custom_end,
                                              denom_start=denom_start, denom_end=denom_end,
                                              create_col=create_col, pay_col=pay_col)
                        st.altair_chart(trend_chart(ts, "Trend: Leads (bars) vs Enrolments (lines)"), use_container_width=True)


elif view == "Predictibility":
    st.subheader("Predictibility – Running Month Enrolment Forecast")
    st.caption("A = payments to date; B = remaining (same-month created rate); C = remaining (prev-months created rate).")
    colp1, colp2, colp3 = st.columns([1,1,2])
    with colp1:
        lookback = st.selectbox("Lookback window (months)", [3, 6, 12], index=0)
    with colp2:
        st.markdown("**Averaging:** Recency-weighted"); weighted = True
    with colp3:
        st.info("Rates computed per source over the last K pay-months (excluding current).")

    cur_start, cur_end = month_bounds(today)
    d_preview = add_month_cols(df_f, create_col, pay_col)
    cur_period = pd.Period(today, freq="M")
    in_cur_pay = d_preview["_pay_m"] == cur_period
    st.caption(f"Payments found this month (after filters): **{int(in_cur_pay.sum()):,}**")

    tbl, totals = predict_running_month(df_f, create_col, pay_col, source_col, lookback, weighted, today)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>A · Actual to date</div><div class='kpi-value' style='color:{PALETTE['A_actual']}'>{totals['A_Actual_ToDate']:.1f}</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>B · Remaining (same-month)</div><div class='kpi-value' style='color:{PALETTE['Rem_same']}'>{totals['B_Remaining_SameMonth']:.1f}</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>C · Remaining (prev-months)</div><div class='kpi-value' style='color:{PALETTE['Rem_prev']}'>{totals['C_Remaining_PrevMonths']:.1f}</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Projected Month-End</div><div class='kpi-value' style='color:{PALETTE['Total']}'>{totals['Projected_MonthEnd_Total']:.1f}</div><div class='kpi-sub'>A + B + C</div></div>", unsafe_allow_html=True)

    st.altair_chart(predict_chart_stacked(tbl), use_container_width=True)

    with st.expander("Detailed table (by source)"):
        show_cols = ["Source","A_Actual_ToDate","B_Remaining_SameMonth","C_Remaining_PrevMonths","Projected_MonthEnd_Total","Rate_Same_Daily","Rate_Prev_Daily","Remaining_Days"]
        if not tbl.empty:
            view_tbl = tbl[show_cols].copy()
            for c in ["B_Remaining_SameMonth","C_Remaining_PrevMonths","Projected_MonthEnd_Total","Rate_Same_Daily","Rate_Prev_Daily"]:
                view_tbl[c] = view_tbl[c].astype(float).round(3)
            st.dataframe(view_tbl, use_container_width=True)
            csv = view_tbl.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv, file_name="predictibility_by_source.csv", mime="text/csv")
        else:
            st.info("No data in scope for the running month after filters.")

    st.subheader("Model Accuracy")
    bt, metrics = backtest_accuracy(df_f, create_col, pay_col, source_col, lookback=lookback, weighted=True, backtest_months=lookback, today=date.today())
    acc_pct = np.nan
    if not pd.isna(metrics.get("WAPE", np.nan)):
        acc_pct = max(0.0, min(100.0, (1.0 - metrics["WAPE"]) * 100.0))
    elif not pd.isna(metrics.get("MAPE", np.nan)):
        acc_pct = max(0.0, min(100.0, (1.0 - metrics["MAPE"]) * 100.0))
    st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Model Accuracy (100 − WAPE)</div><div class='kpi-value'>{'–' if pd.isna(acc_pct) else f'{acc_pct:.1f}%'}</div></div>", unsafe_allow_html=True)

    show_details = st.checkbox("Show detailed metrics", value=False)
    if show_details:
        m1, m2, m3, m4, m5 = st.columns(5)
        def fmt(x, pct=False): return "–" if pd.isna(x) else (f"{x*100:.1f}%" if pct else f"{x:.2f}")
        with m1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>MAPE</div><div class='kpi-value'>{fmt(metrics['MAPE'], pct=True)}</div></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>WAPE</div><div class='kpi-value'>{fmt(metrics['WAPE'], pct=True)}</div></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>MAE</div><div class='kpi-value'>{fmt(metrics['MAE'])}</div></div>", unsafe_allow_html=True)
        with m4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>RMSE</div><div class='kpi-value'>{fmt(metrics['RMSE'])}</div></div>", unsafe_allow_html=True)
        with m5: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>R²</div><div class='kpi-value'>{fmt(metrics['R2'])}</div></div>", unsafe_allow_html=True)
        if bt.empty:
            st.info("Not enough historical data to backtest with the chosen settings.")
        else:
            st.altair_chart(accuracy_scatter(bt), use_container_width=True)

    # ======================
    # Inactivity snapshot (relative to today)
    # ======================
    st.markdown("### Inactivity snapshot (relative to today)")

    # Resolve columns flexibly
    last_activity_col = find_col(df, ["LastActivityDate", "Last Activity Date", "LastActivityTest", "Last Activity"])
    last_connected_col = find_col(df, ["LastConnectedDate", "Last Connected Date", "LastContacted", "Last Contacted"])

    # Toggle which measure to use
    pick = st.radio(
        "Measure",
        ["Last Activity Date", "Last Connected"],
        horizontal=True,
        index=0,
        help="Counts deals where the chosen 'last touch' happened ≥ N days ago, relative to today."
    )

    # Pick column based on toggle
    if pick == "Last Activity Date":
        col_pick = last_activity_col
        missing_msg = "Last Activity Date column not found."
    else:
        col_pick = last_connected_col
        missing_msg = "Last Connected column not found."

    if not col_pick or col_pick not in df_f.columns:
        st.warning(missing_msg + " This snapshot cannot be computed for the selected option.", icon="⚠️")
    else:
        # Slider (seek bar): days since >= N
        thr = st.slider("Days since last touch ≥", min_value=0, max_value=180, value=7, step=1)

        # Normalized datetime and age in days (relative to today)
        s = coerce_datetime(df_f[col_pick])
        today_ts = pd.Timestamp(date.today())
        # convert to age (days); future dates become negative ⇒ clip to 0 so they don't get counted accidentally
        age_days = (today_ts - s).dt.days
        age_days = age_days.where(s.notna(), np.nan).clip(lower=0)

        # Include unknowns?
        include_unknown = st.checkbox(
            "Include deals with unknown/missing dates",
            value=False,
            help="When ON, deals with no date recorded are included in the count."
        )

        # Build mask: (known & age >= thr) OR (include unknown & missing)
        mask_known_old = s.notna() & (age_days >= thr)
        mask = mask_known_old | (include_unknown & s.isna())

        # Count + preview
        count_val = int(mask.sum())

        # KPI card
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-title'>Deals with {pick.lower()} ≥ {thr} day(s) ago</div>"
            f"<div class='kpi-value'>{count_val:,}</div>"
            f"<div class='kpi-sub'>Reference date: {date.today().isoformat()} • "
            f"{'Unknowns included' if include_unknown else 'Unknowns excluded'}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Optional small preview (top 200)
        with st.expander("Preview matching deals (top 200)"):
            show_cols = []
            if create_col: show_cols.append(create_col)
            show_cols.append(col_pick)
            if counsellor_col: show_cols.append(counsellor_col)
            if source_col: show_cols.append(source_col)
            if country_col: show_cols.append(country_col)
            prev = df_f.loc[mask, show_cols].copy() if show_cols else df_f.loc[mask].copy()

            # Append computed "Days Since" for clarity
            prev["_DaysSince"] = age_days.loc[prev.index]
            prev = prev.sort_values("_DaysSince", ascending=False)
            st.dataframe(prev.head(200), use_container_width=True)

            # Download
            st.download_button(
                "Download CSV – Inactivity snapshot",
                prev.to_csv(index=False).encode("utf-8"),
                file_name=f"inactivity_snapshot_{'activity' if pick=='Last Activity Date' else 'connected'}_ge_{thr}d.csv",
                mime="text/csv",
            )

elif view == "Trend & Analysis":
    st.subheader("Trend & Analysis – Grouped Drilldowns (Final rules)")

    # Group-by fields
    available_groups, group_map = [], {}
    if counsellor_col: available_groups.append("Academic Counsellor"); group_map["Academic Counsellor"] = counsellor_col
    if country_col:    available_groups.append("Country");            group_map["Country"] = country_col
    if source_col:     available_groups.append("JetLearn Deal Source"); group_map["JetLearn Deal Source"] = source_col

    sel_group_labels = st.multiselect("Group by (pick one or more)", options=available_groups, default=available_groups[:1] if available_groups else [])
    group_cols = [group_map[l] for l in sel_group_labels if l in group_map]

    # Mode
    level = st.radio("Mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ta_mode")

    # Date scope
    date_mode = st.radio("Date scope", ["This month", "Last month", "Custom date range"], index=0, horizontal=True, key="ta_dscope")
    if date_mode == "This month":
        range_start, range_end = month_bounds(today)
        st.caption(f"Scope: **This month** ({range_start} → {range_end})")
    elif date_mode == "Last month":
        range_start, range_end = last_month_bounds(today)
        st.caption(f"Scope: **Last month** ({range_start} → {range_end})")
    else:
        col_d1, col_d2 = st.columns(2)
        with col_d1: range_start = st.date_input("Start date", value=today.replace(day=1), key="ta_custom_start")
        with col_d2: range_end   = st.date_input("End date", value=month_bounds(today)[1], key="ta_custom_end")
        if range_end < range_start:
            st.error("End date cannot be before start date.")
            st.stop()
        st.caption(f"Scope: **Custom** ({range_start} → {range_end})")

    # Metric picker (includes derived)
    all_metrics = [
        "Payment Received Date — Count",
        "First Calibration Scheduled Date — Count",
        "Calibration Rescheduled Date — Count",
        "Calibration Done Date — Count",
        "Create Date (deals) — Count",
        "Future Calibration Scheduled — Count",
    ]
    metrics_selected = st.multiselect("Metrics to show", options=all_metrics, default=all_metrics, key="ta_metrics")

    metric_cols = {
        "Payment Received Date — Count": pay_col,
        "First Calibration Scheduled Date — Count": first_cal_sched_col,
        "Calibration Rescheduled Date — Count": cal_resched_col,
        "Calibration Done Date — Count": cal_done_col,
        "Create Date (deals) — Count": create_col,
        "Future Calibration Scheduled — Count": None,  # derived
    }

    # Missing column warnings
    miss = []
    for m in metrics_selected:
        if m == "Future Calibration Scheduled — Count":
            if (first_cal_sched_col is None or first_cal_sched_col not in df_f.columns) and \
               (cal_resched_col is None or cal_resched_col not in df_f.columns):
                miss.append("Future Calibration Scheduled (needs First and/or Rescheduled)")
        elif m != "Create Date (deals) — Count":
            if (metric_cols.get(m) is None) or (metric_cols.get(m) not in df_f.columns):
                miss.append(m)
    if miss:
        st.warning("Missing columns for: " + ", ".join(miss) + ". Those counts will show as 0.", icon="⚠️")

    # Build table
    def ta_count_table(
        df_scope: pd.DataFrame,
        group_cols: list[str],
        mode: str,
        range_start: date,
        range_end: date,
        create_col: str,
        metric_cols: dict,
        metrics_selected: list[str],
        *,
        first_cal_col: str | None,
        cal_resched_col: str | None,
    ) -> pd.DataFrame:

        if not group_cols:
            df_work = df_scope.copy()
            df_work["_GroupDummy"] = "All"
            group_cols_local = ["_GroupDummy"]
        else:
            df_work = df_scope.copy()
            group_cols_local = group_cols

        create_dt = coerce_datetime(df_work[create_col]).dt.date

        if first_cal_col and first_cal_col in df_work.columns:
            first_dt = coerce_datetime(df_work[first_cal_col])
        else:
            first_dt = pd.Series(pd.NaT, index=df_work.index)
        if cal_resched_col and cal_resched_col in df_work.columns:
            resch_dt = coerce_datetime(df_work[cal_resched_col])
        else:
            resch_dt = pd.Series(pd.NaT, index=df_work.index)

        eff_cal = resch_dt.copy().fillna(first_dt)
        eff_cal_date = eff_cal.dt.date

        pop_mask_mtd = create_dt.between(range_start, range_end)

        outs = []
        for disp in metrics_selected:
            col = metric_cols.get(disp)

            if disp == "Create Date (deals) — Count":
                idx = pop_mask_mtd if mode == "MTD" else create_dt.between(range_start, range_end)
                gdf = df_work.loc[idx, group_cols_local].copy()
                agg = gdf.assign(_one=1).groupby(group_cols_local)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols_local+[disp])
                outs.append(agg)
                continue

            if disp == "Future Calibration Scheduled — Count":
                if eff_cal_date is None:
                    base_idx = pop_mask_mtd if mode == "MTD" else slice(None)
                    target = df_work.loc[base_idx, group_cols_local] if mode == "MTD" else df_work[group_cols_local]
                    agg = target.assign(**{disp:0}).groupby(group_cols_local)[disp].sum().reset_index() if not target.empty else pd.DataFrame(columns=group_cols_local+[disp])
                    outs.append(agg)
                    continue
                future_mask = eff_cal_date > range_end
                idx = (pop_mask_mtd & future_mask) if mode == "MTD" else future_mask
                gdf = df_work.loc[idx, group_cols_local].copy()
                agg = gdf.assign(_one=1).groupby(group_cols_local)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols_local+[disp])
                outs.append(agg)
                continue

            if (not col) or (col not in df_work.columns):
                base_idx = pop_mask_mtd if mode == "MTD" else slice(None)
                target = df_work.loc[base_idx, group_cols_local] if mode == "MTD" else df_work[group_cols_local]
                agg = target.assign(**{disp:0}).groupby(group_cols_local)[disp].sum().reset_index() if not target.empty else pd.DataFrame(columns=group_cols_local+[disp])
                outs.append(agg)
                continue

            ev_date = coerce_datetime(df_work[col]).dt.date
            ev_in_range = ev_date.between(range_start, range_end)

            if mode == "MTD":
                idx = pop_mask_mtd & ev_in_range
            else:
                idx = ev_in_range

            gdf = df_work.loc[idx, group_cols_local].copy()
            agg = gdf.assign(_one=1).groupby(group_cols_local)["_one"].sum().reset_index().rename(columns={"_one": disp}) if not gdf.empty else pd.DataFrame(columns=group_cols_local+[disp])
            outs.append(agg)

        if outs:
            result = outs[0]
            for f in outs[1:]:
                result = result.merge(f, on=group_cols_local, how="outer")
        else:
            result = pd.DataFrame(columns=group_cols_local)

        for m in metrics_selected:
            if m not in result.columns:
                result[m] = 0
        result[metrics_selected] = result[metrics_selected].fillna(0).astype(int)
        if metrics_selected:
            result = result.sort_values(metrics_selected[0], ascending=False)
        return result.reset_index(drop=True)

    tbl = ta_count_table(
        df_scope=df_f,
        group_cols=group_cols,
        mode=level,
        range_start=range_start,
        range_end=range_end,
        create_col=create_col,
        metric_cols=metric_cols,
        metrics_selected=metrics_selected,
        first_cal_col=first_cal_sched_col,
        cal_resched_col=cal_resched_col,
    )

    st.markdown("### Output")
    if tbl.empty:
        st.info("No rows match the selected filters and date range.")
    else:
        rename_map = {group_map.get(lbl): lbl for lbl in sel_group_labels}
        show = tbl.rename(columns=rename_map)
        st.dataframe(show, use_container_width=True)

        csv = show.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (Trend & Analysis)", data=csv, file_name="trend_analysis_final.csv", mime="text/csv")

    # ---------------------------------------------------------------------
    # NEW: Referral business — Created vs Converted by Referral Intent Source
    # ---------------------------------------------------------------------
    st.markdown("### Referral business — Created vs Converted by Referral Intent Source")
    referral_intent_col = find_col(df, ["Referral Intent Source", "Referral intent source"])

    if (not referral_intent_col) or (referral_intent_col not in df_f.columns):
        st.info("Referral Intent Source column not found.")
    else:
        d_ref = df_f.copy()
        d_ref["_ref"] = d_ref[referral_intent_col].fillna("Unknown").astype(str).str.strip()

        exclude_unknown = st.checkbox("Exclude 'Unknown' (Referral Intent Source)", value=False, key="ta_ref_exclude")
        if exclude_unknown:
            d_ref = d_ref[d_ref["_ref"] != "Unknown"]

        # Normalize dates
        _cdate_r = coerce_datetime(d_ref[create_col]).dt.date if create_col in d_ref.columns else pd.Series(pd.NaT, index=d_ref.index)
        _pdate_r = coerce_datetime(d_ref[pay_col]).dt.date    if pay_col in d_ref.columns    else pd.Series(pd.NaT, index=d_ref.index)

        m_created_r = _cdate_r.between(range_start, range_end) if _cdate_r.notna().any() else pd.Series(False, index=d_ref.index)
        m_paid_r    = _pdate_r.between(range_start, range_end) if _pdate_r.notna().any() else pd.Series(False, index=d_ref.index)

        if level == "MTD":
            # Count payments only from deals whose Create Date is in scope
            created_mask_r   = m_created_r
            converted_mask_r = m_created_r & m_paid_r
        else:
            # Cohort: payments in scope regardless of create-month
            created_mask_r   = m_created_r
            converted_mask_r = m_paid_r

        ref_tbl = pd.DataFrame({
            "Referral Intent Source": d_ref["_ref"],
            "Created":  created_mask_r.astype(int),
            "Converted": converted_mask_r.astype(int),
        })
        grp = (
            ref_tbl.groupby("Referral Intent Source", as_index=False)
                   .sum(numeric_only=True)
                   .sort_values("Created", ascending=False)
        )

        # Controls
        col_r1, col_r2 = st.columns([1,1])
        with col_r1:
            top_k_ref = st.number_input("Show top N Referral Intent Sources", min_value=1, max_value=max(1, len(grp)), value=min(20, len(grp)) if len(grp) else 1, step=1, key="ta_ref_topn")
        with col_r2:
            sort_metric_ref = st.selectbox("Sort by", ["Created (desc)", "Converted (desc)", "A–Z"], index=0, key="ta_ref_sort")

        if sort_metric_ref == "Converted (desc)":
            grp = grp.sort_values("Converted", ascending=False)
        elif sort_metric_ref == "A–Z":
            grp = grp.sort_values("Referral Intent Source", ascending=True)
        else:
            grp = grp.sort_values("Created", ascending=False)

        grp_show = grp.head(int(top_k_ref)) if len(grp) > int(top_k_ref) else grp

        # Chart
        melt_ref = grp_show.melt(
            id_vars=["Referral Intent Source"],
            value_vars=["Created", "Converted"],
            var_name="Metric",
            value_name="Count"
        )
        chart_ref = (
            alt.Chart(melt_ref)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Referral Intent Source:N", sort=grp_show["Referral Intent Source"].tolist(), title="Referral Intent Source"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Metric:N", title="", legend=alt.Legend(orient="bottom")),
                xOffset=alt.XOffset("Metric:N"),
                tooltip=[alt.Tooltip("Referral Intent Source:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Created vs Converted by Referral Intent Source ({level})")
        )
        st.altair_chart(chart_ref, use_container_width=True)

        # Table + download
        st.dataframe(grp_show, use_container_width=True)
        st.download_button(
            "Download CSV — Referral business (Created vs Converted)",
            data=grp_show.to_csv(index=False).encode("utf-8"),
            file_name=f"trend_referral_business_{level.lower()}_{range_start}_{range_end}.csv",
            mime="text/csv",
            key="ta_ref_business_dl"
        )



elif view == "80-20":
    # Everything for 80-20 lives INSIDE this tab (own controls; no sidebar widgets)
    st.subheader("80-20 Pareto + Trajectory + Conversion% + Mix Analyzer")

    # Precompute for this module
    df80 = df.copy()
    df80["_pay_dt"] = coerce_datetime(df80[pay_col])
    df80["_create_dt"] = coerce_datetime(df80[create_col])
    df80["_pay_m"] = df80["_pay_dt"].dt.to_period("M")

    # ✅ Apply Track filter to 80-20 too
    if track != "Both":
        if pipeline_col and pipeline_col in df80.columns:
            _norm80 = df80[pipeline_col].map(normalize_pipeline).fillna("Other")
            before_ct = len(df80)
            df80 = df80.loc[_norm80 == track].copy()
            st.caption(f"80-20 scope after Track filter ({track}): **{len(df80):,}** rows (was {before_ct:,}).")
        else:
            st.warning("Pipeline column not found — the Track filter can’t be applied in 80-20.", icon="⚠️")

    if source_col:
        df80["_src_raw"] = df80[source_col].fillna("Unknown").astype(str)
    else:
        df80["_src_raw"] = "Other"

    # ---- Cohort scope / date pickers (in-tab)
    st.markdown("#### Cohort scope (Payment Received)")
    unique_months = df80["_pay_dt"].dropna().dt.to_period("M").drop_duplicates().sort_values()
    month_labels = [str(p) for p in unique_months]
    use_custom = st.toggle("Use custom date range", value=False, key="eighty_use_custom")

    if not use_custom and len(month_labels) > 0:
        month_pick = st.selectbox("Cohort month", month_labels, index=len(month_labels)-1, key="eighty_month_pick")
        y, m = map(int, month_pick.split("-"))
        start_d = date(y, m, 1)
        end_d = date(y, m, monthrange(y, m)[1])
    else:
        default_start = df80["_pay_dt"].min().date() if df80["_pay_dt"].notna().any() else date.today().replace(day=1)
        default_end   = df80["_pay_dt"].max().date() if df80["_pay_dt"].notna().any() else date.today()
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start date", value=default_start, key="eighty_start")
        with c2: end_d   = st.date_input("End date", value=default_end, key="eighty_end")
        if end_d < start_d:
            st.error("End date cannot be before start date.")
            st.stop()

    # Source include list (Pareto/Cohort) using _src_raw (includes Unknown)
    st.markdown("#### Source filter (for Pareto & Cohort views)")
    if source_col:
        all_sources = sorted(df80["_src_raw"].unique().tolist())
        excl_ref = st.checkbox("Exclude Referral (for Pareto view)", value=False, key="eighty_excl_ref")
        sources_for_pick = [s for s in all_sources if not (excl_ref and "referr" in s.lower())]
        picked_sources = st.multiselect("Include Deal Sources (Pareto)", options=sources_for_pick, default=sources_for_pick, key="eighty_picked_src")
    else:
        picked_sources = None
        st.info("Deal Source column not found; Pareto by source will be limited.")

    # ---- Range KPI (Created vs Enrolments)
    in_create_window = df80["_create_dt"].dt.date.between(start_d, end_d)
    deals_created = int(in_create_window.sum())

    in_pay_window = df80["_pay_dt"].dt.date.between(start_d, end_d)
    enrolments = int(in_pay_window.sum())

    conv_pct_simple = (enrolments / deals_created * 100.0) if deals_created > 0 else 0.0

    st.markdown("<div class='section-title'>Range KPI — Deals Created vs Enrolments</div>", unsafe_allow_html=True)
    cA, cB, cC = st.columns(3)
    with cA:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Deals Created</div><div class='kpi-value'>{deals_created:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with cB:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div><div class='kpi-value'>{enrolments:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with cC:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversion% (Payments / Created)</div><div class='kpi-value'>{conv_pct_simple:.1f}%</div><div class='kpi-sub'>Num: {enrolments:,} • Den: {deals_created:,}</div></div>", unsafe_allow_html=True)

    # ---- Build cohort df for 80-20 section (respect picked_sources)
    scope_mask = df80["_pay_dt"].dt.date.between(start_d, end_d)
    df_cohort = df80.loc[scope_mask].copy()
    if picked_sources is not None and source_col:
        df_cohort = df_cohort[df_cohort["_src_raw"].isin(picked_sources)]

    # ---- Cohort KPIs
    st.markdown("<div class='section-title'>Cohort KPIs</div>", unsafe_allow_html=True)
    total_enr = int(len(df_cohort))
    if source_col and source_col in df_cohort.columns:
        ref_cnt = int(df_cohort[source_col].fillna("").str.contains("referr", case=False).sum())
    else:
        ref_cnt = 0
    ref_pct = (ref_cnt/total_enr*100.0) if total_enr > 0 else 0.0

    src_tbl = build_pareto(df_cohort, source_col, "Deal Source") if total_enr > 0 else pd.DataFrame()
    cty_tbl = build_pareto(df_cohort, country_col, "Country") if total_enr > 0 else pd.DataFrame()
    n_sources_80 = int((src_tbl["CumPct"] <= 80).sum()) if not src_tbl.empty else 0
    n_countries_80 = int((cty_tbl["CumPct"] <= 80).sum()) if not cty_tbl.empty else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Cohort Enrolments</div><div class='kpi-value'>{total_enr:,}</div><div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with k2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Referral % (cohort)</div><div class='kpi-value'>{ref_pct:.1f}%</div><div class='kpi-sub'>{ref_cnt:,} of {total_enr:,}</div></div>", unsafe_allow_html=True)
    with k3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Sources for 80%</div><div class='kpi-value'>{n_sources_80}</div></div>", unsafe_allow_html=True)
    with k4: st.markdown(f"<div class='kpi-card'><div class='kpi-title'># Countries for 80%</div><div class='kpi-value'>{n_countries_80}</div></div>", unsafe_allow_html=True)

    # ---- 80-20 Charts
    c1, c2 = st.columns([2,1])
    with c1: st.altair_chart(pareto_chart(src_tbl, "Deal Source", "Pareto – Enrolments by Deal Source"), use_container_width=True)
    with c2:
        # Donut: Referral vs Non-Referral in cohort
        if source_col and source_col in df_cohort.columns and not df_cohort.empty:
            s = df_cohort[source_col].fillna("Unknown").astype(str)
            is_ref = s.str.contains("referr", case=False, na=False)
            pie = pd.DataFrame({"Category": ["Referral", "Non-Referral"], "Value": [int(is_ref.sum()), int((~is_ref).sum())]})
            donut = alt.Chart(pie).mark_arc(innerRadius=70).encode(
                theta="Value:Q",
                color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
                tooltip=["Category:N", "Value:Q"]
            ).properties(title="Referral vs Non-Referral (cohort)")
            st.altair_chart(donut, use_container_width=True)
        else:
            st.info("Referral split not available (missing source column or empty cohort).")
    st.altair_chart(pareto_chart(cty_tbl, "Country", "Pareto – Enrolments by Country"), use_container_width=True)

    # ---- Conversion% by Key Source
    st.markdown("### Conversion% by Key Source (range-based)")
    def conversion_stats(df_all: pd.DataFrame, start_d: date, end_d: date):
        if create_col is None or pay_col is None:
            return pd.DataFrame(columns=["KeySource","Den","Num","Pct"])
        d = df_all.copy()
        d["_cdate"] = coerce_datetime(d[create_col]).dt.date
        d["_pdate"] = coerce_datetime(d[pay_col]).dt.date
        d["_key_source"] = d[source_col].apply(normalize_key_source) if source_col else "Other"

        denom_mask = d["_cdate"].between(start_d, end_d)
        num_mask = d["_pdate"].between(start_d, end_d)

        rows = []
        for src in ["Referral", "PM - Search", "PM - Social"]:
            src_mask = (d["_key_source"] == src)
            den = int((denom_mask & src_mask).sum())
            num = int((num_mask & src_mask).sum())
            pct = (num/den*100.0) if den > 0 else 0.0
            rows.append({"KeySource": src, "Den": den, "Num": num, "Pct": pct})
        return pd.DataFrame(rows)

    bysrc_conv = conversion_stats(df80, start_d, end_d)
    if not bysrc_conv.empty:
        conv_chart = alt.Chart(bysrc_conv).mark_bar(opacity=0.9).encode(
            x=alt.X("KeySource:N", sort=["Referral","PM - Search","PM - Social"], title="Source"),
            y=alt.Y("Pct:Q", title="Conversion%"),
            tooltip=[alt.Tooltip("KeySource:N"), alt.Tooltip("Den:Q", title="Deals (Created)"),
                     alt.Tooltip("Num:Q", title="Enrolments (Payments)"), alt.Tooltip("Pct:Q", title="Conversion%", format=".1f")]
        ).properties(height=300, title=f"Conversion% (Payments / Created) • {start_d} → {end_d}")
        st.altair_chart(conv_chart, use_container_width=True)
    else:
        st.info("No data to compute Conversion% by key source for this window.")

    # ---- Trajectory – Top Countries × (Key or Raw Deal Sources)
    st.markdown("### Trajectory – Top Countries × Referral / PM - Search / PM - Social (or All Raw Sources)")
    col_t1, col_t2, col_tg, col_t3 = st.columns([1, 1, 1.4, 1.6])
    with col_t1:
        trailing_k = st.selectbox("Trailing window (months)", [3, 6, 12], index=0, key="eighty_trailing")
    with col_t2:
        top_k = st.selectbox("Top countries (by cohort enrolments)", [5, 7], index=0, key="eighty_topk")
    with col_tg:
        traj_grouping = st.radio(
            "Source grouping",
            ["Key (Referral/PM-Search/PM-Social/Other)", "Raw (all)"],
            index=0, horizontal=False, key="eighty_grouping"
        )

    months_list = months_back_list(end_d, trailing_k)
    months_str = [str(p) for p in months_list]
    df_trail = df80[df80["_pay_m"].isin(months_list)].copy()

    if traj_grouping.startswith("Key"):
        df_trail["_traj_source"] = df_trail[source_col].apply(normalize_key_source) if source_col else "Other"
        traj_source_options = ["All sources", "Referral", "PM - Search", "PM - Social", "Other"]
    else:
        df_trail["_traj_source"] = df_trail[source_col].fillna("Unknown").astype(str) if source_col else "Other"
        unique_raw = sorted(df_trail["_traj_source"].dropna().unique().tolist())
        traj_source_options = ["All sources"] + unique_raw

    with col_t3:
        traj_src_pick = st.selectbox("Deal Source for Top Countries", options=traj_source_options, index=0, key="eighty_srcpick")

    if traj_src_pick != "All sources":
        df_trail_src = df_trail[df_trail["_traj_source"] == traj_src_pick].copy()
    else:
        df_trail_src = df_trail.copy()

    if country_col and not df_trail_src.empty:
        cty_counts = df_trail_src.groupby(country_col).size().sort_values(ascending=False)
        top_countries = cty_counts.head(top_k).index.astype(str).tolist()
    else:
        top_countries = []

    monthly_total = df_trail.groupby("_pay_m").size().rename("TotalAll").reset_index()

    if top_countries and source_col and country_col:
        mcs = (
        df_trail_src[df_trail_src[country_col].astype(str).isin(top_countries)]
        .groupby(["_pay_m", country_col, "_traj_source"]).size().rename("Cnt").reset_index()
    )
    else:
        mcs = pd.DataFrame(columns=["_pay_m", country_col if country_col else "Country", "_traj_source", "Cnt"])

    if not mcs.empty:
        mcs = mcs.merge(monthly_total, on="_pay_m", how="left")
        mcs["PctOfOverall"] = np.where(mcs["TotalAll"]>0, mcs["Cnt"]/mcs["TotalAll"]*100.0, 0.0)
        mcs["_pay_m_str"] = pd.Categorical(mcs["_pay_m"].astype(str), categories=months_str, ordered=True)
        # safe categorical cleanup
        mcs["_pay_m_str"] = mcs["_pay_m_str"].cat.remove_unused_categories()

    if not mcs.empty:
        # sort legend by frequency
        src_order = mcs["_traj_source"].value_counts().index.tolist()
        title_suffix = f"{traj_src_pick}" if traj_src_pick != "All sources" else "All sources"
        grouping_suffix = "Key" if traj_grouping.startswith("Key") else "Raw"

        facet_chart = alt.Chart(mcs).mark_bar(opacity=0.9).encode(
            x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
            y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("_traj_source:N", title="Source", sort=src_order),
            tooltip=[
                alt.Tooltip("_pay_m_str:N", title="Month"),
                alt.Tooltip(f"{country_col}:N", title="Country") if country_col else alt.Tooltip("_pay_m_str:N"),
                alt.Tooltip("_traj_source:N", title="Source"),
                alt.Tooltip("Cnt:Q", title="Count"),
                alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
            ],
        ).properties(
            height=220,
            title=f"Top Countries • Basis: {title_suffix} • Grouping: {grouping_suffix}",
        ).facet(
            column=alt.Column(f"{country_col}:N", title="Top Countries", sort=top_countries)
        )
        st.altair_chart(facet_chart, use_container_width=True)

        # Overall contribution lines (only within chosen top countries)
        overall = (
            mcs
            .assign(_pay_m_str=mcs["_pay_m_str"].astype(str))
            .groupby(["_pay_m_str","_traj_source"], observed=True, as_index=False)
            .agg(Cnt=("Cnt","sum"), TotalAll=("TotalAll","first"))
        )
        overall["PctOfOverall"] = np.where(overall["TotalAll"]>0, overall["Cnt"]/overall["TotalAll"]*100.0, 0.0)

        lines = alt.Chart(overall).mark_line(point=True).encode(
            x=alt.X("_pay_m_str:N", title="Month", sort=months_str),
            y=alt.Y("PctOfOverall:Q", title="% of overall business (Top countries)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("_traj_source:N", title="Source", sort=src_order),
            tooltip=[
                alt.Tooltip("_pay_m_str:N", title="Month"),
                alt.Tooltip("_traj_source:N", title="Source"),
                alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
            ],
        ).properties(
            title=f"Overall contribution by source (Top countries • Basis: {title_suffix} • Grouping: {grouping_suffix})",
            height=320,
        )
        st.altair_chart(lines, use_container_width=True)
    else:
        st.info("No data for the selected trailing window to build the trajectory.", icon="ℹ️")

    # =========================
    # Interactive Mix Analyzer
    # =========================
    st.markdown("### Interactive Mix Analyzer — % of overall business from your selection")

    col_im1, col_im2 = st.columns([1.6, 1])
    with col_im1:
        use_key_sources = st.checkbox(
            "Use key-source mapping (Referral / PM - Search / PM - Social)",
            value=True,
            key="eighty_use_key_sources",
            help="On = group sources into 3 key buckets. Off = raw deal source names.",
        )

    # Cohort within window (payments inside window)
    cohort_now = df80[df80["_pay_dt"].dt.date.between(start_d, end_d)].copy()
    cohort_now = assign_src_pick(cohort_now, source_col, use_key_sources)

    # Source option list
    if source_col and source_col in cohort_now.columns:
        if use_key_sources:
            src_options = ["Referral", "PM - Search", "PM - Social", "Other"]
            default_srcs = ["Referral"]
        else:
            src_options = sorted(cohort_now["_src_pick"].unique().tolist())
            default_srcs = src_options[:1] if src_options else []
        picked_srcs = st.multiselect(
            "Select Deal Sources",
            options=src_options,
            default=[s for s in default_srcs if s in src_options],
            key="eighty_mix_sources_pick",
            help="Pick one or more sources. Each source gets its own Country control below.",
        )
    else:
        picked_srcs = []
        st.info("Deal Source column not found, source filtering disabled for Mix Analyzer.")

    # Session keys helpers
    def _mode_key(src): return f"eighty_src_mode::{src}"
    def _countries_key(src): return f"eighty_src_countries::{src}"

    DISPLAY_ANY = "Any country (all)"
    per_source_config = {}  # src -> dict(mode, countries, available)

    for src in picked_srcs:
        available = (
            cohort_now.loc[cohort_now["_src_pick"] == src, country_col]
            .astype(str).fillna("Unknown").value_counts().index.tolist()
            if country_col and country_col in cohort_now.columns else []
        )
        if _mode_key(src) not in st.session_state:
            st.session_state[_mode_key(src)] = "All"
        if _countries_key(src) not in st.session_state:
            st.session_state[_countries_key(src)] = available.copy()

        if st.session_state[_mode_key(src)] == "Specific":
            prev = st.session_state[_countries_key(src)]
            st.session_state[_countries_key(src)] = [c for c in prev if (c in available) or (c == DISPLAY_ANY)]
            if not st.session_state[_countries_key(src)] and available:
                st.session_state[_countries_key(src)] = available[:5]

        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"**Source:** {src}")
                mode = st.radio(
                    "Country scope",
                    options=["All", "None", "Specific"],
                    index=["All", "None", "Specific"].index(st.session_state[_mode_key(src)]),
                    key=_mode_key(src),
                    horizontal=True,
                )
            with c2:
                if mode == "Specific":
                    options = [DISPLAY_ANY] + available
                    st.multiselect(
                        f"Countries for {src}",
                        options=options,
                        default=st.session_state[_countries_key(src)],
                        key=_countries_key(src),
                        help="Pick countries or choose 'Any country (all)' to include all countries for this source.",
                    )
                elif mode == "All":
                    st.caption(f"All countries for **{src}** ({len(available)}).")
                else:
                    st.caption(f"Excluded **{src}** (no countries).")

        per_source_config[src] = {
            "mode": st.session_state[_mode_key(src)],
            "countries": st.session_state[_countries_key(src)],
            "available": available,
        }

    # Build masks from per-source config
    def make_union_mask(df_in: pd.DataFrame, per_cfg: dict, use_key: bool) -> pd.Series:
        d = assign_src_pick(df_in, source_col, use_key)
        base = pd.Series(False, index=d.index)
        if not per_cfg:
            return base
        if country_col and country_col in d.columns:
            c_series = d[country_col].astype(str).fillna("Unknown")
        else:
            c_series = pd.Series("Unknown", index=d.index)

        for src, info in per_cfg.items():
            mode = info["mode"]
            if mode == "None":
                continue
            src_mask = (d["_src_pick"] == src)
            if mode == "All":
                base = base | src_mask
            else:  # Specific
                chosen = set(info["countries"])
                if not chosen:
                    continue
                if DISPLAY_ANY in chosen:
                    base = base | src_mask
                else:
                    base = base | (src_mask & c_series.isin(chosen))
        return base

    def active_sources(per_cfg: dict) -> list[str]:
        return [s for s, v in per_cfg.items() if v["mode"] != "None"]

    mix_view = st.radio(
        "Mix view",
        ["Aggregate (range total)", "Month-wise"],
        index=0,
        horizontal=True,
        key="eighty_mix_view",
        help="Aggregate = single % for whole range. Month-wise = monthly % time series with one line per picked source.",
    )

    total_payments = int(len(cohort_now))
    if total_payments == 0:
        st.warning("No payments (enrolments) in the selected window.", icon="⚠️")
    else:
        sel_mask = make_union_mask(cohort_now, per_source_config, use_key_sources)
        if not sel_mask.any():
            st.info("No selection applied (pick at least one source in All/Specific).")
        else:
            selected_payments = int(sel_mask.sum())
            pct_of_overall = (selected_payments / total_payments * 100.0) if total_payments > 0 else 0.0

            st.markdown(
                f"<div class='kpi-card'>"
                f"<div class='kpi-title'>Contribution of your selection ({start_d} → {end_d})</div>"
                f"<div class='kpi-value'>{pct_of_overall:.1f}%</div>"
                f"<div class='kpi-sub'>Enrolments in selection: {selected_payments:,} • Total: {total_payments:,}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Quick breakdown by source
            dsel = cohort_now.loc[sel_mask].copy()
            if not dsel.empty:
                bysrc = dsel.groupby("_src_pick").size().rename("SelCnt").reset_index()
                bysrc["PctOfOverall"] = bysrc["SelCnt"] / total_payments * 100.0
                chart = alt.Chart(bysrc).mark_bar(opacity=0.9).encode(
                    x=alt.X("_src_pick:N", title="Source"),
                    y=alt.Y("PctOfOverall:Q", title="% of overall business"),
                    tooltip=[
                        alt.Tooltip("_src_pick:N", title="Source"),
                        alt.Tooltip("SelCnt:Q", title="Enrolments (selected)"),
                        alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
                    ],
                    color=alt.Color("_src_pick:N", legend=alt.Legend(orient="bottom")),
                ).properties(height=320, title="Selection breakdown by source — % of overall")
                st.altair_chart(chart, use_container_width=True)

            # Month-wise lines
            if mix_view == "Month-wise":
                cohort_now["_pay_m"] = cohort_now["_pay_dt"].dt.to_period("M")
                months_in_range = (
                    cohort_now["_pay_m"].dropna().sort_values().unique().astype(str).tolist()
                )

                # Overall monthly totals
                overall_m = cohort_now.groupby("_pay_m").size().rename("TotalAll").reset_index()
                overall_m["Month"] = overall_m["_pay_m"].astype(str)

                # All Selected monthly counts using union mask
                all_sel_m = cohort_now.loc[sel_mask].groupby("_pay_m").size().rename("SelCnt").reset_index()
                all_sel_m["Month"] = all_sel_m["_pay_m"].astype(str)

                all_line = overall_m.merge(all_sel_m[["_pay_m","SelCnt","Month"]], on=["_pay_m","Month"], how="left").fillna({"SelCnt":0})
                all_line["PctOfOverall"] = np.where(all_line["TotalAll"]>0, all_line["SelCnt"]/all_line["TotalAll"]*100.0, 0.0)
                all_line["Series"] = "All Selected"
                all_line = all_line[["Month","Series","SelCnt","TotalAll","PctOfOverall"]]
                all_line["Month"] = pd.Categorical(all_line["Month"], categories=months_in_range, ordered=True)

                # Per-source monthly lines honoring each source's country selection
                per_src_frames = []
                for src in active_sources(per_source_config):
                    one_cfg = {src: per_source_config[src]}
                    smask = make_union_mask(cohort_now, one_cfg, use_key_sources)
                    s_cnt = cohort_now.loc[smask].groupby("_pay_m").size().rename("SelCnt").reset_index()
                    if s_cnt.empty:
                        continue
                    s_cnt["Month"] = s_cnt["_pay_m"].astype(str)
                    s_join = overall_m.merge(s_cnt[["_pay_m","SelCnt","Month"]], on=["_pay_m","Month"], how="left").fillna({"SelCnt":0})
                    s_join["PctOfOverall"] = np.where(s_join["TotalAll"]>0, s_join["SelCnt"]/s_join["TotalAll"]*100.0, 0.0)
                    s_join["Series"] = src
                    s_join = s_join[["Month","Series","SelCnt","TotalAll","PctOfOverall"]]
                    s_join["Month"] = pd.Categorical(s_join["Month"], categories=months_in_range, ordered=True)
                    per_src_frames.append(s_join)

                if per_src_frames:
                    lines_df = pd.concat([all_line] + per_src_frames, ignore_index=True)
                else:
                    lines_df = all_line.copy()

                avg_monthly_pct = lines_df.loc[lines_df["Series"]=="All Selected", "PctOfOverall"].mean() if not lines_df.empty else 0.0
                st.markdown(
                    f"<div class='kpi-card'>"
                    f"<div class='kpi-title'>Month-wise: average % contribution (All Selected)</div>"
                    f"<div class='kpi-value'>{avg_monthly_pct:.1f}%</div>"
                    f"<div class='kpi-sub'>Months: {lines_df['Month'].nunique() if not lines_df.empty else 0}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                stroke_width = alt.condition("datum.Series == 'All Selected'", alt.value(4), alt.value(2))
                chart = alt.Chart(lines_df).mark_line(point=True).encode(
                    x=alt.X("Month:N", sort=months_in_range, title="Month"),
                    y=alt.Y("PctOfOverall:Q", title="% of overall business", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Series:N", title="Series"),
                    strokeWidth=stroke_width,
                    tooltip=[
                        alt.Tooltip("Month:N"),
                        alt.Tooltip("Series:N"),
                        alt.Tooltip("SelCnt:Q", title="Enrolments (selected)"),
                        alt.Tooltip("TotalAll:Q", title="Total enrolments"),
                        alt.Tooltip("PctOfOverall:Q", title="% of overall", format=".1f"),
                    ],
                ).properties(height=360, title="Month-wise % of overall — All Selected vs each picked source")
                st.altair_chart(chart, use_container_width=True)

    # =========================
    # Deals vs Enrolments — current selection
    # =========================
    st.markdown("### Deals vs Enrolments — for your current selection")
    def _build_created_paid_monthly(df_all: pd.DataFrame, start_d: date, end_d: date) -> tuple[pd.DataFrame, pd.DataFrame]:
        d = df_all.copy()
        d["_cdate"] = coerce_datetime(d[create_col]).dt.date
        d["_pdate"] = coerce_datetime(d[pay_col]).dt.date
        d["_cmonth"] = coerce_datetime(d[create_col]).dt.to_period("M")
        d["_pmonth"] = coerce_datetime(d[pay_col]).dt.to_period("M")

        cwin = d["_cdate"].between(start_d, end_d)
        pwin = d["_pdate"].between(start_d, end_d)

        month_index = pd.period_range(start=start_d.replace(day=1), end=end_d.replace(day=1), freq="M")

        created_m = (
            d.loc[cwin].groupby("_cmonth").size()
              .reindex(month_index, fill_value=0)
              .rename_axis(index="_month").reset_index(name="CreatedCnt")
        )
        paid_m = (
            d.loc[pwin].groupby("_pmonth").size()
              .reindex(month_index, fill_value=0)
              .rename_axis(index="_month").reset_index(name="PaidCnt")
        )

        monthly = created_m.merge(paid_m, on="_month", how="outer").fillna(0)
        monthly["Month"] = monthly["_month"].astype(str)
        monthly = monthly[["Month", "CreatedCnt", "PaidCnt"]]
        monthly["ConvPct"] = np.where(monthly["CreatedCnt"] > 0,
                                      monthly["PaidCnt"] / monthly["CreatedCnt"] * 100.0, 0.0)

        total_created = int(monthly["CreatedCnt"].sum())
        total_paid    = int(monthly["PaidCnt"].sum())
        agg = pd.DataFrame({
            "CreatedCnt": [total_created],
            "PaidCnt":    [total_paid],
            "ConvPct":    [float((total_paid / total_created * 100.0) if total_created > 0 else 0.0)]
        })
        return monthly, agg

    if picked_srcs:
        union_mask_all = make_union_mask(df80, per_source_config, use_key_sources)
    else:
        union_mask_all = pd.Series(False, index=df80.index)

    df_sel_all = df80.loc[union_mask_all].copy()
    monthly_sel, agg_sel = _build_created_paid_monthly(df_sel_all, start_d, end_d)

    kpa, kpb, kpc = st.columns(3)
    with kpa:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Deals (Created)</div>"
            f"<div class='kpi-value'>{int(agg_sel['CreatedCnt'].iloc[0]) if not agg_sel.empty else 0:,}</div>"
            f"<div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with kpb:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Enrolments (Payments)</div>"
            f"<div class='kpi-value'>{int(agg_sel['PaidCnt'].iloc[0]) if not agg_sel.empty else 0:,}</div>"
            f"<div class='kpi-sub'>{start_d} → {end_d}</div></div>", unsafe_allow_html=True)
    with kpc:
        conv_val = float(agg_sel['ConvPct'].iloc[0]) if not agg_sel.empty else 0.0
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Conversion% (Payments / Created)</div>"
            f"<div class='kpi-value'>{conv_val:.1f}%</div>"
            f"<div class='kpi-sub'>Num: {int(agg_sel['PaidCnt'].iloc[0]) if not agg_sel.empty else 0:,} • Den: {int(agg_sel['CreatedCnt'].iloc[0]) if not agg_sel.empty else 0:,}</div></div>",
            unsafe_allow_html=True)

    show_conv_line = st.checkbox("Overlay Conversion% line on bars", value=True, key="eighty_mix_conv_line")

    if not monthly_sel.empty:
        bar_df = monthly_sel.melt(
            id_vars=["Month"],
            value_vars=["CreatedCnt", "PaidCnt"],
            var_name="Metric",
            value_name="Count"
        )
        bar_df["Metric"] = bar_df["Metric"].map({"CreatedCnt": "Deals Created", "PaidCnt": "Enrolments"})

        bars = alt.Chart(bar_df).mark_bar(opacity=0.9).encode(
            x=alt.X("Month:N", sort=monthly_sel["Month"].tolist(), title="Month"),
            y=alt.Y("Count:Q", title="Count"),
            color=alt.Color("Metric:N", title=""),
            xOffset=alt.XOffset("Metric:N"),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")],
        ).properties(height=360, title="Month-wise — Deals & Enrolments (bars)")

        if show_conv_line:
            line = alt.Chart(monthly_sel).mark_line(point=True).encode(
                x=alt.X("Month:N", sort=monthly_sel["Month"].tolist(), title="Month"),
                y=alt.Y("ConvPct:Q", title="Conversion%", axis=alt.Axis(orient="right")),
                tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("ConvPct:Q", title="Conversion%", format=".1f")],
                color=alt.value("#16a34a"),
            )
            st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent'), use_container_width=True)
        else:
            st.altair_chart(bars, use_container_width=True)

        with st.expander("Download: Month-wise Deals / Enrolments / Conversion% (selection)"):
            out_tbl = monthly_sel.rename(columns={
                "CreatedCnt": "Deals Created",
                "PaidCnt": "Enrolments",
                "ConvPct": "Conversion %"
            })
            st.dataframe(out_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – Month-wise Deals/Enrolments/Conversion",
                data=out_tbl.to_csv(index=False).encode("utf-8"),
                file_name="selection_deals_enrolments_conversion_monthwise.csv",
                mime="text/csv",
                key="eighty_download_monthwise",
            )
    else:
        st.info("No month-wise data to plot for the current selection. Pick at least one source in All/Specific.")

    # ----------------------------
    # Tables + Downloads
    # ----------------------------
    st.markdown("<div class='section-title'>Tables</div>", unsafe_allow_html=True)
    tabs80 = st.tabs(["Deal Source 80-20", "Country 80-20", "Cohort Rows", "Trajectory table", "Conversion by Source"])

    with tabs80[0]:
        if src_tbl.empty:
            st.info("No enrollments in scope.")
        else:
            st.dataframe(src_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – Deal Source Pareto",
                src_tbl.to_csv(index=False).encode("utf-8"),
                "pareto_deal_source.csv",
                "text/csv",
                key="eighty_dl_srcpareto",
            )

    with tabs80[1]:
        if cty_tbl.empty:
            st.info("No enrollments in scope.")
        else:
            st.dataframe(cty_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – Country Pareto",
                cty_tbl.to_csv(index=False).encode("utf-8"),
                "pareto_country.csv",
                "text/csv",
                key="eighty_dl_ctypareto",
            )

    with tabs80[2]:
        show_cols = []
        if create_col: show_cols.append(create_col)
        if pay_col: show_cols.append(pay_col)
        if source_col: show_cols.append(source_col)
        if country_col: show_cols.append(country_col)
        preview = df_cohort[show_cols].copy() if show_cols else df_cohort.copy()
        st.dataframe(preview.head(1000), use_container_width=True)
        st.download_button(
            "Download CSV – Cohort subset",
            preview.to_csv(index=False).encode("utf-8"),
            "cohort_subset.csv",
            "text/csv",
            key="eighty_dl_cohort",
        )

    with tabs80[3]:
        if 'mcs' in locals() and not mcs.empty:
            show = mcs.rename(columns={country_col: "Country"})[["Country","_pay_m_str","_traj_source","Cnt","TotalAll","PctOfOverall"]]
            show = show.sort_values(["Country","_pay_m_str","_traj_source"])
            st.dataframe(show, use_container_width=True)
            st.download_button(
                "Download CSV – Trajectory",
                show.to_csv(index=False).encode("utf-8"),
                "trajectory_top_countries_sources.csv",
                "text/csv",
                key="eighty_dl_traj",
            )
        else:
            st.info("No trajectory table for the current selection.")

    with tabs80[4]:
        if not bysrc_conv.empty:
            st.dataframe(bysrc_conv, use_container_width=True)
            st.download_button(
                "Download CSV – Conversion by Key Source",
                bysrc_conv.to_csv(index=False).encode("utf-8"),
                "conversion_by_key_source.csv",
                "text/csv",
                key="eighty_dl_conv",
            )
        else:
            st.info("No conversion table for the current selection.")

elif view == "Stuck deals":
    st.subheader("Stuck deals – Funnel & Propagation (Created → Trial → Cal Done → Payment)")

    # ==== Column presence (warn but never stop)
    missing_cols = []
    for col_label, col_var in [
        ("Create Date", create_col),
        ("First Calibration Scheduled Date", first_cal_sched_col),
        ("Calibration Rescheduled Date", cal_resched_col),
        ("Calibration Done Date", cal_done_col),
        ("Payment Received Date", pay_col),
    ]:
        if not col_var or col_var not in df_f.columns:
            missing_cols.append(col_label)
    if missing_cols:
        st.warning(
            "Missing columns: " + ", ".join(missing_cols) +
            ". Funnel/metrics will skip the missing stages where applicable.",
            icon="⚠️"
        )

    # Try to find the Slot column if not already mapped
    if ("calibration_slot_col" not in locals()) or (not calibration_slot_col) or (calibration_slot_col not in df_f.columns):
        calibration_slot_col = find_col(df_f, [
            "Calibration Slot (Deal)", "Calibration Slot", "Book Slot", "Trial Slot"
        ])

    # ==== Scope controls
    scope_mode = st.radio(
        "Scope",
        ["Month", "Trailing days"],
        horizontal=True,
        index=0,
        help="Month = a single calendar month. Trailing days = rolling window ending today."
    )

    if scope_mode == "Month":
        # Build month list from whatever date columns exist
        candidates = []
        if create_col:
            candidates.append(coerce_datetime(df_f[create_col]))
        if first_cal_sched_col and first_cal_sched_col in df_f.columns:
            candidates.append(coerce_datetime(df_f[first_cal_sched_col]))
        if cal_resched_col and cal_resched_col in df_f.columns:
            candidates.append(coerce_datetime(df_f[cal_resched_col]))
        if cal_done_col and cal_done_col in df_f.columns:
            candidates.append(coerce_datetime(df_f[cal_done_col]))
        if pay_col:
            candidates.append(coerce_datetime(df_f[pay_col]))

        if candidates:
            all_months = (
                pd.to_datetime(pd.concat(candidates, ignore_index=True))
                  .dropna()
                  .dt.to_period("M")
                  .sort_values()
                  .unique()
                  .astype(str)
                  .tolist()
            )
        else:
            all_months = []

        # Ensure at least the running month is present
        if not all_months:
            all_months = [str(pd.Period(date.today(), freq="M"))]

        # Preselect running month if present; else fallback to last available month
        running_period = str(pd.Period(date.today(), freq="M"))
        default_idx = all_months.index(running_period) if running_period in all_months else len(all_months) - 1

        sel_month = st.selectbox("Select month (YYYY-MM)", options=all_months, index=default_idx)
        yy, mm = map(int, sel_month.split("-"))
        range_start, range_end = month_bounds(date(yy, mm, 1))
        st.caption(f"Scope: **{range_start} → {range_end}**")
    else:
        trailing = st.slider("Trailing window (days)", min_value=7, max_value=60, value=15, step=1)
        range_end = date.today()
        range_start = range_end - timedelta(days=trailing - 1)
        st.caption(f"Scope: **{range_start} → {range_end}** (last {trailing} days)")

    # ==== Prepare normalized datetime columns from FILTERED data
    d = df_f.copy()
    d["_c"]  = coerce_datetime(d[create_col]) if create_col else pd.Series(pd.NaT, index=d.index)
    d["_f"]  = coerce_datetime(d[first_cal_sched_col]) if first_cal_sched_col and first_cal_sched_col in d.columns else pd.Series(pd.NaT, index=d.index)
    d["_r"]  = coerce_datetime(d[cal_resched_col])     if cal_resched_col and cal_resched_col in d.columns     else pd.Series(pd.NaT, index=d.index)
    d["_fd"] = coerce_datetime(d[cal_done_col])        if cal_done_col and cal_done_col in d.columns          else pd.Series(pd.NaT, index=d.index)
    d["_p"]  = coerce_datetime(d[pay_col]) if pay_col else pd.Series(pd.NaT, index=d.index)

    # Effective trial date = min(First Cal, Rescheduled), NaT-safe
    d["_trial"] = d[["_f", "_r"]].min(axis=1, skipna=True)

    # ==== Filter: Booking type (Pre-Book vs Self-Book) based on Trial + Slot
    # Rule:
    #   Pre-Book  = has a Trial date AND Calibration Slot (Deal) is non-empty
    #   Self-Book = everything else (no trial OR empty slot)
    if calibration_slot_col and calibration_slot_col in d.columns:
        slot_series = d[calibration_slot_col].astype(str)
        _s = slot_series.str.strip().str.lower()
        has_slot = _s.ne("") & _s.ne("nan") & _s.ne("none")

        is_prebook = d["_trial"].notna() & has_slot
        d["_booking_type"] = np.where(is_prebook, "Pre-Book", "Self-Book")

        booking_choice = st.radio(
            "Booking type",
            options=["All", "Pre-Book", "Self-Book"],
            index=0,
            horizontal=True,
            help="Pre-Book = Trial present AND slot filled. Self-Book = otherwise."
        )
        if booking_choice != "All":
            d = d[d["_booking_type"] == booking_choice].copy()
            st.caption(f"Booking type filter: **{booking_choice}** • Rows now: **{len(d):,}**")
    else:
        st.info("Calibration Slot (Deal) column not found — booking type filter not applied.")

    # NOTE: Inactivity seek bars have been removed as requested. No inactivity filtering is applied.

    # ==== Cohort: deals CREATED within scope
    mask_created = d["_c"].dt.date.between(range_start, range_end)
    cohort = d.loc[mask_created].copy()
    total_created = int(len(cohort))

    # Stage 2: Trial in SAME scope & same cohort
    trial_mask = cohort["_trial"].dt.date.between(range_start, range_end)
    trial_df = cohort.loc[trial_mask].copy()
    total_trial = int(len(trial_df))

    # Stage 3: Cal Done in SAME scope from those that had Trial in scope
    caldone_mask = trial_df["_fd"].dt.date.between(range_start, range_end)
    caldone_df = trial_df.loc[caldone_mask].copy()
    total_caldone = int(len(caldone_df))

    # Stage 4: Payment in SAME scope from those that had Cal Done in scope
    pay_mask = caldone_df["_p"].dt.date.between(range_start, range_end)
    pay_df = caldone_df.loc[pay_mask].copy()
    total_pay = int(len(pay_df))

    # ==== Funnel summary (always include Payment stage now)
    funnel_rows = [
        {"Stage": "Created (T)",            "Count": total_created, "FromPrev_pct": 100.0},
        {"Stage": "Trial (First/Resched)",  "Count": total_trial,   "FromPrev_pct": (total_trial / total_created * 100.0) if total_created > 0 else 0.0},
        {"Stage": "Calibration Done",       "Count": total_caldone, "FromPrev_pct": (total_caldone / total_trial * 100.0) if total_trial > 0 else 0.0},
        {"Stage": "Payment Received",       "Count": total_pay,     "FromPrev_pct": (total_pay / total_caldone * 100.0) if total_caldone > 0 else 0.0},
    ]
    funnel_df = pd.DataFrame(funnel_rows)

    # Always show something (even when all zeros)
    bar = alt.Chart(funnel_df).mark_bar(opacity=0.9).encode(
        x=alt.X("Count:Q", title="Count"),
        y=alt.Y("Stage:N", sort=list(funnel_df["Stage"])[::-1], title=""),
        tooltip=[
            alt.Tooltip("Stage:N"),
            alt.Tooltip("Count:Q"),
            alt.Tooltip("FromPrev_pct:Q", title="% from previous", format=".1f"),
        ],
        color=alt.Color("Stage:N", legend=None),
    ).properties(height=240, title="Funnel (same cohort within scope)")
    txt = alt.Chart(funnel_df).mark_text(align="left", dx=5).encode(
        x="Count:Q",
        y=alt.Y("Stage:N", sort=list(funnel_df["Stage"])[::-1]),
        text=alt.Text("Count:Q"),
    )
    st.altair_chart(bar + txt, use_container_width=True)

    # Quick debug line so you can see data even if bars look empty
    st.caption(
        f"Created: {total_created} • Trial: {total_trial} • Cal Done: {total_caldone} • Payments: {total_pay}"
    )

    # ==== Propagation (average days) – computed only on the same filtered sets
    def avg_days(src_series, dst_series) -> float:
        s = (dst_series - src_series).dt.days
        s = s.dropna()
        return float(s.mean()) if len(s) else np.nan

    avg_ct = avg_days(trial_df["_c"], trial_df["_trial"]) if not trial_df.empty else np.nan
    avg_tc = avg_days(caldone_df["_trial"], caldone_df["_fd"]) if not caldone_df.empty else np.nan
    avg_dp = avg_days(pay_df["_fd"], pay_df["_p"]) if not pay_df.empty else np.nan

    def fmtd(x): return "–" if pd.isna(x) else f"{x:.1f} days"
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Created → Trial</div><div class='kpi-value'>{fmtd(avg_ct)}</div></div>",
            unsafe_allow_html=True
        )
    with g2:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Trial → Cal Done</div><div class='kpi-value'>{fmtd(avg_tc)}</div></div>",
            unsafe_allow_html=True
        )
    with g3:
        st.markdown(
            f"<div class='kpi-card'><div class='kpi-title'>Cal Done → Payment</div><div class='kpi-value'>{fmtd(avg_dp)}</div></div>",
            unsafe_allow_html=True
        )

    # ==== Month-on-Month comparison
    st.markdown("### Month-on-Month comparison")
    compare_k = st.slider("Compare last N months (ending at selected month or current)", 2, 12, 6, step=1)

    # Decide anchor month
    anchor_day = range_end if scope_mode == "Month" else date.today()
    months = months_back_list(anchor_day, compare_k)  # returns list of monthly Periods

    def month_funnel(m_period: pd.Period):
        ms = date(m_period.year, m_period.month, 1)
        me = date(m_period.year, m_period.month, monthrange(m_period.year, m_period.month)[1])

        coh = d[d["_c"].dt.date.between(ms, me)].copy()
        ct = int(len(coh))

        tr_mask = coh["_trial"].dt.date.between(ms, me)
        coh_tr = coh.loc[tr_mask].copy()
        tr = int(len(coh_tr))

        cd_mask = coh_tr["_fd"].dt.date.between(ms, me)
        coh_cd = coh_tr.loc[cd_mask].copy()
        cd = int(len(coh_cd))

        py = int(coh_cd["_p"].dt.date.between(ms, me).sum())

        # propagation avgs
        avg1 = avg_days(coh_tr["_c"], coh_tr["_trial"]) if not coh_tr.empty else np.nan
        avg2 = avg_days(coh_cd["_trial"], coh_cd["_fd"]) if not coh_cd.empty else np.nan
        avg3 = avg_days(coh_cd["_fd"], coh_cd["_p"]) if not coh_cd.empty else np.nan

        return {
            "Month": str(m_period),
            "Created": ct,
            "Trial": tr,
            "CalDone": cd,
            "Paid": py,
            "Trial_from_Created_pct": (tr / ct * 100.0) if ct > 0 else 0.0,
            "CalDone_from_Trial_pct": (cd / tr * 100.0) if tr > 0 else 0.0,
            "Paid_from_CalDone_pct": (py / cd * 100.0) if cd > 0 else 0.0,
            "Avg_Created_to_Trial_days": avg1,
            "Avg_Trial_to_CalDone_days": avg2,
            "Avg_CalDone_to_Payment_days": avg3,
        }

    mom_tbl = pd.DataFrame([month_funnel(m) for m in months])

    if mom_tbl.empty:
        st.info("Not enough historical data to build month-on-month comparison.")
    else:
        # Conversion step lines
        conv_long = mom_tbl.melt(
            id_vars=["Month"],
            value_vars=["Trial_from_Created_pct", "CalDone_from_Trial_pct", "Paid_from_CalDone_pct"],
            var_name="Step",
            value_name="Pct",
        )
        conv_chart = alt.Chart(conv_long).mark_line(point=True).encode(
            x=alt.X("Month:N", sort=mom_tbl["Month"].tolist()),
            y=alt.Y("Pct:Q", title="Step conversion %", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Step:N", title="Step"),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Step:N"), alt.Tooltip("Pct:Q", format=".1f")],
        ).properties(height=320, title="Step conversion% (MoM)")
        st.altair_chart(conv_chart, use_container_width=True)

        # Propagation lines
        lag_long = mom_tbl.melt(
            id_vars=["Month"],
            value_vars=["Avg_Created_to_Trial_days", "Avg_Trial_to_CalDone_days", "Avg_CalDone_to_Payment_days"],
            var_name="Lag",
            value_name="Days",
        )
        lag_chart = alt.Chart(lag_long).mark_line(point=True).encode(
            x=alt.X("Month:N", sort=mom_tbl["Month"].tolist()),
            y=alt.Y("Days:Q", title="Avg days"),
            color=alt.Color("Lag:N", title="Propagation"),
            tooltip=[alt.Tooltip("Month:N"), alt.Tooltip("Lag:N"), alt.Tooltip("Days:Q", format=".1f")],
        ).properties(height=320, title="Average propagation (MoM)")
        st.altair_chart(lag_chart, use_container_width=True)

        with st.expander("Month-on-Month table"):
            st.dataframe(mom_tbl, use_container_width=True)
            st.download_button(
                "Download CSV – MoM Funnel & Propagation",
                data=mom_tbl.to_csv(index=False).encode("utf-8"),
                file_name="stuck_deals_mom_funnel_propagation.csv",
                mime="text/csv",
            )


elif view == "Lead Movement":
    st.subheader("Lead Movement — inactivity by Last Connected / Lead Activity (Create-date scoped)")

    # ---- Column mapping
    lead_activity_col = find_col(df, [
        "Lead Activity Date", "Lead activity date", "Last Activity Date", "Last Activity"
    ])
    last_connected_col = find_col(df, [
        "Last Connected", "Last connected", "Last Contacted", "Last Contacted Date"
    ])

    if not create_col:
        st.error("Create Date column not found — this tab scopes the population by Create Date.")
        st.stop()

    # ---- Optional Deal Stage filter (applies to population)
    d = df_f.copy()
    if dealstage_col and dealstage_col in d.columns:
        stage_vals = ["All"] + sorted(d[dealstage_col].dropna().astype(str).unique().tolist())
        sel_stages = st.multiselect(
            "Deal Stage (optional filter on population)",
            stage_vals, default=["All"], key="lm_stage_filter"
        )
        if "All" not in sel_stages:
            d = d[d[dealstage_col].astype(str).isin(sel_stages)].copy()
    else:
        st.caption("Deal Stage column not found — stage filter disabled.")

    if d.empty:
        st.info("No rows after filters.")
        st.stop()

    # ---- Date scope (population by Create Date)
    st.markdown("**Date scope (based on Create Date)**")
    c1, c2 = st.columns(2)
    scope_pick = st.radio(
        "Presets",
        ["Yesterday", "Today", "This month", "Last month", "Custom"],
        index=2, horizontal=True, key="lm_scope"
    )
    if scope_pick == "Yesterday":
        scope_start, scope_end = yday, yday
    elif scope_pick == "Today":
        scope_start, scope_end = today, today
    elif scope_pick == "This month":
        scope_start, scope_end = month_bounds(today)
    elif scope_pick == "Last month":
        scope_start, scope_end = last_month_bounds(today)
    else:
        with c1:
            scope_start = st.date_input("Start (Create Date)", value=today.replace(day=1), key="lm_cstart")
        with c2:
            scope_end = st.date_input("End (Create Date)", value=month_bounds(today)[1], key="lm_cend")
        if scope_end < scope_start:
            st.error("End date cannot be before start date.")
            st.stop()

    st.caption(f"Create-date scope: **{scope_start} → {scope_end}**")

    # ---- Choose reference date for inactivity
    ref_pick = st.radio(
        "Reference date (for inactivity days)",
        ["Last Connected", "Lead Activity Date"],
        index=0, horizontal=True, key="lm_ref_pick"
    )
    if ref_pick == "Last Connected":
        ref_col = last_connected_col if (last_connected_col and last_connected_col in d.columns) else None
    else:
        ref_col = lead_activity_col if (lead_activity_col and lead_activity_col in d.columns) else None

    if not ref_col:
        st.warning(f"Selected reference column for '{ref_pick}' not found in data.")
        st.stop()

    # ---- Build in-scope dataset (population by Create Date)
    d["_cdate"] = coerce_datetime(d[create_col]).dt.date
    pop_mask = d["_cdate"].between(scope_start, scope_end)
    d_work = d.loc[pop_mask].copy()

    # Compute inactivity days from chosen reference column
    d_work["_ref_dt"] = coerce_datetime(d_work[ref_col])
    d_work["_days_since"] = (pd.Timestamp(today) - d_work["_ref_dt"]).dt.days  # NaT-safe diff

    # ---- Slider (inactivity range)
    valid_days = d_work["_days_since"].dropna()
    if valid_days.empty:
        min_d, max_d = 0, 90
    else:
        min_d, max_d = int(valid_days.min()), int(valid_days.max())
        min_d = min(0, min_d)
        max_d = max(1, max_d)
    days_low, days_high = st.slider(
        "Inactivity range (days)",
        min_value=int(min_d), max_value=int(max_d),
        value=(min(7, max(0, min_d)), min(30, max_d)),
        step=1, key="lm_range"
    )
    range_mask = d_work["_days_since"].between(days_low, days_high)

    # ---- Bucketize inactivity for stacked charts
    def bucketize(n):
        if pd.isna(n):
            return "Unknown"
        n = int(n)
        if n <= 1:   return "0–1"
        if n <= 3:   return "2–3"
        if n <= 7:   return "4–7"
        if n <= 14:  return "8–14"
        if n <= 30:  return "15–30"
        if n <= 60:  return "31–60"
        if n <= 90:  return "61–90"
        return "90+"

    d_work["Bucket"] = d_work["_days_since"].apply(bucketize)
    bucket_order = ["0–1","2–3","4–7","8–14","15–30","31–60","61–90","90+","Unknown"]

    # ---- Stacked by Deal Source
    st.markdown("### Inactivity distribution — stacked by JetLearn Deal Source")
    if source_col and source_col in d_work.columns:
        d_work["_source"] = d_work[source_col].fillna("Unknown").astype(str)
        by_src = (
            d_work.groupby(["Bucket","_source"])
                  .size().reset_index(name="Count")
        )
        chart_src = (
            alt.Chart(by_src)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", stack=True, title="Count"),
                color=alt.Color("_source:N", title="Deal Source", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("_source:N", title="Deal Source"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Inactivity by {ref_pick} — stacked by Deal Source")
        )
        st.altair_chart(chart_src, use_container_width=True)
    else:
        st.info("Deal Source column not found — skipping source-wise stack.")

    # ---- Stacked by Country (Top-5 toggle)
    st.markdown("### Inactivity distribution — stacked by Country")
    if country_col and country_col in d_work.columns:
        d_work["_country"] = d_work[country_col].fillna("Unknown").astype(str)
        totals_country = d_work.groupby("_country").size().sort_values(ascending=False)
        show_all_countries = st.checkbox(
            "Show all countries (uncheck to show Top 5 only)",
            value=False, key="lm_show_all_cty"
        )
        if show_all_countries:
            keep_countries = totals_country.index.tolist()
            title_suffix = "All countries"
        else:
            keep_countries = totals_country.head(5).index.tolist()
            title_suffix = "Top 5 countries"

        by_cty = (
            d_work[d_work["_country"].isin(keep_countries)]
            .groupby(["Bucket","_country"]).size().reset_index(name="Count")
        )
        chart_cty = (
            alt.Chart(by_cty)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", stack=True, title="Count"),
                color=alt.Color("_country:N", title="Country", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("_country:N", title="Country"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Inactivity by {ref_pick} — stacked by Country ({title_suffix})")
        )
        st.altair_chart(chart_cty, use_container_width=True)
    else:
        st.info("Country column not found — skipping country-wise stack.")

    # ---- Deal Stage detail for selected inactivity range
    st.markdown("### Deal Stage detail — for selected inactivity range")
    if dealstage_col and dealstage_col in d_work.columns:
        stage_counts = (
            d_work.loc[range_mask, dealstage_col]
                  .fillna("Unknown").astype(str)
                  .value_counts().reset_index()
        )
        stage_counts.columns = ["Deal Stage", "Count"]
        st.dataframe(stage_counts, use_container_width=True)
        st.download_button(
            "Download CSV — Deal Stage counts (selected inactivity range)",
            data=stage_counts.to_csv(index=False).encode("utf-8"),
            file_name="lead_movement_dealstage_counts.csv",
            mime="text/csv",
            key="lm_stage_dl"
        )

        with st.expander("Show matching rows (first 1000)"):
            cols_show = []
            for c in [create_col, dealstage_col, ref_col, country_col, source_col]:
                if c and c in d_work.columns:
                    cols_show.append(c)
            preview = d_work.loc[range_mask, cols_show].head(1000)
            st.dataframe(preview, use_container_width=True)
            st.download_button(
                "Download CSV — Matching rows",
                data=d_work.loc[range_mask, cols_show].to_csv(index=False).encode("utf-8"),
                file_name="lead_movement_matching_rows.csv",
                mime="text/csv",
                key="lm_rows_dl"
            )
    else:
        st.info("Deal Stage column not found — cannot show stage detail.")

    # ---- Quick KPIs
    total_in_scope = int(len(d_work))
    missing_ref = int(d_work["_ref_dt"].isna().sum())
    selected_cnt = int(range_mask.sum())
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-title'>In-scope leads (Create Date {scope_start} → {scope_end})</div>"
        f"<div class='kpi-value'>{total_in_scope:,}</div>"
        f"<div class='kpi-sub'>Missing {ref_pick}: {missing_ref:,} • In range {days_low}–{days_high} days: {selected_cnt:,}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    # =====================================================================
    #        📊 Inactivity distribution — Deal Owner / Academic Counselor
    # =====================================================================
    st.markdown("---")
    st.markdown("### 📊 Inactivity distribution — Deal Owner (Academic Counselor)")

    # Detect both fields separately
    deal_owner_raw = find_col(df, ["Deal Owner", "Owner"])
    acad_couns_raw = find_col(df, [
        "Student/Academic Counselor", "Student/Academic Counsellor",
        "Academic Counselor", "Academic Counsellor",
        "Counselor", "Counsellor"
    ])

    # Owner field selection: choose one or combine
    owner_mode = st.selectbox(
        "Owner dimension for analysis",
        [
            "Deal Owner",
            "Student/Academic Counselor",
            "Combine (Deal Owner → Student/Academic Counselor)",
            "Combine (Student/Academic Counselor → Deal Owner)",
        ],
        index=0,
        key="lm_owner_mode"
    )

    # Validate availability
    def _series_or_none(colname):
        return d_work[colname] if (colname and colname in d_work.columns) else None

    s_owner = _series_or_none(deal_owner_raw)
    s_acad  = _series_or_none(acad_couns_raw)

    if owner_mode == "Deal Owner" and s_owner is None:
        st.info("‘Deal Owner’ column not found in the current dataset.")
        st.stop()
    if owner_mode == "Student/Academic Counselor" and s_acad is None:
        st.info("‘Student/Academic Counselor’ column not found in the current dataset.")
        st.stop()
    if "Combine" in owner_mode and (s_owner is None and s_acad is None):
        st.info("Neither ‘Deal Owner’ nor ‘Student/Academic Counselor’ columns are present.")
        st.stop()

    # Build the owner dimension
    if owner_mode == "Deal Owner":
        d_work["_owner"] = s_owner.fillna("Unknown").replace("", "Unknown").astype(str)
    elif owner_mode == "Student/Academic Counselor":
        d_work["_owner"] = s_acad.fillna("Unknown").replace("", "Unknown").astype(str)
    elif owner_mode == "Combine (Deal Owner → Student/Academic Counselor)":
        # Prefer Deal Owner, fallback to Academic Counselor
        d_work["_owner"] = (
            (s_owner.fillna("").astype(str))
            .mask(lambda x: x.str.strip().eq("") & s_acad.notna(), s_acad.astype(str))
            .replace("", "Unknown")
            .fillna("Unknown")
            .astype(str)
        )
    else:  # Combine (Student/Academic Counselor → Deal Owner)
        d_work["_owner"] = (
            (s_acad.fillna("").astype(str))
            .mask(lambda x: x.str.strip().eq("") & (s_owner.notna()), s_owner.astype(str))
            .replace("", "Unknown")
            .fillna("Unknown")
            .astype(str)
        )

    # Controls: Aggregate vs Split + Top-N owners
    col_oview, col_topn = st.columns([1.2, 1])
    with col_oview:
        owner_view = st.radio(
            "View mode",
            ["Aggregate (overall)", "Split by Academic Counselor"],
            index=1, horizontal=False, key="lm_owner_view"
        )
    with col_topn:
        owner_counts_all = d_work["_owner"].value_counts()
        max_top = min(30, max(5, len(owner_counts_all)))
        top_n = st.number_input("Top N owners for charts", min_value=5, max_value=max_top, value=min(12, max_top), step=1, key="lm_owner_topn")

    # Limit to Top-N for readability
    top_owners = owner_counts_all.head(int(top_n)).index.tolist()
    d_top = d_work[d_work["_owner"].isin(top_owners)].copy()

    # Aggregate mode: bucket totals overall (no owner split)
    if owner_view == "Aggregate (overall)":
        agg_bucket = (
            d_top.groupby("Bucket")
                 .size().reset_index(name="Count")
        )
        chart_owner_agg = (
            alt.Chart(agg_bucket)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", title="Count"),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("Count:Q")]
            )
            .properties(height=320, title=f"Inactivity by {ref_pick} — Aggregate (Top {len(top_owners)} owners)")
        )
        st.altair_chart(chart_owner_agg, use_container_width=True)

    else:
        # Split mode: stacked by owner across buckets (Bucket on x, colors = owner)
        by_owner_bucket = (
            d_top.groupby(["Bucket", "_owner"])
                 .size().reset_index(name="Count")
        )
        chart_owner_split = (
            alt.Chart(by_owner_bucket)
            .mark_bar(opacity=0.9)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
                y=alt.Y("Count:Q", stack=True, title="Count"),
                color=alt.Color("_owner:N", title="Academic Counselor", legend=alt.Legend(orient="bottom")),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("_owner:N", title="Academic Counselor"), alt.Tooltip("Count:Q")]
            )
            .properties(height=360, title=f"Inactivity by {ref_pick} — stacked by Academic Counselor (Top {len(top_owners)})")
        )
        st.altair_chart(chart_owner_split, use_container_width=True)

    # ================= Option: Exclude Unknown on Owner-on-X chart =================
    st.markdown("#### Inactivity distribution — stacked by Bucket (Owner on X-axis)")
    exclude_unknown_owner = st.checkbox(
        "Exclude ‘Unknown’ owners from this chart",
        value=True,
        key="lm_owner_exclude_unknown_xaxis"
    )

    owner_x_df = d_top.copy()
    if exclude_unknown_owner:
        owner_x_df = owner_x_df[owner_x_df["_owner"] != "Unknown"]

    owner_x_bucket = (
        owner_x_df.groupby(["_owner", "Bucket"])
                  .size().reset_index(name="Count")
    )

    chart_owner_x = (
        alt.Chart(owner_x_bucket)
        .mark_bar(opacity=0.9)
        .encode(
            x=alt.X("_owner:N", title="Academic Counselor", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Count:Q", stack=True, title="Count"),
            color=alt.Color("Bucket:N", sort=bucket_order, title="Inactivity bucket (days)"),
            tooltip=[alt.Tooltip("_owner:N", title="Academic Counselor"),
                     alt.Tooltip("Bucket:N", title="Bucket"),
                     alt.Tooltip("Count:Q")]
        )
        .properties(height=380, title=f"Inactivity by {ref_pick} — Academic Counselor on X-axis (Top {len(top_owners)})")
    )
    st.altair_chart(chart_owner_x, use_container_width=True)

    # Owner table for currently selected inactivity range (actionable)
    st.markdown("#### Owners in selected inactivity range")
    owner_range = (
        d_work.loc[range_mask, "_owner"]
             .fillna("Unknown").astype(str)
             .value_counts().reset_index()
    )
    owner_range.columns = ["Academic Counselor", "Count"]
    owner_range["Share %"] = (owner_range["Count"] / max(int(range_mask.sum()), 1) * 100).round(1)
    st.dataframe(owner_range, use_container_width=True)

    st.download_button(
        "Download CSV — Owners (selected inactivity range)",
        data=owner_range.to_csv(index=False).encode("utf-8"),
        file_name="lead_movement_owners_selected_range.csv",
        mime="text/csv",
        key="lm_owner_dl"
    )

    with st.expander("Show matching rows by owner (first 1000)"):
        cols_show_owner = []
        for c in [create_col, ref_col, dealstage_col, country_col, source_col, deal_owner_raw, acad_couns_raw]:
            if c and c in d_work.columns:
                cols_show_owner.append(c)
        preview_owner = d_work.loc[range_mask, cols_show_owner].head(1000)
        st.dataframe(preview_owner, use_container_width=True)



elif view == "AC Wise Detail":
    st.subheader("AC Wise Detail – Create-date scoped counts & % conversions")

    # ---- Required cols & special columns
    referral_intent_col = find_col(df, ["Referral Intent Source", "Referral intent source"])
    if not create_col or not counsellor_col:
        st.error("Missing required columns (Create Date and Academic Counsellor).")
        st.stop()

    # ---- Date scope (population by Create Date) & Counting mode
    st.markdown("**Date scope (based on Create Date) & Counting mode**")
    c1, c2 = st.columns(2)
    scope_pick = st.radio(
        "Presets",
        ["Yesterday", "Today", "This month", "Last month", "Custom"],
        index=2, horizontal=True, key="ac_scope"
    )
    if scope_pick == "Yesterday":
        scope_start, scope_end = yday, yday
    elif scope_pick == "Today":
        scope_start, scope_end = today, today
    elif scope_pick == "This month":
        scope_start, scope_end = month_bounds(today)
    elif scope_pick == "Last month":
        scope_start, scope_end = last_month_bounds(today)
    else:
        with c1:
            scope_start = st.date_input("Start (Create Date)", value=today.replace(day=1), key="ac_cstart")
        with c2:
            scope_end = st.date_input("End (Create Date)", value=month_bounds(today)[1], key="ac_cend")
        if scope_end < scope_start:
            st.error("End date cannot be before start date.")
            st.stop()

    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ac_mode")
    st.caption(f"Create-date scope: **{scope_start} → {scope_end}** • Mode: **{mode}**")

    # ---- Start from globally filtered df_f, optional Deal Stage filter
    d = df_f.copy()

    if dealstage_col and dealstage_col in d.columns:
        stage_vals = ["All"] + sorted(d[dealstage_col].dropna().astype(str).unique().tolist())
        sel_stages = st.multiselect(
            "Deal Stage (optional filter on population)",
            stage_vals, default=["All"], key="ac_stage"
        )
        if "All" not in sel_stages:
            d = d[d[dealstage_col].astype(str).isin(sel_stages)].copy()
    else:
        st.caption("Deal Stage column not found — stage filter disabled.")

    if d.empty:
        st.info("No rows after filters.")
        st.stop()

    # ---- Normalize helper columns
    d["_ac"] = d[counsellor_col].fillna("Unknown").astype(str)

    _cdate = coerce_datetime(d[create_col]).dt.date
    _first = coerce_datetime(d[first_cal_sched_col]).dt.date if first_cal_sched_col and first_cal_sched_col in d.columns else pd.Series(pd.NaT, index=d.index)
    _resch = coerce_datetime(d[cal_resched_col]).dt.date     if cal_resched_col     and cal_resched_col     in d.columns else pd.Series(pd.NaT, index=d.index)
    _done  = coerce_datetime(d[cal_done_col]).dt.date        if cal_done_col        and cal_done_col        in d.columns else pd.Series(pd.NaT, index=d.index)
    _paid  = coerce_datetime(d[pay_col]).dt.date             if pay_col             and pay_col             in d.columns else pd.Series(pd.NaT, index=d.index)

    # Masks
    pop_mask = _cdate.between(scope_start, scope_end)  # population by Create Date
    m_first = _first.between(scope_start, scope_end) if _first.notna().any() else pd.Series(False, index=d.index)
    m_resch = _resch.between(scope_start, scope_end) if _resch.notna().any() else pd.Series(False, index=d.index)
    m_done  = _done.between(scope_start, scope_end)  if _done.notna().any()  else pd.Series(False, index=d.index)
    m_paid  = _paid.between(scope_start, scope_end)  if _paid.notna().any()  else pd.Series(False, index=d.index)

    # Apply mode to event indicators
    if mode == "MTD":
        ind_create = pop_mask
        ind_first  = pop_mask & m_first
        ind_resch  = pop_mask & m_resch
        ind_done   = pop_mask & m_done
        ind_paid   = pop_mask & m_paid
    else:  # Cohort
        ind_create = pop_mask
        ind_first  = m_first
        ind_resch  = m_resch
        ind_done   = m_done
        ind_paid   = m_paid

    # ---------- Referral Intent Source = "Sales Generated" only ----------
    if referral_intent_col and referral_intent_col in d.columns:
        _ref = d[referral_intent_col].astype(str).str.strip().str.lower()
        sales_generated_mask = (_ref == "sales generated")
    else:
        sales_generated_mask = pd.Series(False, index=d.index)
    ind_ref_sales = pop_mask & sales_generated_mask

    # ---------- Aggregate toggle (All Academic Counsellors) ----------
    st.markdown("#### Display mode")
    show_all_ac = st.checkbox("Aggregate all Academic Counsellors (show totals only)", value=False, key="ac_all_toggle")

    # ---- AC-wise table
    col_label_ref = "Referral Intent Source = Sales Generated — Count"

    base_sub = pd.DataFrame({
        "Academic Counsellor": d["_ac"],
        "Create Date — Count": ind_create.astype(int),
        "First Cal — Count": ind_first.astype(int),
        "Cal Rescheduled — Count": ind_resch.astype(int),
        "Cal Done — Count": ind_done.astype(int),
        "Payment Received — Count": ind_paid.astype(int),
        col_label_ref:           ind_ref_sales.astype(int),
    })

    if show_all_ac:
        agg = (
            base_sub.drop(columns=["Academic Counsellor"])
                    .sum(numeric_only=True)
                    .to_frame().T
        )
        agg.insert(0, "Academic Counsellor", "All ACs (Total)")
    else:
        agg = (
            base_sub.groupby("Academic Counsellor", as_index=False)
                    .sum(numeric_only=True)
                    .sort_values("Create Date — Count", ascending=False)
        )

    st.markdown("### AC-wise counts")
    st.dataframe(agg, use_container_width=True)
    st.download_button(
        "Download CSV — AC-wise counts",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name=f"ac_wise_counts_{'all' if show_all_ac else 'by_ac'}_{mode.lower()}.csv",
        mime="text/csv",
        key="ac_dl_counts"
    )

    # ---- % Conversion between two chosen metrics
    st.markdown("### Conversion % between two metrics")
    metric_labels = [
        "Create Date — Count",
        "First Cal — Count",
        "Cal Rescheduled — Count",
        "Cal Done — Count",
        "Payment Received — Count",
        col_label_ref,
    ]
    c3, c4 = st.columns(2)
    with c3:
        denom_label = st.selectbox("Denominator", metric_labels, index=0, key="ac_pct_denom")
    with c4:
        numer_label = st.selectbox("Numerator",  metric_labels, index=3, key="ac_pct_numer")

    pct_tbl = agg[["Academic Counsellor", denom_label, numer_label]].copy()
    pct_tbl["%"] = np.where(
        pct_tbl[denom_label] > 0,
        (pct_tbl[numer_label] / pct_tbl[denom_label]) * 100.0,
        0.0
    ).round(1)
    pct_tbl = pct_tbl.sort_values("%", ascending=False) if not show_all_ac else pct_tbl

    st.dataframe(pct_tbl, use_container_width=True)
    st.download_button(
        "Download CSV — Conversion %",
        data=pct_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"ac_conversion_percent_{'all' if show_all_ac else 'by_ac'}_{mode.lower()}.csv",
        mime="text/csv",
        key="ac_dl_pct"
    )

    # Overall KPI
    den_sum = int(pct_tbl[denom_label].sum())
    num_sum = int(pct_tbl[numer_label].sum())
    overall_pct = (num_sum / den_sum * 100.0) if den_sum > 0 else 0.0
    st.markdown(
        f"<div class='kpi-card'><div class='kpi-title'>Overall {numer_label} / {denom_label} ({mode})</div>"
        f"<div class='kpi-value'>{overall_pct:.1f}%</div>"
        f"<div class='kpi-sub'>Num: {num_sum:,} • Den: {den_sum:,}</div></div>",
        unsafe_allow_html=True
    )

    # ---- Breakdown: AC × (Deal Source or Country)
    st.markdown("### Breakdown")
    grp_mode = st.radio("Group by", ["JetLearn Deal Source", "Country"], index=0, horizontal=True, key="ac_grp_mode")

    have_grp = False
    if grp_mode == "JetLearn Deal Source":
        if not source_col or source_col not in d.columns:
            st.info("Deal Source column not found.")
        else:
            d["_grp"] = d[source_col].fillna("Unknown").astype(str)
            have_grp = True
    else:
        if not country_col or country_col not in d.columns:
            st.info("Country column not found.")
        else:
            d["_grp"] = d[country_col].fillna("Unknown").astype(str)
            have_grp = True

    if have_grp:
        sub2 = pd.DataFrame({
            "Academic Counsellor": d["_ac"],
            "_grp": d["_grp"],
            "Create Date — Count": ind_create.astype(int),
            "First Cal — Count": ind_first.astype(int),
            "Cal Rescheduled — Count": ind_resch.astype(int),
            "Cal Done — Count": ind_done.astype(int),
            "Payment Received — Count": ind_paid.astype(int),
            col_label_ref:           ind_ref_sales.astype(int),
        })

        if show_all_ac:
            gb = (
                sub2.drop(columns=["Academic Counsellor"])
                    .groupby("_grp", as_index=False)
                    .sum(numeric_only=True)
                    .rename(columns={"_grp": grp_mode})
                    .sort_values("Create Date — Count", ascending=False)
            )
        else:
            gb = (
                sub2.groupby(["Academic Counsellor","_grp"], as_index=False)
                    .sum(numeric_only=True)
                    .rename(columns={"_grp": grp_mode})
                    .sort_values(["Academic Counsellor","Create Date — Count"], ascending=[True, False])
            )

        st.dataframe(gb, use_container_width=True)
        st.download_button(
            f"Download CSV — {'Totals × ' if show_all_ac else 'AC × '}{grp_mode} breakdown ({mode})",
            data=gb.to_csv(index=False).encode("utf-8"),
            file_name=f"{'totals' if show_all_ac else 'ac'}_breakdown_by_{'deal_source' if grp_mode.startswith('JetLearn') else 'country'}_{mode.lower()}.csv",
            mime="text/csv",
            key="ac_dl_breakdown"
        )

    # ==== AC × Deal Source — Stacked charts (Payments, Deals Created, and Conversion%) ====
    st.markdown("### AC × Deal Source — Stacked charts (Payments, Deals Created & Conversion %)")

    if (not source_col) or (source_col not in d.columns):
        st.info("Deal Source column not found — cannot draw stacked charts.")
    else:
        _idx = d.index
        ac_series  = (pd.Series("All ACs (Total)", index=_idx) if show_all_ac else d["_ac"])
        src_series = d[source_col].fillna("Unknown").astype(str)

        ind_paid_series   = pd.Series(ind_paid, index=_idx).astype(bool)
        ind_create_series = pd.Series(ind_create, index=_idx).astype(bool)

        # Payments stacked
        df_pay = pd.DataFrame({
            "Academic Counsellor": ac_series,
            "Deal Source": src_series,
            "Count": ind_paid_series.astype(int)
        })
        g_pay = df_pay.groupby(["Academic Counsellor", "Deal Source"], as_index=False)["Count"].sum()
        totals_pay = g_pay.groupby("Academic Counsellor", as_index=False)["Count"].sum().rename(columns={"Count": "Total"})

        # Deals Created stacked
        df_create = pd.DataFrame({
            "Academic Counsellor": ac_series,
            "Deal Source": src_series,
            "Count": ind_create_series.astype(int)
        })
        g_create = df_create.groupby(["Academic Counsellor", "Deal Source"], as_index=False)["Count"].sum()
        totals_create = g_create.groupby("Academic Counsellor", as_index=False)["Count"].sum().rename(columns={"Count": "Total"})

        # --- Options (added Conversion % sort)
        col_opt1, col_opt2, col_opt3 = st.columns([1, 1, 1])
        with col_opt1:
            normalize_pct = st.checkbox(
                "Show Payments/Created as % of AC total (for the first two charts)",
                value=False, key="ac_stack_pct"
            )
        with col_opt2:
            sort_mode = st.selectbox(
                "Sort ACs by",
                ["Payments (desc)", "Deals Created (desc)", "Conversion % (desc)", "A–Z"],
                index=0, key="ac_stack_sort"
            )
        with col_opt3:
            top_n = st.number_input("Max ACs to show", min_value=1, max_value=500, value=30, step=1, key="ac_stack_topn")

        # --- Build AC ordering, including Conversion % option ---
        if sort_mode == "Payments (desc)":
            order_src = totals_pay.copy().sort_values("Total", ascending=False)

        elif sort_mode == "Deals Created (desc)":
            order_src = totals_create.copy().sort_values("Total", ascending=False)

        elif sort_mode == "Conversion % (desc)":
            # AC-level conversion% = (sum Paid) / (sum Created) * 100
            ac_conv = (
                totals_pay.rename(columns={"Total": "Paid"})
                .merge(totals_create.rename(columns={"Total": "Created"}), on="Academic Counsellor", how="outer")
                .fillna({"Paid": 0, "Created": 0})
            )
            ac_conv["ConvPct"] = np.where(ac_conv["Created"] > 0, ac_conv["Paid"] / ac_conv["Created"] * 100.0, 0.0)
            order_src = ac_conv.sort_values("ConvPct", ascending=False)[["Academic Counsellor"]]

        else:  # "A–Z"
            base_totals = totals_pay if not totals_pay.empty else totals_create
            order_src = base_totals[["Academic Counsellor"]].copy().sort_values("Academic Counsellor", ascending=True)

        ac_order = order_src["Academic Counsellor"].head(int(top_n)).tolist() if not order_src.empty else []

        def prep_for_chart(g_df, totals_df):
            g = g_df.merge(totals_df, on="Academic Counsellor", how="left")
            if ac_order:
                g = g[g["Academic Counsellor"].isin(ac_order)].copy()
                g["Academic Counsellor"] = pd.Categorical(g["Academic Counsellor"], categories=ac_order, ordered=True)
            else:
                g["Academic Counsellor"] = g["Academic Counsellor"].astype(str)
            if normalize_pct:
                g["Pct"] = np.where(g["Total"] > 0, g["Count"] / g["Total"] * 100.0, 0.0)
            return g

        g_pay_c    = prep_for_chart(g_pay, totals_pay)
        g_create_c = prep_for_chart(g_create, totals_create)

        def stacked_chart(g, title, use_pct):
            if g.empty:
                return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()

            y_field = alt.Y(
                ("Pct:Q" if use_pct else "Count:Q"),
                title=("% of AC total" if use_pct else "Count"),
                stack=True,
                scale=(alt.Scale(domain=[0, 100]) if use_pct else alt.Undefined)
            )
            tooltips = [
                alt.Tooltip("Academic Counsellor:N"),
                alt.Tooltip("Deal Source:N"),
                alt.Tooltip("Count:Q", title="Count"),
                alt.Tooltip("Total:Q", title="AC Total"),
            ]
            if use_pct:
                tooltips.append(alt.Tooltip("Pct:Q", title="% of AC", format=".1f"))

            chart = (
                alt.Chart(g)
                .mark_bar(opacity=0.9)
                .encode(
                    x=alt.X("Academic Counsellor:N", sort=ac_order, title="Academic Counsellor"),
                    y=y_field,
                    color=alt.Color("Deal Source:N", legend=alt.Legend(orient="bottom", title="Deal Source")),
                    tooltip=tooltips,
                )
                .properties(height=360, title=title)
            )
            return chart

        # ---- Conversion% stacked (Payments / Created within AC × Source)
        g_merge = (
            g_create.rename(columns={"Count": "Created"})
                    .merge(g_pay.rename(columns={"Count": "Paid"}),
                           on=["Academic Counsellor", "Deal Source"], how="outer")
                    .fillna({"Created": 0, "Paid": 0})
        )
        # keep AC order and top_n selection
        if ac_order:
            g_merge = g_merge[g_merge["Academic Counsellor"].isin(ac_order)].copy()
            g_merge["Academic Counsellor"] = pd.Categorical(g_merge["Academic Counsellor"], categories=ac_order, ordered=True)

        g_merge["ConvPct"] = np.where(g_merge["Created"] > 0, g_merge["Paid"] / g_merge["Created"] * 100.0, 0.0)

        def conversion_chart(g):
            if g.empty:
                return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_bar()
            tooltips = [
                alt.Tooltip("Academic Counsellor:N"),
                alt.Tooltip("Deal Source:N"),
                alt.Tooltip("Created:Q"),
                alt.Tooltip("Paid:Q"),
                alt.Tooltip("ConvPct:Q", title="Conversion %", format=".1f"),
            ]
            return (
                alt.Chart(g)
                .mark_bar(opacity=0.9)
                .encode(
                    x=alt.X("Academic Counsellor:N", sort=ac_order, title="Academic Counsellor"),
                    y=alt.Y("ConvPct:Q", title="Conversion % (Paid / Created)", scale=alt.Scale(domain=[0, 100]), stack=True),
                    color=alt.Color("Deal Source:N", legend=alt.Legend(orient="bottom", title="Deal Source")),
                    tooltip=tooltips,
                )
                .properties(height=360, title="Conversion % — stacked by Deal Source")
            )

        col_pay, col_create, col_conv = st.columns(3)
        with col_pay:
            st.altair_chart(
                stacked_chart(g_pay_c, "Payments (Payment Received — stacked by Deal Source)", use_pct=normalize_pct),
                use_container_width=True
            )
        with col_create:
            st.altair_chart(
                stacked_chart(g_create_c, "Deals Created (Create Date — stacked by Deal Source)", use_pct=normalize_pct),
                use_container_width=True
            )
        with col_conv:
            st.altair_chart(conversion_chart(g_merge), use_container_width=True)

        with st.expander("Download data used in stacked charts"):
            st.download_button(
                "Download CSV — Payments by AC × Deal Source",
                data=g_pay_c.sort_values(["Academic Counsellor", "Deal Source"]).to_csv(index=False).encode("utf-8"),
                file_name="ac_by_dealsource_payments.csv",
                mime="text/csv",
                key="ac_stack_dl_pay"
            )
            st.download_button(
                "Download CSV — Deals Created by AC × Deal Source",
                data=g_create_c.sort_values(["Academic Counsellor", "Deal Source"]).to_csv(index=False).encode("utf-8"),
                file_name="ac_by_dealsource_created.csv",
                mime="text/csv",
                key="ac_stack_dl_created"
            )
            st.download_button(
                "Download CSV — Conversion% by AC × Deal Source",
                data=g_merge.sort_values(["Academic Counsellor", "Deal Source"]).to_csv(index=False).encode("utf-8"),
                file_name="ac_by_dealsource_conversion_pct.csv",
                mime="text/csv",
                key="ac_stack_dl_conv"
            )
elif view == "Dashboard":
    st.subheader("Dashboard — Business Snapshot")

    # =========================
    # Safe base dataframe pick
    # =========================
    try:
        base = df_f.copy()
    except NameError:
        base = df.copy()

    # =========================
    # Local helpers (self-contained)
    # =========================
    def _get_col(dframe, candidates):
        """
        Return the first existing column from candidates.
        Tries exact, stripped, and case-insensitive matches.
        """
        cols = list(dframe.columns)
        # exact
        for c in candidates:
            if c in cols:
                return c
        # strip compare
        strip_map = {c: c.strip() for c in cols}
        for c in candidates:
            for orig, stripped in strip_map.items():
                if stripped == c:
                    return orig
        # case-insensitive strip
        low_map = {c.lower().strip(): c for c in cols}
        for c in candidates:
            key = c.lower().strip()
            if key in low_map:
                return low_map[key]
        return None

    def _to_dt(series):
        """Coerce to pandas datetime (errors->NaT)."""
        return pd.to_datetime(series, errors="coerce")

    def _norm_str(s):
        """Normalize string series safely."""
        return s.fillna("").astype(str).str.strip()

    def _month_bounds(any_date):
        """Return first_day, last_day for the calendar month of a date."""
        first = any_date.replace(day=1)
        # next month begin
        next_month = (pd.Timestamp(first) + pd.offsets.MonthBegin(1)).date()
        last = (pd.Timestamp(next_month) - pd.Timedelta(days=1)).date()
        return first, last

    # =========================
    # Column resolution
    # =========================
    CREATE_COL  = _get_col(base, ["Create Date"])
    PAY_COL     = _get_col(base, ["Payment Received Date", "Payment Received Date "])  # trailing space variant
    SRC_COL     = _get_col(base, ["JetLearn Deal Source", "_src_raw"])
    REF_INTENT  = _get_col(base, ["Referral Intent Source"])

    # Guard rails — warn if missing
    missing = [label for label, col in [
        ("Create Date", CREATE_COL),
        ("Payment Received Date", PAY_COL),
        ("JetLearn Deal Source", SRC_COL),
        ("Referral Intent Source", REF_INTENT),
    ] if col is None]
    if missing:
        st.warning(
            "Missing columns: " + ", ".join(missing) + ". Some dashboard metrics may be zero.",
            icon="⚠️"
        )

    # =========================
    # Parse dates & normalize text
    # =========================
    c_dt = _to_dt(base[CREATE_COL]) if CREATE_COL else pd.Series(pd.NaT, index=base.index)
    p_dt = _to_dt(base[PAY_COL])    if PAY_COL   else pd.Series(pd.NaT, index=base.index)

    src  = _norm_str(base[SRC_COL])     if SRC_COL    else pd.Series("", index=base.index)
    rint = _norm_str(base[REF_INTENT])  if REF_INTENT else pd.Series("", index=base.index)

    # Referral / Intent matchers (based on your sheet values)
    # - Source uses "Referrals" (plural). Be robust: any text containing "referr".
    is_referral = src.str.contains("referr", case=False, na=False)
    # - Intent should include "Sales Generated" variants (e.g., "Gift Card - Sales Generated")
    is_sales_generated = rint.str.contains(r"Sales Generated", case=False, na=False)

    # =========================
    # Time windows (Asia/Kolkata)
    # =========================
    now_ist = pd.Timestamp.now(tz="Asia/Kolkata")
    today = now_ist.date()
    yesterday = (now_ist - pd.Timedelta(days=1)).date()

    # Current month (window for "This Month")
    cm_start = today.replace(day=1)
    cm_end = today  # inclusive up to today

    # Last month window (full calendar month)
    lm_start_tmp, lm_end_tmp = _month_bounds((pd.Timestamp(cm_start) - pd.Timedelta(days=1)).date())
    lm_start, lm_end = lm_start_tmp, lm_end_tmp

    # Date-between helper over .dt.date
    def _between(series_dt, start_d, end_d):
        s_date = series_dt.dt.date
        return s_date.between(start_d, end_d, inclusive="both")

    # =========================
    # Mode toggle (per your definition)
    # =========================
    mode = st.radio(
        "Counting mode",
        ["Cohort", "MTD"],   # Cohort default
        index=0,
        horizontal=True,
        help=(
            "Cohort: Outputs in the selected period; deal Create Date can be anything.\n"
            "MTD: Enrolments only count if the deal was CREATED in the active calendar month of that period."
        ),
    )

    # =========================
    # Metric computer
    # =========================
    def compute_metrics(window_start, window_end, active_month_start, active_month_end):
        """
        window_* : the visible KPI window (e.g., Yesterday, Today, Last Month, This Month)
        active_month_* : the full calendar month corresponding to the window (for MTD enrolment restriction)
        """
        # Created-based masks (by Create Date in the window)
        deals_created_mask = _between(c_dt, window_start, window_end)
        referral_created_mask = deals_created_mask & is_referral
        referral_sales_from_created = referral_created_mask & is_sales_generated

        # Payments in window (by Payment Received Date)
        payments_in_window = _between(p_dt, window_start, window_end)

        # Enrolments logic
        if mode == "Cohort":
            enrol_count = int(payments_in_window.sum())  # payment-by-date only
        else:
            # MTD: restrict enrolments to deals created in the active calendar month
            created_in_active_month = _between(c_dt, active_month_start, active_month_end)
            enrol_count = int((payments_in_window & created_in_active_month).sum())

        return {
            "Deals Created": int(deals_created_mask.sum()),
            "Enrolments": enrol_count,
            "Referral Deals Created": int(referral_created_mask.sum()),
            "Referral – Sales Generated": int(referral_sales_from_created.sum()),
        }

    # =========================
    # Define periods & their active months
    # =========================
    # Yesterday
    y_m_start, y_m_end = _month_bounds(yesterday)
    # Today
    t_m_start, t_m_end = _month_bounds(today)
    # Last Month (active month is the last calendar month)
    lm_m_start, lm_m_end = lm_start, lm_end
    # This Month (active month is the current calendar month)
    tm_m_start, tm_m_end = _month_bounds(today)

    PERIODS = [
        # label, window_start, window_end, active_month_start, active_month_end
        ("Yesterday", yesterday, yesterday, y_m_start, y_m_end),
        ("Today", today, today, t_m_start, t_m_end),
        ("Last Month", lm_start, lm_end, lm_m_start, lm_m_end),
        ("This Month", cm_start, cm_end, tm_m_start, tm_m_end),
    ]

    # =========================
    # Render four KPI boxes
    # =========================
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]

    for (title, w_start, w_end, m_start, m_end), col in zip(PERIODS, cols):
        mets = compute_metrics(w_start, w_end, m_start, m_end)
        with col:
            st.markdown(f"### {title}")
            st.markdown(
                f"""
                <div style="border:1px solid #e5e7eb; border-radius:14px; padding:10px; background:#ffffff;">
                  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span><strong>Deals Created</strong></span><span>{mets['Deals Created']}</span>
                  </div>
                  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span><strong>Enrolments</strong></span><span>{mets['Enrolments']}</span>
                  </div>
                  <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                    <span><strong>Referral Deals Created</strong></span><span>{mets['Referral Deals Created']}</span>
                  </div>
                  <div style="display:flex; justify-content:space-between;">
                    <span><strong>Referral – Sales Generated</strong></span><span>{mets['Referral – Sales Generated']}</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
