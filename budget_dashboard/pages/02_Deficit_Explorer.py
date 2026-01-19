import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import aggregate_bucket_totals, get_meta

st.title("📉 Deficit Explorer (Outlays vs Receipts)")

if "df" not in st.session_state:
    st.warning("Load data on the main page first.")
    st.stop()

df = st.session_state["df"]
meta = get_meta(df)

agg = aggregate_bucket_totals(df)

# -----------------------------
# Copilot dashboard state defaults
# -----------------------------
dash = st.session_state.get("dash_state", {})
if not isinstance(dash, dict):
    dash = {}

# Define outlays buckets and receipts bucket (adjust if your bucket naming differs)
outlay_buckets_default = dash.get("deficit_outlay_buckets", ["discretionary", "mandatory", "net_interest"])
receipts_bucket_default = dash.get("deficit_receipts_bucket", "receipts")

bucket_names = sorted(agg["bucket"].unique())

colA, colB = st.columns([2, 1])
with colA:
    outlay_buckets = st.multiselect(
        "Buckets to count as OUTLAYS",
        options=bucket_names,
        default=[b for b in outlay_buckets_default if b in bucket_names],
    )
with colB:
    receipts_bucket = st.selectbox(
        "Bucket to count as RECEIPTS",
        options=bucket_names,
        index=bucket_names.index(receipts_bucket_default) if receipts_bucket_default in bucket_names else 0,
    )

years = sorted(agg["fy"].unique())
min_y, max_y = int(min(years)), int(max(years))

# FY range default from dash_state if present
dash_range = dash.get("fy_range")
if isinstance(dash_range, (list, tuple)) and len(dash_range) == 2:
    default_range = (max(min_y, int(dash_range[0])), min(max_y, int(dash_range[1])))
else:
    default_range = (min_y, max_y)

fy_range = st.slider(
    "Fiscal year range",
    min_value=min_y,
    max_value=max_y,
    value=default_range
)

view = agg[(agg["fy"] >= fy_range[0]) & (agg["fy"] <= fy_range[1])].copy()

outlays = (
    view[view["bucket"].isin(outlay_buckets)]
    .groupby("fy", as_index=False)["display_total"]
    .sum()
    .rename(columns={"display_total": "outlays"})
)

receipts = (
    view[view["bucket"] == receipts_bucket]
    .groupby("fy", as_index=False)["display_total"]
    .sum()
    .rename(columns={"display_total": "receipts"})
)

gdps = (
    view[view["bucket"] == receipts_bucket]
    .groupby("fy", as_index=False)["gdp_total"]
    .last()
    .rename(columns={"gdp_total": "GDP"})
)

merged = outlays.merge(receipts, on="fy", how="outer").sort_values("fy")
merged = merged.merge(gdps, on="fy", how="left").sort_values("fy").copy()
merged["outlays"] = merged["outlays"].fillna(0.0)
merged["receipts"] = merged["receipts"].fillna(0.0)
merged["deficit"] = merged["outlays"] - merged["receipts"]
merged["deficit_over_gdp"] = merged["deficit"] / merged["GDP"]

fig = go.Figure()
fig.add_vrect(
    x0=2023.5, x1=2029.5,
    annotation_text="Estimate", annotation_position="top left",
    fillcolor="green", opacity=0.25, line_width=0
)

fig.add_vrect(
    x0=2029.5, x1=merged["fy"].max() + 0.5,
    annotation_text="Forecast", annotation_position="top left",
    fillcolor="blue", opacity=0.25, line_width=0
)
fig.add_trace(go.Scatter(x=merged["fy"], y=merged["outlays"], mode="lines+markers", name="Outlays"))
fig.add_trace(go.Scatter(x=merged["fy"], y=merged["receipts"], mode="lines+markers", name="Receipts"))
fig.update_layout(
    title="Outlays vs Receipts (Actual (-2023) + Estimate (2024-29) + Forecast (2030-35) combined)",
    xaxis_title="FY",
    yaxis_title="Value",
    hovermode="x unified"
)
st.plotly_chart(fig, width="stretch")

fig2 = make_subplots(specs=[[{"secondary_y": True}]])
fig2.add_vrect(
    x0=2023.5, x1=2029.5,
    annotation_text="Estimate", annotation_position="top left",
    fillcolor="green", opacity=0.25, line_width=0, secondary_y=False
)

fig2.add_vrect(
    x0=2029.5, x1=merged["fy"].max() + 0.5,
    annotation_text="Forecast", annotation_position="top left",
    fillcolor="blue", opacity=0.25, line_width=0, secondary_y=False
)
fig2.add_trace(go.Bar(x=merged["fy"], y=merged["deficit"], name="Deficit (Outlays - Receipts) (left axis)"), secondary_y=False)
fig2.add_trace(go.Scatter(x=merged["fy"], y=merged["deficit_over_gdp"], mode="lines+markers", name="Defict as % of GDP (right-axis)"), secondary_y=True)
fig2.update_layout(
    title="Deficit over time",
    xaxis_title="FY",
    yaxis_title="Deficit (Outlays - Receipts)",
    yaxis2_title="Defict as % of GDP",
    hovermode="x unified",
    yaxis2=dict(tickformat=".2%")
)
st.plotly_chart(fig2, width="stretch")

st.subheader("Data table")
st.dataframe(merged, width="stretch")

st.download_button(
    "Download deficit table as CSV",
    data=merged.to_csv(index=False).encode("utf-8"),
    file_name="deficit_table.csv",
    mime="text/csv",
)

st.caption(f"Detected last actual FY: {meta['fy_last_actual']}. Forecast begins after that year.")

dash = st.session_state.get("dash_state", {})
if isinstance(dash, dict):
    dash["deficit_outlay_buckets"] = outlay_buckets
    dash["deficit_receipts_bucket"] = receipts_bucket
    dash["fy_range"] = [int(fy_range[0]), int(fy_range[1])]
    st.session_state["dash_state"] = dash