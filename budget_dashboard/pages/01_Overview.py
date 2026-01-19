import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils import describe_df, aggregate_bucket_totals, get_meta

st.title("🏁 Overview")

if "df" not in st.session_state:
    st.warning("Load data on the main page first.")
    st.stop()

df = st.session_state["df"]
info = describe_df(df)
meta = get_meta(df)

# -----------------------------
# Copilot dashboard state defaults
# -----------------------------
dash = st.session_state.get("dash_state", {})
if not isinstance(dash, dict):
    dash = {}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{info['rows']:,}")
c2.metric("Buckets", f"{info['n_buckets']}")
c3.metric("Items", f"{info['n_items']}")
c4.metric("Last Actual FY", f"{meta['fy_last_actual']}")

agg = aggregate_bucket_totals(df)

st.subheader("Totals by Bucket (Actual (upto 2023) + Estimate (2024-29) + Forecast (2030-35) combined)")
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

fig = go.Figure()

fig.add_vrect(
    x0=2023.5, x1=2029.5,
    annotation_text="Estimate", annotation_position="top left",
    fillcolor="green", opacity=0.25, line_width=0
)

fig.add_vrect(
    x0=2029.5, x1=agg["fy"].max() + 0.5,
    annotation_text="Forecast", annotation_position="top left",
    fillcolor="blue", opacity=0.25, line_width=0
)

fig.add_trace(go.Scatter(
    name='mandatory',
    x=view[view["bucket"] == "mandatory"]["fy"],
    y=view[view["bucket"] == "mandatory"]["display_total"],
    mode='lines',
    stackgroup='one',
    fillcolor='lightblue',
    opacity=0.2,
    line=dict(width=0)
))

fig.add_trace(go.Scatter(
    name='discretionary',
    x=view[view["bucket"] == "discretionary"]["fy"],
    y=view[view["bucket"] == "discretionary"]["display_total"],
    mode='lines',
    stackgroup='one',
    fillcolor='wheat',
    opacity=0.2,
    line=dict(width=0)
))

fig.add_trace(go.Scatter(
    name='net_interest',
    x=view[view["bucket"] == "net_interest"]["fy"],
    y=view[view["bucket"] == "net_interest"]["display_total"],
    mode='lines',
    stackgroup='one',
    fillcolor='lightgrey',
    opacity=0.2,
    line=dict(width=0)
))

fig.add_trace(
    go.Scatter(
        x=view[view["bucket"] == "receipts"]["fy"],
        y=view[view["bucket"] == "receipts"]["display_total"],
        mode="lines+markers",
        name="receipts",
        line=dict(color='green', width=2),
        marker=dict(color='black', size=5, line=dict(color='lightgreen', width=2))
    )
)

fig.update_layout(
    title="Receipts and Outlay bucket totals over time (actual (upto 2023) and estimate (2023-29), forecast afterward (2030-35))",
    xaxis_title="Fiscal Year",
    yaxis_title="Total",
    hovermode="x unified",
)

st.plotly_chart(fig, width="stretch")

st.subheader("Snapshot: FY totals")
fys = sorted(view["fy"].unique())

# default selected FY from dash_state if possible, else 2025 if present, else first
dash_fy = dash.get("fy")
if isinstance(dash_fy, (int, float)) and int(dash_fy) in fys:
    default_value = int(dash_fy)
elif 2025 in fys:
    default_value = 2025
else:
    default_value = fys[0]

default_ix = fys.index(default_value)
selected_fy = st.selectbox("FY", fys, index=default_ix)

latest = view[view["fy"] == selected_fy].sort_values("display_total", ascending=False)

fig2 = go.Figure()

fig2.add_trace(
    go.Bar(
        x=["Receipts"],
        y=[latest[latest["bucket"] == "receipts"]["display_total"].iloc[0]],
        name="Receipts",
        marker_color='lightgreen'
    )
)

fig2.add_trace(
    go.Bar(
        x=["Outlays"],
        y=[latest[latest["bucket"] == "mandatory"]["display_total"].iloc[0]],
        name="Mandatory",
        marker_color='lightblue'
    )
)

fig2.add_trace(
    go.Bar(
        x=["Outlays"],
        y=[latest[latest["bucket"] == "discretionary"]["display_total"].iloc[0]],
        name="Discretionary",
        marker_color='wheat'
    )
)

fig2.add_trace(
    go.Bar(
        x=["Outlays"],
        y=[latest[latest["bucket"] == "net_interest"]["display_total"].iloc[0]],
        name="Net Interest",
        marker_color='lightgrey'
    )
)

fig2.update_layout(
    barmode="stack",
    title="Federal Receipts vs Outlays Breakdown",
    yaxis_title="Amount (Trillions USD)",
    xaxis_title="",
    legend_title="Category",
    height=500
)

st.plotly_chart(fig2, width="stretch")

with st.expander("Download aggregated bucket totals"):
    st.download_button(
        "Download CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="bucket_totals.csv",
        mime="text/csv",
    )

dash = st.session_state.get("dash_state", {})
if isinstance(dash, dict):
    dash["fy_range"] = [int(fy_range[0]), int(fy_range[1])]
    dash["fy"] = int(selected_fy)
    st.session_state["dash_state"] = dash