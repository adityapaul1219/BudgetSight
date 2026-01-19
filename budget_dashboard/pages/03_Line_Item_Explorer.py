import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from streamlit_plotly_events import plotly_events

st.title("🔍 Line Item Explorer (Drill-down)")

if "df" not in st.session_state:
    st.warning("Load data on the main page first.")
    st.stop()

df = st.session_state["df"].copy()

# -----------------------------
# Prepare / validate
# -----------------------------
required = {"bucket", "item", "fy", "value", "yhat", "yhat_lower", "yhat_upper"}
missing = sorted(list(required - set(df.columns)))
if missing:
    st.error(f"Missing required columns in df: {missing}")
    st.stop()

df["bucket"] = df["bucket"].astype(str).str.strip()
df["item"] = df["item"].astype(str).str.strip()
df["fy"] = pd.to_numeric(df["fy"], errors="coerce").astype("Int64")
for c in ["value", "yhat", "yhat_lower", "yhat_upper"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["bucket", "item", "fy"]).copy()
df["fy"] = df["fy"].astype(int)

fy_last_actual = int(df.loc[df["value"].notna(), "fy"].max()) if df["value"].notna().any() else int(df["fy"].max())

# Display series: actual where available else forecast
df["display_value"] = np.where(df["value"].notna(), df["value"], df["yhat"])
df["display_value"] = pd.to_numeric(df["display_value"], errors="coerce")  # IMPORTANT (fixes pie issues)

# Identify receipts bucket
is_receipts = df["bucket"].str.lower() == "receipts"
outlay_df = df[~is_receipts].copy()
receipts_df = df[is_receipts].copy()

# -----------------------------
# Session state (selections)
# -----------------------------
st.session_state.setdefault("sel_outlay_bucket", None)
st.session_state.setdefault("sel_outlay_item", None)
st.session_state.setdefault("sel_receipt_item", None)

# Apply copilot dash_state defaults
dash = st.session_state.get("dash_state", {})
if isinstance(dash, dict):
    if dash.get("outlay_bucket") is not None:
        st.session_state["sel_outlay_bucket"] = dash["outlay_bucket"]
    if dash.get("outlay_item") is not None:
        st.session_state["sel_outlay_item"] = dash["outlay_item"]
    if dash.get("receipt_item") is not None:
        st.session_state["sel_receipt_item"] = dash["receipt_item"]

# -----------------------------
# Debug toggle
# -----------------------------
debug = False # st.toggle("Debug (show pie inputs + click payloads)", value=False)

# -----------------------------
# Helpers
# -----------------------------

def pie_click(labels, values, key: str, height: int = 360, debug: bool = False):
    """
    Render a Plotly pie AND return the clicked slice label (or None).

    Fix: some environments only return {"curveNumber","pointNumber"} from streamlit-plotly-events,
    so we map pointNumber -> labels list.
    """
    # Convert to plain Python types (important for stable rendering)
    lab = [str(x) for x in list(labels)]
    val = [float(x) for x in list(values)]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=lab,
                values=val,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Total: %{value:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        clickmode="event+select",
        colorway=px.colors.qualitative.Plotly,
    )

    evt = plotly_events(
        fig,
        click_event=True,
        select_event=True,
        hover_event=False,
        key=key,
    )

    if debug:
        st.write(f"{key} raw event:", evt)

    if not evt:
        return None

    d0 = evt[0] if isinstance(evt, list) and len(evt) > 0 else {}
    # Preferred: label returned directly (sometimes available)
    if isinstance(d0, dict) and d0.get("label") is not None:
        return str(d0["label"])

    # Fallback: map pointNumber -> label
    pn = d0.get("pointNumber") if isinstance(d0, dict) else None
    if pn is None:
        return None

    try:
        idx = int(pn)
        if 0 <= idx < len(lab):
            return lab[idx]
    except Exception:
        return None

    return None


    
def clicked_label_from_selection(sel: dict | None) -> str | None:
    """
    Extract pie-slice label from Streamlit's Plotly selection dict.
    """
    if not sel or not isinstance(sel, dict):
        return None
    pts = sel.get("selection", {}).get("points", [])
    if not pts:
        return None
    p0 = pts[0]
    # Prefer customdata since we control it
    return p0.get("customdata") or p0.get("label") or p0.get("text") or None

    
def fy_group_sum(d: pd.DataFrame, group_cols: list[str], fy: int) -> pd.DataFrame:
    s = d[d["fy"] == fy].copy()
    if s.empty:
        return s
    g = s.groupby(group_cols, as_index=False)["display_value"].sum()
    g = g.rename(columns={"display_value": "amount"})
    # sanitize (fixes equal-sized/black pies)
    g["amount"] = pd.to_numeric(g["amount"], errors="coerce")
    g = g.dropna(subset=["amount"])
    g = g[g["amount"] != 0]
    return g

def build_dual_axis_chart(series: pd.DataFrame, title: str) -> go.Figure:
    s = series.sort_values("fy").copy()
    s["yoy_change"] = s["display_value"]/s["display_value"].shift(1) - 1.0

    fig = make_subplots(specs=[[{"secondary_y": True}]])



    
    fig.add_trace(
        go.Scatter(
            x=s["fy"], y=s["display_value"],
            mode="lines+markers",
            name="Value (actual/forecast)",
            hovertemplate="FY %{x}<br>Value: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=s["fy"], y=s["yoy_change"],
            name="YoY change",
            opacity=0.55,
            hovertemplate="FY %{x}<br>YoY: %{y:,.0f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # fig.add_vline(
    #     x=fy_last_actual,
    #     line_width=1,
    #     line_dash="dot",
    #     annotation_text=f"Last actual FY {fy_last_actual}",
    #     annotation_position="top left",
    # )

    fig.add_vrect(x0=2023.5, x1=2029.5,
                  annotation_text="Estimate", annotation_position="top left",
                  fillcolor="green", opacity=0.25, line_width=0)
    
    
    fig.add_vrect(x0=2029.5, x1=series["fy"].max()+0.5,
                  annotation_text="Forecast", annotation_position="top left",
                  fillcolor="blue", opacity=0.25, line_width=0)
    

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Fiscal Year")
    fig.update_yaxes(title_text="Value", secondary_y=False)
    fig.update_yaxes(title_text="YoY change", secondary_y=True)
    return fig

def series_for(bucket: str, item: str) -> pd.DataFrame:
    s = df[(df["bucket"].str.lower() == bucket.lower()) & (df["item"].str.lower() == item.lower())].copy()
    if s.empty:
        return s
    s = (
        s.groupby(["fy"], as_index=False)
        .agg(display_value=("display_value", "sum"))
        .sort_values("fy")
    )
    s["display_value"] = pd.to_numeric(s["display_value"], errors="coerce")
    s = s.dropna(subset=["display_value"])
    return s

def pick_label(event_list):
    # streamlit-plotly-events returns a list of dicts; pie slices usually include "label"
    if not event_list:
        return None
    d0 = event_list[0]
    return d0.get("label") or d0.get("text") or None

# -----------------------------
# Reset selections button
# -----------------------------
if st.button("🔄 Reset selections", key="reset_selections"):
    st.session_state["sel_outlay_bucket"] = None
    st.session_state["sel_outlay_item"] = None
    st.session_state["sel_receipt_item"] = None
    st.rerun()

# -----------------------------
# 1) Descriptive “table” (bucket list with expand)
# -----------------------------
st.subheader("Outlay buckets (click a bucket to expand its items)")

bucket_summary = (
    outlay_df.groupby("bucket")["item"]
    .nunique()
    .reset_index(name="num_items")
    .sort_values(["num_items", "bucket"], ascending=[False, True])
    .reset_index(drop=True)
)

hdr = st.columns([4, 4, 2, 2])
hdr[0].markdown("**Bucket**")
hdr[1].markdown("**Description**")
hdr[2].markdown("**# Items**")
hdr[3].markdown("**Get Details**")

bucket_description = ["These outlays do not require Congressional approval, as they are already pre-decided by law. They grow or shrink automatically based on demographics and economic situations.",
                     "These are decided through Congress annually, and are more short-term, flexible adjustments. This category was the cause for the government shutdown recently.\nExamples: Defense, Education, Transportation, Scientific Research",
                     "Net interest is interest paid on the national debt"]

for i, r in bucket_summary.iterrows():
    b = r["bucket"]
    n = int(r["num_items"])
    cols = st.columns([4, 4, 2, 2])
    is_selected = (st.session_state["sel_outlay_bucket"] or "").lower() == str(b).lower()
    cols[0].write(f"👉 {b}" if is_selected else b)
    cols[1].write(bucket_description[i])
    cols[2].write(n)
    if cols[3].button("Get Details", key=f"open_bucket_{b}"):
        st.session_state["sel_outlay_bucket"] = b
        st.session_state["sel_outlay_item"] = None

sel_bucket = st.session_state.get("sel_outlay_bucket")
if sel_bucket:
    st.markdown(f"**Items in: {sel_bucket}**")
    items_tbl = (
        outlay_df[outlay_df["bucket"].str.lower() == sel_bucket.lower()][["item"]]
        .drop_duplicates()
        .sort_values("item")
        .reset_index(drop=True)
    )
    st.dataframe(items_tbl,              
                 width="stretch",
                # use_container_width=True, 
                 height=240)

st.markdown("---")

st.subheader("Receipts items")
receipts_items = receipts_df[["item"]].drop_duplicates().sort_values("item").reset_index(drop=True)
st.dataframe(receipts_items,             
             width="stretch",
            # use_container_width=True, 
             height=220)

st.markdown("---")

# -----------------------------
# 2) FY dropdown + pie charts
# -----------------------------
all_fys = sorted(df["fy"].unique().tolist())
dash = st.session_state.get("dash_state", {})
default_fy = int(dash.get("fy", 2025)) if isinstance(dash, dict) else 2025
fy_selected = st.selectbox("Select fiscal year (FY)", all_fys, index=all_fys.index(default_fy))

outlay_by_bucket = fy_group_sum(outlay_df, ["bucket"], fy_selected).sort_values("amount", ascending=False)
receipts_by_item = fy_group_sum(receipts_df, ["item"], fy_selected).sort_values("amount", ascending=False)

if debug:
    st.write("outlay_by_bucket (pie input):", outlay_by_bucket)
    st.write("receipts_by_item (pie input):", receipts_by_item)

outlays_total = float(outlay_by_bucket["amount"].sum()) if not outlay_by_bucket.empty else 0.0
receipts_total = float(receipts_by_item["amount"].sum()) if not receipts_by_item.empty else 0.0
den = outlays_total + receipts_total

if den <= 0:
    w_out, w_rec = 5, 5
else:
    w_out = max(3, int(round(10 * outlays_total / den)))
    w_rec = 10 - w_out
    w_rec = max(3, w_rec)
    w_out = 10 - w_rec

st.subheader(f"FY {fy_selected}: Click pies to drill down")
col_out, col_rec = st.columns([w_out, w_rec], vertical_alignment="top")

# -----------------------------
# 3) Outlays pie (bucket) -> click sets selected bucket -> item pie
# -----------------------------
with col_out:
    st.markdown("### 🥧 Outlays (by bucket)")
    
    o = outlay_by_bucket.copy()
    o["amount"] = pd.to_numeric(o["amount"], errors="coerce")
    o = o.dropna(subset=["amount"])
    o["amount_for_pie"] = o["amount"].clip(lower=0)
    
    if not o.empty and o["amount_for_pie"].sum() > 0:
        clicked_bucket = pie_click(
            labels=o["bucket"].tolist(),
            values=o["amount_for_pie"].tolist(),
            key=f"pie_outlay_bucket_{fy_selected}",
            height=360,
            debug=debug,
        )
        if clicked_bucket:
            st.session_state["sel_outlay_bucket"] = clicked_bucket
            st.session_state["sel_outlay_item"] = None
    
        st.caption(f"Selected bucket: **{st.session_state.get('sel_outlay_bucket') or '—'}**")
    
    # =======================================================
    
    sel_bucket = st.session_state.get("sel_outlay_bucket")
    if sel_bucket:
        out_items_fy = (
            outlay_df[(outlay_df["bucket"].str.lower() == sel_bucket.lower()) & (outlay_df["fy"] == fy_selected)]
            .groupby("item", as_index=False)["display_value"].sum()
            .rename(columns={"display_value": "amount"})
            .sort_values("amount", ascending=False)
        )
        out_items_fy["amount"] = pd.to_numeric(out_items_fy["amount"], errors="coerce")
        out_items_fy = out_items_fy.dropna(subset=["amount"])
        out_items_fy["amount_for_pie"] = out_items_fy["amount"].clip(lower=0)
    
        if not out_items_fy.empty and out_items_fy["amount_for_pie"].sum() > 0:
            clicked_item = pie_click(
                labels=out_items_fy["item"].tolist(),
                values=out_items_fy["amount_for_pie"].tolist(),
                key=f"pie_outlay_items_{fy_selected}_{sel_bucket}",
                height=420,
                debug=debug,
            )
            if clicked_item:
                st.session_state["sel_outlay_item"] = str(clicked_item)
    
            st.caption(f"Selected outlay item: **{st.session_state.get('sel_outlay_item') or '—'}**")

# -----------------------------
# 4) Receipts pie (item) -> click sets receipt item
# -----------------------------
with col_rec:
    st.markdown("### 🥧 Receipts (by item)")
    
    # receipts_by_item must have columns: item, amount
    r = receipts_by_item.copy()
    r["amount"] = pd.to_numeric(r["amount"], errors="coerce")
    r = r.dropna(subset=["amount"])
    
    # If any negative values exist, pies behave badly—clip for pie only
    r["amount_for_pie"] = r["amount"].clip(lower=0)
    
    if r.empty or r["amount_for_pie"].sum() <= 0:
        st.info("No receipts data found for this FY (after cleaning/clipping).")
    else:
        clicked = pie_click(
            labels=r["item"].tolist(),
            values=r["amount_for_pie"].tolist(),
            key=f"pie_receipts_items_{fy_selected}",
            height=360,
            debug=debug,
        )
        if clicked:
            st.session_state["sel_receipt_item"] = clicked
    
        st.caption(f"Selected receipts item: **{st.session_state.get('sel_receipt_item') or '—'}**")

st.markdown("---")

# -----------------------------
# 5) Time-series charts
# -----------------------------
st.subheader("Selected outlay item (time series)")
sel_bucket = st.session_state.get("sel_outlay_bucket")
sel_out_item = st.session_state.get("sel_outlay_item")

if sel_bucket and sel_out_item:
    s = series_for(sel_bucket, sel_out_item)
    if s.empty:
        st.info("No time series found for the selected outlay item.")
    else:
        st.plotly_chart(
            build_dual_axis_chart(s, f"Outlays — {sel_bucket} → {sel_out_item}"),
            width="stretch",
            # use_container_width=True,
        )
else:
    st.info("Click an outlay bucket slice, then click a line-item slice to generate the chart.")

st.markdown("---")

st.subheader("Selected receipts item (time series)")
sel_rec_item = st.session_state.get("sel_receipt_item")

if sel_rec_item:
    s = series_for("receipts", sel_rec_item)
    if s.empty:
        st.info("No time series found for the selected receipts item.")
    else:
        st.plotly_chart(
            build_dual_axis_chart(s, f"Receipts — {sel_rec_item}"),
            width="stretch",
            # use_container_width=True,
        )
else:
    st.info("Click a receipts slice in the receipts pie to generate the chart.")


dash = st.session_state.get("dash_state", {})
if isinstance(dash, dict):
    dash["fy"] = int(fy_selected)
    dash["outlay_bucket"] = st.session_state.get("sel_outlay_bucket")
    dash["outlay_item"] = st.session_state.get("sel_outlay_item")
    dash["receipt_item"] = st.session_state.get("sel_receipt_item")
    st.session_state["dash_state"] = dash

