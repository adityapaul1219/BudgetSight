import streamlit as st
import pandas as pd
from pathlib import Path

from utils import load_df, validate_and_prepare_df, describe_df



DATA_PATH = Path(__file__).parent / "data" / "forecasts_actuals_df.pkl"






















st.set_page_config(
    page_title="BudgetSight Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 BudgetSight Dashboard (Actual + Prophet Forecast)")

# -----------------------------
# 0) Initialize shared dashboard state (for Copilot + cross-page sync)
# -----------------------------
if "dash_state" not in st.session_state or not isinstance(st.session_state["dash_state"], dict):
    st.session_state["dash_state"] = {
        "fy": None,
        "fy_range": None,
        "outlay_bucket": None,
        "outlay_item": None,
        "receipt_item": None,
        "deficit_outlay_buckets": ["discretionary", "mandatory", "net_interest"],
        "deficit_receipts_bucket": "receipts",
        "page": None,
    }

st.markdown(
    """

Welcome to the BudgetSight dashboard, which help understand the users the components of Federal budget, deficit, in a easy-to-interact way with the power of AI and Machine Learning

A pre-calculated file is already uploaded with 
- Actual hisorical data from 1980 to 2023, 
- Government provided estimate from 2024 to 2029, and 
- Machine Learning generated forecast from 2030 to 2035

Please navigate to individual pages using the sidebar on the left

The dataset contains these columns:

- `bucket` (e.g., discretionary, mandatory, net_interest, receipts)
- `item`
- `fy` (fiscal year, integer)
- `value` (actual)
- `yhat` (forecast)
- `yhat_lower`
- `yhat_upper`

This app automatically detects:
- detect the last actual fiscal year,
- build a unified “display value” series (actual where available, forecast afterward),
- enable interactive exploration across pages.
"""
)

downloads = Path("~/Downloads").expanduser()

# uploaded = st.file_uploader("Upload data file (.csv or .parquet or .pkl)", type=["csv", "parquet", "pkl"])

col1, col2 = st.columns([1, 2], vertical_alignment="top")

with col1:
    # use_example = st.toggle("Use a tiny example dataset (for testing)", value=False)

    st.markdown("---")
    st.write("Pre-calculated data loading")
    # if st.button("🔄 Reset dashboard selections"):
    #     # reset only the "view state" (not df)
    #     st.session_state["dash_state"].update(
    #         {
    #             "fy": None,
    #             "fy_range": None,
    #             "outlay_bucket": None,
    #             "outlay_item": None,
    #             "receipt_item": None,
    #             "deficit_outlay_buckets": ["discretionary", "mandatory", "net_interest"],
    #             "deficit_receipts_bucket": "receipts",
    #             "page": None,
    #         }
    #     )
    #     # also reset local selection keys if user is on another page later
    #     st.session_state.pop("sel_outlay_bucket", None)
    #     st.session_state.pop("sel_outlay_item", None)
    #     st.session_state.pop("sel_receipt_item", None)
    #     st.rerun()

# -----------------------------
# 1) Load data
#    Priority:
#      a) example toggle (optional)
#      b) uploaded file
#      c) local fallback (your current behavior) — ONLY if file exists
# -----------------------------

df = None

# if use_example:
#     demo = pd.DataFrame(
#         {
#             "bucket": ["receipts"] * 6 + ["discretionary"] * 6,
#             "item": ["Individual Income Taxes"] * 6 + ["Education"] * 6,
#             "fy": [2018, 2019, 2020, 2021, 2022, 2023] * 2,
#             "value": [1500, 1550, 1400, 1650, 1700, 1750, 100, 105, 120, 115, 118, 125],
#             "yhat": [None] * 12,
#             "yhat_lower": [None] * 12,
#             "yhat_upper": [None] * 12,
#         }
#     )
#     demo_fore = pd.DataFrame(
#         {
#             "bucket": ["receipts"] * 3 + ["discretionary"] * 3,
#             "item": ["Individual Income Taxes"] * 3 + ["Education"] * 3,
#             "fy": [2024, 2025, 2026, 2024, 2025, 2026],
#             "value": [None] * 6,
#             "yhat": [1800, 1850, 1900, 130, 132, 135],
#             "yhat_lower": [1750, 1800, 1850, 125, 128, 130],
#             "yhat_upper": [1850, 1900, 1950, 135, 136, 140],
#         }
#     )
#     df_raw = pd.concat([demo, demo_fore], ignore_index=True)
#     df = validate_and_prepare_df(df_raw)

# elif uploaded is not None:
#     df_raw = load_df(uploaded)
#     df = validate_and_prepare_df(df_raw)

# else:
#     # Local fallback for YOUR machine (won't work on Streamlit Cloud unless file is in repo)
#     local_path = downloads / "forecasts_actuals_df.pkl"
#     if local_path.exists():
#         df_raw = pd.read_pickle(local_path)
#         df_raw = df_raw[df_raw["fy"] >= 1980].copy()
#         df = validate_and_prepare_df(df_raw)

# Store df if loaded






local_path = downloads / "forecasts_actuals_df.pkl"
if local_path.exists():
    df_raw = pd.read_pickle(DATA_PATH)
    df_raw = df_raw[df_raw["fy"] >= 1980].copy()
    df = validate_and_prepare_df(df_raw)


if df is not None:
    st.session_state["df"] = df


with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(200), width="stretch")

st.caption("Use the left sidebar to navigate pages.")

# -----------------------------
# 2) Show status + initialize dash_state defaults if missing
# -----------------------------
with col2:
    if "df" in st.session_state:
        st.success("Data loaded ✅")
        info = describe_df(st.session_state["df"])

        st.markdown(
            f"""
            **Rows:** {info["rows"]:,}  
            **Buckets:** {info["n_buckets"]}  
            **Items:** {info["n_items"]}  
            **FY range:** {info["fy_min"]} – {info["fy_max"]}  
            **Last actual FY (detected):** {info["fy_last_actual"]}  
            **Last forecast FY (detected):** {info["fy_last_forecast"]}  
            """
        )

        # Fill initial dash_state defaults ONLY if not already set
        dash = st.session_state["dash_state"]
        if dash.get("fy_range") is None:
            dash["fy_range"] = [int(info["fy_min"]), int(info["fy_max"])]
        if dash.get("fy") is None:
            # prefer 2025 if in range, else last actual FY, else max FY
            fy_min, fy_max = int(info["fy_min"]), int(info["fy_max"])
            if 2025 >= fy_min and 2025 <= fy_max:
                dash["fy"] = 2025
            else:
                dash["fy"] = int(info["fy_last_actual"]) if info.get("fy_last_actual") else fy_max
        st.session_state["dash_state"] = dash


    else:
        st.info("Upload a file (or use the example toggle) to begin.")
        st.caption("Note: the local ~/Downloads fallback only works on your machine, not on Streamlit Cloud.")


st.markdown("---")
st.caption("Use the left sidebar to navigate pages. Pages appear once data is loaded.")