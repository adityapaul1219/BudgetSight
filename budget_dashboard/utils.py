from __future__ import annotations

import io
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

REQUIRED_COLS = ["bucket", "item", "fy", "value", "yhat", "yhat_lower", "yhat_upper"]

def load_df(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()

    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    elif name.endswith(".pkl"):
        return pd.read_pickle(uploaded_file)

    # default: csv
    # handle both text and bytes
    content = uploaded_file.getvalue()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        return pd.read_csv(io.StringIO(content.decode("utf-8", errors="ignore")))


def validate_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # Normalize strings
    out["bucket"] = out["bucket"].astype(str).str.strip()
    out["item"] = out["item"].astype(str).str.strip()

    # Numeric coercion
    out["fy"] = pd.to_numeric(out["fy"], errors="coerce")
    for c in ["value", "yhat", "yhat_lower", "yhat_upper"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["bucket", "item", "fy"]).copy()
    out["fy"] = out["fy"].astype(int)

    # Detect last actual FY: max FY with any non-null value
    fy_last_actual = int(out.loc[out["value"].notna(), "fy"].max()) if out["value"].notna().any() else int(out["fy"].min())
    fy_last_forecast = int(out.loc[out["yhat"].notna(), "fy"].max()) if out["yhat"].notna().any() else fy_last_actual

    out["is_actual"] = out["fy"] <= fy_last_actual
    out["is_forecast"] = out["fy"] > fy_last_actual

    # Unified display series:
    # - prefer actual value where available
    # - else use yhat (forecast or in-sample prediction)
    out["display_value"] = np.where(out["value"].notna(), out["value"], out["yhat"])

    out["display_value"] = pd.to_numeric(out["display_value"], errors="coerce")


    # For intervals: only meaningful where yhat exists; keep NaN otherwise
    out["display_lower"] = out["yhat_lower"]
    out["display_upper"] = out["yhat_upper"]

    # Store metadata in attrs for convenience
    out.attrs["fy_last_actual"] = fy_last_actual
    out.attrs["fy_last_forecast"] = fy_last_forecast

    # Sort for consistent visuals
    out = out.sort_values(["bucket", "item", "fy"]).reset_index(drop=True)
    return out


def get_meta(df: pd.DataFrame) -> Dict[str, int]:
    fy_last_actual = int(df.attrs.get("fy_last_actual", df.loc[df["value"].notna(), "fy"].max()))
    fy_last_forecast = int(df.attrs.get("fy_last_forecast", df.loc[df["yhat"].notna(), "fy"].max()))
    return {"fy_last_actual": fy_last_actual, "fy_last_forecast": fy_last_forecast}


def describe_df(df: pd.DataFrame) -> Dict[str, int]:
    meta = get_meta(df)
    return {
        "rows": int(len(df)),
        "n_buckets": int(df["bucket"].nunique()),
        "n_items": int(df["item"].nunique()),
        "fy_min": int(df["fy"].min()),
        "fy_max": int(df["fy"].max()),
        "fy_last_actual": meta["fy_last_actual"],
        "fy_last_forecast": meta["fy_last_forecast"],
    }


@st.cache_data(show_spinner=False)
def aggregate_bucket_totals(df: pd.DataFrame) -> pd.DataFrame:
    agg_dict = dict(
        actual_total=("value", "sum"),
        yhat_total=("yhat", "sum"),
        display_total=("display_value", "sum"),
    )
    if "GDP" in df.columns:
        agg_dict["gdp_total"] = ("GDP", "last")

    g = (
        df.groupby(["bucket", "fy"], as_index=False)
        .agg(**agg_dict)
    )
    return g


@st.cache_data(show_spinner=False)
def aggregate_item_series(df: pd.DataFrame, bucket: str, item: str) -> pd.DataFrame:
    d = df[(df["bucket"].str.lower() == bucket.lower()) & (df["item"].str.lower() == item.lower())].copy()
    if d.empty:
        return d
    # ensure one row per FY (sum if duplicates)
    d = (
        d.groupby(["bucket", "item", "fy"], as_index=False)
        .agg(
            value=("value", "sum"),
            yhat=("yhat", "sum"),
            yhat_lower=("yhat_lower", "sum"),
            yhat_upper=("yhat_upper", "sum"),
            display_value=("display_value", "sum"),
        )
        .sort_values("fy")
    )
    return d
