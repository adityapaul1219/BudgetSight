from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from openai import OpenAI

import re


# IMPORTANT: use your existing utils so Copilot matches dashboard logic
from utils import (
    describe_df,
    get_meta,
    aggregate_bucket_totals,
    aggregate_item_series,
)

# ----------------------------
# Shared dashboard state
# ----------------------------

DEFAULT_DASH_STATE: Dict[str, Any] = {
    # global selections
    "fy": None,                 # int
    "fy_range": None,           # [start, end] or None

    # Line Item Explorer selections
    "outlay_bucket": None,      # str
    "outlay_item": None,        # str
    "receipt_item": None,       # str

    # Deficit Explorer selections
    "deficit_outlay_buckets": ["discretionary", "mandatory", "net_interest"],
    "deficit_receipts_bucket": "receipts",

    # optional navigation hint
    "page": None,               # "overview" | "deficit_explorer" | "line_item_explorer" | "home"
}

PAGE_MAP = {
    "home": "app.py",
    "overview": "pages/01_Overview.py",
    "deficit_explorer": "pages/02_Deficit_Explorer.py",
    "line_item_explorer": "pages/03_Line_Item_Explorer.py",
    "copilot": "pages/04_Copilot.py",
}


def _fmt_money(x: float | None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    ax = abs(float(x))
    # Assumes your dataset is in dollars (common in budget tables).
    # If your dataset is in billions, this will still be consistent—just a display choice.
    if ax >= 1e12:
        return f"${x/1e12:,.2f}T"
    if ax >= 1e9:
        return f"${x/1e9:,.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:,.2f}M"
    return f"${x:,.0f}"


def auto_insights(df: pd.DataFrame, fy_start: int, fy_end: int, top_k: int = 8) -> Dict[str, Any]:
    """
    Compute story-like insights from the same underlying data used by the dashboard.
    Returns cards that can be applied as dashboard state.
    """
    fy_start, fy_end = int(fy_start), int(fy_end)
    top_k = int(top_k)

    meta = get_meta(df)
    fy_last_actual = int(meta["fy_last_actual"])

    # Ensure display_value exists (your validate_and_prepare_df creates it)
    d = df.copy()
    if "display_value" not in d.columns:
        d["display_value"] = np.where(d["value"].notna(), d["value"], d["yhat"])

    d["bucket"] = d["bucket"].astype(str).str.strip()
    d["item"] = d["item"].astype(str).str.strip()
    d["fy"] = pd.to_numeric(d["fy"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["bucket", "item", "fy"]).copy()
    d["fy"] = d["fy"].astype(int)

    # Restrict to selected range
    d = d[(d["fy"] >= fy_start) & (d["fy"] <= fy_end)].copy()
    if d.empty:
        return {"fy_start": fy_start, "fy_end": fy_end, "cards": [], "error": "No data in that FY range."}

    # Item-level yearly totals
    g = (
        d.groupby(["bucket", "item", "fy"], as_index=False)["display_value"].sum()
        .sort_values(["bucket", "item", "fy"])
        .copy()
    )
    g["prev"] = g.groupby(["bucket", "item"])["display_value"].shift(1)
    g["yoy_change"] = g["display_value"] - g["prev"]
    g["yoy_pct"] = g["yoy_change"] / g["prev"].replace(0, np.nan)

    # Filter to rows with a valid YoY
    g = g.dropna(subset=["prev", "yoy_change"]).copy()

    # Segment actual vs forecast by FY (matches your dashboard definition)
    g["segment"] = np.where(g["fy"] <= fy_last_actual, "actual", "forecast")

    # Buckets
    buckets = sorted(g["bucket"].unique().tolist())
    outlay_buckets = [b for b in buckets if b.lower() != "receipts"]
    receipts_bucket = "receipts" if any(b.lower() == "receipts" for b in buckets) else None

    cards: List[Dict[str, Any]] = []
    cid = 1

    def add_card(
        title: str,
        subtitle: str,
        evidence: List[Dict[str, Any]],
        apply_updates: Dict[str, Any],
        page_key: str,
        tag: str,
        why_it_matters: str,
        confidence_note: str,
    ):
        nonlocal cid, cards
        cards.append(
            {
                "id": f"ins_{cid:02d}_{tag}",
                "title": title,
                "subtitle": subtitle,
                "why_it_matters": why_it_matters,
                "confidence_note": confidence_note,
                "evidence": evidence,
                "apply_updates": apply_updates,
                "page_key": page_key,
            }
        )
        cid += 1

    def confidence_note(kind: str, is_forecast: bool, fy: int) -> str:
        base = "Computed directly from the dataset shown in this dashboard (no external sources)."
        if is_forecast:
            return (
                f"{base} Uses Prophet-based forecast values for FY {fy} (not a causal claim; treat as scenario guidance)."
            )
        return (
            f"{base} Uses actual values where available up to the last-actual FY; this highlights correlation, not causation."
        )
    
    def why_text(kind: str, direction: str | None = None, is_forecast: bool = False) -> str:
        """
        kind: 'outlay' | 'receipts' | 'deficit'
        direction: 'inc' | 'dec' | None
        """
        if kind == "deficit":
            if direction == "widen":
                return (
                    "A sharp widening of the deficit usually means more borrowing, which can increase future interest costs "
                    "and reduce flexibility for other priorities."
                )
            return (
                "A large improvement in the deficit often signals either spending restraint or revenue recovery—useful for "
                "spotting turning points in the fiscal outlook."
            )
    
        if kind == "receipts":
            if is_forecast:
                return (
                    "Because receipts drive how much the government can fund without borrowing, a strong forecast move here "
                    "can materially change the deficit path—worth stress-testing assumptions."
                )
            if direction == "inc":
                return (
                    "A major jump in this revenue source can narrow the deficit; it’s a key clue when diagnosing why fiscal "
                    "balance improved in that year."
                )
            return (
                "A major drop in this revenue source can widen deficits; it often aligns with recessions, policy changes, or "
                "tax-base shifts."
            )
    
        # outlays
        if is_forecast:
            return (
                "If this forecast holds, it becomes a key driver of future spending and deficits—useful for prioritizing which "
                "program areas to monitor and debate."
            )
        if direction == "inc":
            return (
                "This was one of the biggest year-over-year drivers of federal spending in this period—helpful for explaining "
                "why total outlays jumped that year."
            )
        return (
            "This was a major year-over-year pullback in spending—helpful for explaining why total outlays fell or grew more "
            "slowly in that year."
        )

    
    # ---- 1) Outlay bucket movers (actual) + (forecast)
    for b in outlay_buckets:
        gb = g[g["bucket"].str.lower() == b.lower()].copy()
        if gb.empty:
            continue

        # Actual movers
        ga = gb[gb["segment"] == "actual"].copy()
        if not ga.empty:
            row_inc = ga.sort_values("yoy_change", ascending=False).head(1).iloc[0]
            row_dec = ga.sort_values("yoy_change", ascending=True).head(1).iloc[0]

            for direction, r in [("inc", row_inc), ("dec", row_dec)]:
                fy = int(r["fy"])
                yoy = float(r["yoy_change"])
                item = str(r["item"])
                val = float(r["display_value"])
                pct = None if pd.isna(r["yoy_pct"]) else float(r["yoy_pct"])

                title = "📈" if direction == "inc" else "📉"
                title += f" {b}: {item} ({fy})"

                subtitle = (
                    f"Year-over-year change: **{_fmt_money(yoy)}** "
                    + (f"({pct*100:.1f}%)" if pct is not None else "")
                )

                evidence = [
                    {"bucket": b, "item": item, "fy": fy, "value": _fmt_money(val), "yoy_change": _fmt_money(yoy)},
                ]

                apply_updates = {
                    "fy": fy,
                    "fy_range": [max(fy_start, fy - 10), min(fy_end, fy + 10)],
                    "outlay_bucket": b,
                    "outlay_item": item,
                    "receipt_item": None,
                    "page": "line_item_explorer",
                }

                add_card(
                    title, subtitle, evidence, apply_updates,
                    "line_item_explorer", f"out_{b}_{direction}",
                    why_it_matters=why_text("outlay", direction=direction, is_forecast=False),
                    confidence_note=confidence_note("outlay", is_forecast=False, fy=fy),
                )

        # Forecast movers (only the biggest increase; decreases often get weird with baseline effects)
        gf = gb[gb["segment"] == "forecast"].copy()
        if not gf.empty:
            r = gf.sort_values("yoy_change", ascending=False).head(1).iloc[0]
            fy = int(r["fy"])
            yoy = float(r["yoy_change"])
            item = str(r["item"])
            val = float(r["display_value"])
            pct = None if pd.isna(r["yoy_pct"]) else float(r["yoy_pct"])

            title = f"🔮 Forecast {b}: {item} ({fy})"
            subtitle = (
                f"Forecast YoY change: **{_fmt_money(yoy)}** "
                + (f"({pct*100:.1f}%)" if pct is not None else "")
            )
            evidence = [
                {"bucket": b, "item": item, "fy": fy, "forecast_value": _fmt_money(val), "forecast_yoy": _fmt_money(yoy)},
            ]
            apply_updates = {
                "fy": fy,
                "fy_range": [max(fy_start, fy - 10), min(fy_end, fy + 10)],
                "outlay_bucket": b,
                "outlay_item": item,
                "receipt_item": None,
                "page": "line_item_explorer",
            }
            add_card(
                title, subtitle, evidence, apply_updates,
                "line_item_explorer", f"fout_{b}",
                why_it_matters=why_text("outlay", direction="inc", is_forecast=True),
                confidence_note=confidence_note("outlay", is_forecast=True, fy=fy),
            )

    # ---- 2) Receipts movers (actual + forecast)
    if receipts_bucket:
        gr = g[g["bucket"].str.lower() == "receipts"].copy()
        if not gr.empty:
            # Actual
            gra = gr[gr["segment"] == "actual"].copy()
            if not gra.empty:
                row_inc = gra.sort_values("yoy_change", ascending=False).head(1).iloc[0]
                row_dec = gra.sort_values("yoy_change", ascending=True).head(1).iloc[0]
                for direction, r in [("inc", row_inc), ("dec", row_dec)]:
                    fy = int(r["fy"])
                    yoy = float(r["yoy_change"])
                    item = str(r["item"])
                    val = float(r["display_value"])
                    pct = None if pd.isna(r["yoy_pct"]) else float(r["yoy_pct"])

                    title = ("📈" if direction == "inc" else "📉") + f" Receipts: {item} ({fy})"
                    subtitle = (
                        f"Year-over-year change: **{_fmt_money(yoy)}** "
                        + (f"({pct*100:.1f}%)" if pct is not None else "")
                    )
                    evidence = [
                        {"bucket": "receipts", "item": item, "fy": fy, "value": _fmt_money(val), "yoy_change": _fmt_money(yoy)},
                    ]
                    apply_updates = {
                        "fy": fy,
                        "fy_range": [max(fy_start, fy - 10), min(fy_end, fy + 10)],
                        "outlay_bucket": None,
                        "outlay_item": None,
                        "receipt_item": item,
                        "page": "line_item_explorer",
                    }
                    add_card(
                        title, subtitle, evidence, apply_updates,
                        "line_item_explorer", f"rec_{direction}",
                        why_it_matters=why_text("receipts", direction=direction, is_forecast=False),
                        confidence_note=confidence_note("receipts", is_forecast=False, fy=fy),
                    )

            # Forecast
            grf = gr[gr["segment"] == "forecast"].copy()
            if not grf.empty:
                r = grf.sort_values("yoy_change", ascending=False).head(1).iloc[0]
                fy = int(r["fy"])
                yoy = float(r["yoy_change"])
                item = str(r["item"])
                val = float(r["display_value"])
                pct = None if pd.isna(r["yoy_pct"]) else float(r["yoy_pct"])

                title = f"🔮 Forecast receipts: {item} ({fy})"
                subtitle = (
                    f"Forecast YoY change: **{_fmt_money(yoy)}** "
                    + (f"({pct*100:.1f}%)" if pct is not None else "")
                )
                evidence = [
                    {"bucket": "receipts", "item": item, "fy": fy, "forecast_value": _fmt_money(val), "forecast_yoy": _fmt_money(yoy)},
                ]
                apply_updates = {
                    "fy": fy,
                    "fy_range": [max(fy_start, fy - 10), min(fy_end, fy + 10)],
                    "outlay_bucket": None,
                    "outlay_item": None,
                    "receipt_item": item,
                    "page": "line_item_explorer",
                }
                add_card(
                    title, subtitle, evidence, apply_updates,
                    "line_item_explorer", "frec",
                    why_it_matters=why_text("receipts", direction="inc", is_forecast=True),
                    confidence_note=confidence_note("receipts", is_forecast=True, fy=fy),
                )

    # ---- 3) Deficit “biggest widen” / “biggest improve” (uses your bucket totals)
    try:
        agg = aggregate_bucket_totals(df).copy()
        agg = agg[(agg["fy"] >= fy_start) & (agg["fy"] <= fy_end)].copy()

        out_set = {"discretionary", "mandatory", "net_interest"}
        outlays = (
            agg[agg["bucket"].astype(str).str.lower().isin(out_set)]
            .groupby("fy", as_index=False)["display_total"].sum()
            .rename(columns={"display_total": "outlays"})
        )
        receipts = (
            agg[agg["bucket"].astype(str).str.lower() == "receipts"]
            .groupby("fy", as_index=False)["display_total"].sum()
            .rename(columns={"display_total": "receipts"})
        )
        merged = outlays.merge(receipts, on="fy", how="outer").sort_values("fy").copy()
        merged["outlays"] = merged["outlays"].fillna(0.0)
        merged["receipts"] = merged["receipts"].fillna(0.0)
        merged["deficit"] = merged["outlays"] - merged["receipts"]
        merged["deficit_yoy"] = merged["deficit"] - merged["deficit"].shift(1)
        merged = merged.dropna(subset=["deficit_yoy"]).copy()

        if not merged.empty:
            widen = merged.sort_values("deficit_yoy", ascending=False).head(1).iloc[0]
            improve = merged.sort_values("deficit_yoy", ascending=True).head(1).iloc[0]

            for tag, r in [("widen", widen), ("improve", improve)]:
                fy = int(r["fy"])
                dy = float(r["deficit_yoy"])
                deficit = float(r["deficit"])
                title = ("⚠️ Deficit widened most" if tag == "widen" else "✅ Deficit improved most") + f" ({fy})"
                subtitle = f"YoY change in deficit: **{_fmt_money(dy)}**; deficit level: {_fmt_money(deficit)}"
                evidence = [
                    {"fy": fy, "outlays": _fmt_money(float(r["outlays"])), "receipts": _fmt_money(float(r["receipts"])),
                     "deficit": _fmt_money(deficit), "deficit_yoy": _fmt_money(dy)}
                ]
                apply_updates = {
                    "fy": fy,
                    "fy_range": [max(fy_start, fy - 10), min(fy_end, fy + 10)],
                    "deficit_outlay_buckets": ["discretionary", "mandatory", "net_interest"],
                    "deficit_receipts_bucket": "receipts",
                    "page": "deficit_explorer",
                }
                is_f = fy > fy_last_actual
                
                add_card(
                    title, subtitle, evidence, apply_updates,
                    "deficit_explorer", f"def_{tag}",
                    why_it_matters=why_text("deficit", direction=tag, is_forecast=is_f),
                    confidence_note=confidence_note("deficit", is_forecast=is_f, fy=fy),
                )
    except Exception:
        # If GDP/display_total missing etc., just skip deficit insights.
        pass

    # Keep the most “wow” ones first (deficit, receipts, then outlays)
    # and cap count.
    # (You can tweak ordering later.)
    cards = cards[:top_k]
    # cards = cards[: max(6, top_k)]

    return {"fy_start": fy_start, "fy_end": fy_end, "fy_last_actual": fy_last_actual, "cards": cards}


def ensure_dash_state(session_state: Dict[str, Any]) -> Dict[str, Any]:
    if "dash_state" not in session_state or not isinstance(session_state["dash_state"], dict):
        session_state["dash_state"] = dict(DEFAULT_DASH_STATE)
    else:
        for k, v in DEFAULT_DASH_STATE.items():
            session_state["dash_state"].setdefault(k, v)
    return session_state["dash_state"]


def _norm(x: Any) -> str:
    return str(x).strip()


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return int(x)
    except Exception:
        return None


# ----------------------------
# Tool implementations
# ----------------------------

def tool_get_metadata(df: pd.DataFrame) -> Dict[str, Any]:
    info = describe_df(df)
    meta = get_meta(df)
    buckets = sorted(df["bucket"].astype(str).str.strip().unique().tolist())
    return {
        "rows": info["rows"],
        "fy_min": info["fy_min"],
        "fy_max": info["fy_max"],
        "fy_last_actual": meta["fy_last_actual"],
        "fy_last_forecast": meta["fy_last_forecast"],
        "buckets": buckets,
        "n_items": info["n_items"],
    }


def tool_list_items(df: pd.DataFrame, bucket: str) -> Dict[str, Any]:
    bucket = _norm(bucket)
    items = (
        df.loc[df["bucket"].astype(str).str.lower() == bucket.lower(), "item"]
        .astype(str).str.strip().unique().tolist()
    )
    items = sorted(items)
    return {"bucket": bucket, "count": len(items), "items": items}


def tool_get_bucket_totals(df: pd.DataFrame, fy_start: int, fy_end: int, buckets: Optional[List[str]] = None) -> Dict[str, Any]:
    agg = aggregate_bucket_totals(df).copy()
    fy_start, fy_end = int(fy_start), int(fy_end)
    view = agg[(agg["fy"] >= fy_start) & (agg["fy"] <= fy_end)].copy()

    if buckets:
        allowed = {_norm(b).lower() for b in buckets}
        view = view[view["bucket"].astype(str).str.lower().isin(allowed)]

    view = view.sort_values(["fy", "bucket"])
    rows = []
    for _, r in view.iterrows():
        rows.append({
            "bucket": str(r["bucket"]),
            "fy": int(r["fy"]),
            "display_total": _safe_float(r.get("display_total")),
            "actual_total": _safe_float(r.get("actual_total")),
            "yhat_total": _safe_float(r.get("yhat_total")),
            # GDP may be missing depending on your df; keep None if so
            "gdp_total": _safe_float(r.get("gdp_total")),
        })
    return {"fy_start": fy_start, "fy_end": fy_end, "buckets_filter": buckets, "rows": rows}


def tool_get_snapshot_fy(df: pd.DataFrame, fy: int) -> Dict[str, Any]:
    fy = int(fy)
    agg = aggregate_bucket_totals(df).copy()
    s = agg[agg["fy"] == fy].copy()
    if s.empty:
        return {"fy": fy, "error": "No data for that FY."}

    outlays = s[s["bucket"].astype(str).str.lower() != "receipts"]
    receipts = s[s["bucket"].astype(str).str.lower() == "receipts"]

    outlays_total = float(outlays["display_total"].sum(skipna=True))
    receipts_total = float(receipts["display_total"].sum(skipna=True))

    outlay_by_bucket = (
        outlays.groupby("bucket", as_index=False)["display_total"].sum()
        .rename(columns={"display_total": "total"})
        .sort_values("total", ascending=False)
    )

    # NOTE: receipts are by bucket here; line-items require the raw df (which you have).
    return {
        "fy": fy,
        "outlays_total": outlays_total,
        "receipts_total": receipts_total,
        "outlay_by_bucket": [{"bucket": str(r["bucket"]), "total": float(r["total"])} for _, r in outlay_by_bucket.iterrows()],
        "note": "For receipts by item, ask: 'show receipts item series <name>' or use Line Item Explorer.",
    }


def tool_get_item_series(
    df: pd.DataFrame,
    bucket: str,
    item: str,
    fy_start: Optional[int] = None,
    fy_end: Optional[int] = None,
    max_points: int = 25,   # NEW
) -> Dict[str, Any]:
    bucket = _norm(bucket)
    item = _norm(item)
    max_points = int(max_points) if max_points is not None else 25
    max_points = max(5, min(max_points, 200))  # guardrails

    s = aggregate_item_series(df, bucket=bucket, item=item).copy()
    if s.empty:
        return {"bucket": bucket, "item": item, "error": "No data found for that (bucket,item)."}

    if fy_start is not None:
        s = s[s["fy"] >= int(fy_start)]
    if fy_end is not None:
        s = s[s["fy"] <= int(fy_end)]

    s = s.sort_values("fy").copy()
    s["yoy_change"] = s["display_value"] - s["display_value"].shift(1)
    s["yoy_pct"] = np.where(
        s["display_value"].shift(1).abs() > 1e-9,
        s["yoy_change"] / s["display_value"].shift(1),
        np.nan,
    )

    # ✅ SPEED: If user did not ask for a specific range, return only last N points
    if fy_start is None and fy_end is None and len(s) > max_points:
        s = s.tail(max_points).copy()

    rows = []
    for _, r in s.iterrows():
        rows.append({
            "fy": int(r["fy"]),
            "value": _safe_float(r.get("value")),
            "yhat": _safe_float(r.get("yhat")),
            "yhat_lower": _safe_float(r.get("yhat_lower")),
            "yhat_upper": _safe_float(r.get("yhat_upper")),
            "display_value": _safe_float(r.get("display_value")),
            "yoy_change": _safe_float(r.get("yoy_change")),
            "yoy_pct": _safe_float(r.get("yoy_pct")),
        })

    return {
        "bucket": bucket,
        "item": item,
        "fy_start": fy_start,
        "fy_end": fy_end,
        "max_points": max_points,   # NEW (helps transparency)
        "rows": rows
    }


def tool_get_deficit_series(
    df: pd.DataFrame,
    fy_start: int,
    fy_end: int,
    outlay_buckets: Optional[List[str]] = None,
    receipts_bucket: str = "receipts",
    max_points: int = 35,   # NEW (default slightly larger)
) -> Dict[str, Any]:
    fy_start, fy_end = int(fy_start), int(fy_end)
    outlay_buckets = outlay_buckets or ["discretionary", "mandatory", "net_interest"]
    receipts_bucket = _norm(receipts_bucket)

    max_points = int(max_points) if max_points is not None else 35
    max_points = max(5, min(max_points, 200))

    agg = aggregate_bucket_totals(df).copy()
    view = agg[(agg["fy"] >= fy_start) & (agg["fy"] <= fy_end)].copy()

    out_set = {_norm(b).lower() for b in outlay_buckets}
    outlays = (
        view[view["bucket"].astype(str).str.lower().isin(out_set)]
        .groupby("fy", as_index=False)["display_total"].sum()
        .rename(columns={"display_total": "outlays"})
    )
    receipts = (
        view[view["bucket"].astype(str).str.lower() == receipts_bucket.lower()]
        .groupby("fy", as_index=False)["display_total"].sum()
        .rename(columns={"display_total": "receipts"})
    )

    gdps = (
        view[view["bucket"].astype(str).str.lower() == receipts_bucket.lower()]
        .groupby("fy", as_index=False)["gdp_total"].last()
        .rename(columns={"gdp_total": "GDP"})
    )

    merged = (
        outlays.merge(receipts, on="fy", how="outer")
        .merge(gdps, on="fy", how="left")
        .sort_values("fy")
        .copy()
    )
    merged["outlays"] = merged["outlays"].fillna(0.0)
    merged["receipts"] = merged["receipts"].fillna(0.0)
    merged["deficit"] = merged["outlays"] - merged["receipts"]
    merged["deficit_over_gdp"] = merged["deficit"] / merged["GDP"]

    # ✅ SPEED: if the requested window is huge, keep only last N FYs by default
    # (In practice 1980-2035 is only ~56 rows, but this still keeps payload small.)
    if len(merged) > max_points:
        merged = merged.tail(max_points).copy()

    rows = []
    for _, r in merged.iterrows():
        rows.append({
            "fy": int(r["fy"]),
            "outlays": _safe_float(r.get("outlays")) or 0.0,
            "receipts": _safe_float(r.get("receipts")) or 0.0,
            "deficit": _safe_float(r.get("deficit")) or 0.0,
            "GDP": _safe_float(r.get("GDP")),
            "deficit_over_gdp": _safe_float(r.get("deficit_over_gdp")),
        })

    return {
        "fy_start": fy_start,
        "fy_end": fy_end,
        "max_points": max_points,     # NEW
        "outlay_buckets": outlay_buckets,
        "receipts_bucket": receipts_bucket,
        "rows": rows,
        "note": "GDP may be null if your source df lacks a GDP column.",
    }


def tool_suggest_insights(df: pd.DataFrame, bucket: str, fy_start: int, fy_end: int, top_k: int = 5) -> Dict[str, Any]:
    """
    Find top YoY movers for a bucket across the range using item series.
    Uses actual values where available, else display_value.
    """
    bucket = _norm(bucket)
    fy_start, fy_end = int(fy_start), int(fy_end)
    top_k = int(top_k)

    meta = get_meta(df)
    fy_last_actual = meta["fy_last_actual"]

    sub = df[df["bucket"].astype(str).str.lower() == bucket.lower()].copy()
    sub = sub[sub["fy"].between(fy_start, fy_end)].copy()
    if sub.empty:
        return {"bucket": bucket, "error": "No rows in that bucket/range."}

    # build per-item per-fy totals
    g = (
        sub.groupby(["item", "fy"], as_index=False)
        .agg(value=("value", "sum"), yhat=("yhat", "sum"), display_value=("display_value", "sum"))
        .sort_values(["item", "fy"])
    )

    # actual movers (<= last actual FY)
    a = g[g["fy"] <= fy_last_actual].copy()
    a["prev"] = a.groupby("item")["value"].shift(1)
    a["chg"] = a["value"] - a["prev"]
    a = a.dropna(subset=["prev", "chg"])
    top_a_inc = a.sort_values("chg", ascending=False).head(top_k)[["item", "fy", "chg"]]
    top_a_dec = a.sort_values("chg", ascending=True).head(top_k)[["item", "fy", "chg"]]

    # forecast movers (> last actual FY) using yhat
    f = g[g["fy"] > fy_last_actual].copy()
    f["prev"] = f.groupby("item")["yhat"].shift(1)
    f["chg"] = f["yhat"] - f["prev"]
    f = f.dropna(subset=["prev", "chg"])
    top_f_inc = f.sort_values("chg", ascending=False).head(top_k)[["item", "fy", "chg"]]
    top_f_dec = f.sort_values("chg", ascending=True).head(top_k)[["item", "fy", "chg"]]

    def rows(df_: pd.DataFrame) -> List[Dict[str, Any]]:
        out = []
        for _, r in df_.iterrows():
            out.append({"item": str(r["item"]), "fy": int(r["fy"]), "chg": float(r["chg"])})
        return out

    storyline = []
    if not top_a_inc.empty:
        storyline.append(f"Biggest actual YoY increase: {top_a_inc.iloc[0]['item']} (FY {int(top_a_inc.iloc[0]['fy'])}).")
    if not top_f_inc.empty:
        storyline.append(f"Biggest forecast YoY increase: {top_f_inc.iloc[0]['item']} (FY {int(top_f_inc.iloc[0]['fy'])}).")

    return {
        "bucket": bucket,
        "fy_range": [fy_start, fy_end],
        "fy_last_actual": fy_last_actual,
        "top_actual_yoy_increases": rows(top_a_inc),
        "top_actual_yoy_decreases": rows(top_a_dec),
        "top_forecast_yoy_increases": rows(top_f_inc),
        "top_forecast_yoy_decreases": rows(top_f_dec),
        "storyline": " ".join(storyline) if storyline else "No movers found (check coverage).",
    }

def tool_get_top_outlay_items(df: pd.DataFrame, fy: int, k: int = 5) -> Dict[str, Any]:
    fy = int(fy)
    k = int(k)

    d = df.copy()
    d["bucket"] = d["bucket"].astype(str).str.strip()
    d["item"] = d["item"].astype(str).str.strip()
    d["fy"] = pd.to_numeric(d["fy"], errors="coerce")
    d = d.dropna(subset=["fy"]).copy()
    d["fy"] = d["fy"].astype(int)

    d = d[d["fy"] == fy].copy()
    if d.empty:
        return {"fy": fy, "rows": [], "error": "No data for that FY."}

    # Outlays = everything except receipts
    d = d[d["bucket"].str.lower() != "receipts"].copy()
    if d.empty:
        return {"fy": fy, "rows": [], "error": "No outlay rows for that FY."}

    # Use unified display_value if present; else compute it
    if "display_value" not in d.columns:
        d["display_value"] = np.where(d["value"].notna(), d["value"], d["yhat"])
    d["display_value"] = pd.to_numeric(d["display_value"], errors="coerce")
    d = d.dropna(subset=["display_value"]).copy()

    g = (
        d.groupby(["bucket", "item"], as_index=False)["display_value"]
        .sum()
        .sort_values("display_value", ascending=False)
    )

    top = g.head(k).copy()
    rows = [
        {"rank": i + 1, "bucket": str(r["bucket"]), "item": str(r["item"]), "value": float(r["display_value"])}
        for i, (_, r) in enumerate(top.iterrows())
    ]
    best = rows[0] if rows else None
    return {"fy": fy, "k": k, "top_item": best, "rows": rows}


def tool_get_top_yoy_change(
    df: pd.DataFrame,
    fy: int,
    scope: str = "outlays",   # outlays | receipts | all
    metric: str = "change",   # change | pct
    k: int = 5,
) -> Dict[str, Any]:
    fy = int(fy)
    k = int(k)
    scope = (scope or "outlays").strip().lower()
    metric = (metric or "change").strip().lower()

    d = df.copy()
    d["bucket"] = d["bucket"].astype(str).str.strip()
    d["item"] = d["item"].astype(str).str.strip()
    d["fy"] = pd.to_numeric(d["fy"], errors="coerce")
    d = d.dropna(subset=["fy"]).copy()
    d["fy"] = d["fy"].astype(int)

    if "display_value" not in d.columns:
        d["display_value"] = np.where(d["value"].notna(), d["value"], d["yhat"])
    d["display_value"] = pd.to_numeric(d["display_value"], errors="coerce")
    d = d.dropna(subset=["display_value"]).copy()

    # scope filter
    if scope == "outlays":
        d = d[d["bucket"].str.lower() != "receipts"]
    elif scope == "receipts":
        d = d[d["bucket"].str.lower() == "receipts"]
    # all => no filter

    # Need FY and FY-1
    sub = d[d["fy"].isin([fy - 1, fy])].copy()
    if sub.empty:
        return {"error": f"No data for FY {fy} and FY {fy-1} under scope={scope}."}

    g = (
        sub.groupby(["bucket", "item", "fy"], as_index=False)["display_value"].sum()
        .sort_values(["bucket", "item", "fy"])
        .copy()
    )

    g["prev"] = g.groupby(["bucket", "item"])["display_value"].shift(1)
    g["yoy_change"] = g["display_value"] - g["prev"]
    g["yoy_pct"] = g["yoy_change"] / g["prev"].replace(0, np.nan)

    # keep only target FY rows with a previous year
    g = g[(g["fy"] == fy) & g["prev"].notna()].copy()
    g = g.replace([np.inf, -np.inf], np.nan)

    if g.empty:
        return {"error": f"No YoY rows available for FY {fy} under scope={scope}."}

    if metric == "pct":
        g = g.dropna(subset=["yoy_pct"])
        g2 = g.sort_values("yoy_pct", ascending=False).head(k)
    else:
        g2 = g.sort_values("yoy_change", ascending=False).head(k)

    rows = []
    for i, (_, r) in enumerate(g2.iterrows(), start=1):
        rows.append({
            "rank": i,
            "bucket": str(r["bucket"]),
            "item": str(r["item"]),
            "fy": int(r["fy"]),
            "value": float(r["display_value"]),
            "yoy_change": float(r["yoy_change"]),
            "yoy_pct": None if pd.isna(r["yoy_pct"]) else float(r["yoy_pct"]),
        })

    return {"fy": fy, "scope": scope, "metric": metric, "top": rows[0] if rows else None, "rows": rows}


# ----------------------------
# Tools spec for OpenAI Responses tool-calling
# ----------------------------

TOOLS = [
    {
        "type": "function",
        "name": "get_metadata",
        "description": "Get dataset metadata (FY range, last actual FY, buckets).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "list_items",
        "description": "List items available in a bucket.",
        "parameters": {
            "type": "object",
            "properties": {"bucket": {"type": "string"}},
            "required": ["bucket"],
        },
    },
    {
        "type": "function",
        "name": "get_snapshot_fy",
        "description": "FY snapshot: totals + outlays by bucket.",
        "parameters": {
            "type": "object",
            "properties": {"fy": {"type": "integer"}},
            "required": ["fy"],
        },
    },
    {
        "type": "function",
        "name": "get_bucket_totals",
        "description": "Bucket totals by FY within a range; optional buckets filter.",
        "parameters": {
            "type": "object",
            "properties": {
                "fy_start": {"type": "integer"},
                "fy_end": {"type": "integer"},
                "buckets": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["fy_start", "fy_end"],
        },
    },
    {
      "type": "function",
      "name": "get_item_series",
      "description": "Time series for a (bucket,item): actual/forecast/display + YoY.",
      "parameters": {
        "type": "object",
        "properties": {
          "bucket": {"type": "string"},
          "item": {"type": "string"},
          "fy_start": {"type": "integer"},
          "fy_end": {"type": "integer"},
          "max_points": {"type": "integer"}   # NEW
        },
        "required": ["bucket", "item"]
      }
    },
    {
      "type": "function",
      "name": "get_deficit_series",
      "description": "Deficit series (outlays - receipts) and deficit/GDP if GDP exists.",
      "parameters": {
        "type": "object",
        "properties": {
          "fy_start": {"type": "integer"},
          "fy_end": {"type": "integer"},
          "outlay_buckets": {"type": "array", "items": {"type": "string"}},
          "receipts_bucket": {"type": "string"},
          "max_points": {"type": "integer"}   # NEW
        },
        "required": ["fy_start", "fy_end"]
      }
    },
    {
        "type": "function",
        "name": "suggest_insights",
        "description": "Top YoY movers within a bucket across a FY range.",
        "parameters": {
            "type": "object",
            "properties": {
                "bucket": {"type": "string"},
                "fy_start": {"type": "integer"},
                "fy_end": {"type": "integer"},
                "top_k": {"type": "integer"},
            },
            "required": ["bucket", "fy_start", "fy_end"],
        },
    },
    {
        "type": "function",
        "name": "get_dashboard_state",
        "description": "Return dash_state and known pages.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "set_dashboard_state",
        "description": "Update dash_state keys (FY, FY range, selections).",
        "parameters": {
            "type": "object",
            "properties": {"updates": {"type": "object"}},
            "required": ["updates"],
        },
    },
    {
        "type": "function",
        "name": "navigate",
        "description": "Request navigation to a page key.",
        "parameters": {
            "type": "object",
            "properties": {"page_key": {"type": "string"}},
            "required": ["page_key"],
        },
    },
    {
    "type": "function",
    "name": "auto_insights",
    "description": "Generate story-like insight cards with one-click apply updates for the dashboard.",
    "parameters": {
        "type": "object",
        "properties": {
            "fy_start": {"type": "integer"},
            "fy_end": {"type": "integer"},
            "top_k": {"type": "integer"},
        },
        "required": ["fy_start", "fy_end"],
    },
    },
    {
      "type": "function",
      "name": "get_top_outlay_items",
      "description": "Get the top K outlay line items (across all outlay buckets) for a given FY.",
      "parameters": {
        "type": "object",
        "properties": {
          "fy": {"type": "integer"},
          "k": {"type": "integer"},
        },
        "required": ["fy"]
     },
    }, 
    {
      "type": "function",
      "name": "get_top_yoy_change",
      "description": "Find the item with the highest year-over-year change (absolute $ change or % change) for a given FY across outlays/receipts/all. If FY is beyond last actual, compute YoY using display_value (forecast) and label it as forecast-based.",
      "parameters": {
        "type": "object",
        "properties": {
          "fy": {"type": "integer", "description": "Target fiscal year (e.g., 2021)."},
          "scope": {"type": "string", "description": "outlays|receipts|all. Default outlays."},
          "metric": {"type": "string", "description": "change|pct. Default change."},
          "k": {"type": "integer", "description": "How many top rows to return. Default 5."}
        },
        "required": ["fy"]
      }
    },

    
]


KNOWN_TOOL_NAMES = {t["name"] for t in TOOLS if t.get("type") == "function" and t.get("name")}

def run_tool(tool_name: str, args: Dict[str, Any], df: pd.DataFrame, session_state: Dict[str, Any]) -> Dict[str, Any]:
    if session_state.get("copilot_debug"):
        print("run_tool called:", tool_name, "args:", args)

    if tool_name not in KNOWN_TOOL_NAMES:
        return {"error": f"Unknown tool: {tool_name}", "known_tools": sorted(KNOWN_TOOL_NAMES)}
        
    dash = ensure_dash_state(session_state)

    if tool_name == "get_metadata":
        return tool_get_metadata(df)
    if tool_name == "list_items":
        return tool_list_items(df, **args)
    if tool_name == "get_snapshot_fy":
        return tool_get_snapshot_fy(df, **args)
    if tool_name == "get_bucket_totals":
        return tool_get_bucket_totals(df, **args)
    if tool_name == "get_item_series":
        return tool_get_item_series(df, **args)
    if tool_name == "get_deficit_series":
        return tool_get_deficit_series(df, **args)
    if tool_name == "suggest_insights":
        return tool_suggest_insights(df, **args)

    if tool_name == "get_dashboard_state":
        return {"dash_state": dash, "pages": sorted(PAGE_MAP.keys())}

    if tool_name == "set_dashboard_state":
        updates = args.get("updates", {}) or {}
        if not isinstance(updates, dict):
            return {"error": "updates must be a dict/object."}

        for k, v in updates.items():
            if k in DEFAULT_DASH_STATE:
                dash[k] = v

        # mirror into existing page session keys (Line Item Explorer) for immediate effect
        if dash.get("outlay_bucket") is not None:
            session_state["sel_outlay_bucket"] = dash["outlay_bucket"]
        if dash.get("outlay_item") is not None:
            session_state["sel_outlay_item"] = dash["outlay_item"]
        if dash.get("receipt_item") is not None:
            session_state["sel_receipt_item"] = dash["receipt_item"]

        session_state["dash_state"] = dash
        return {"ok": True, "dash_state": dash}

    if tool_name == "navigate":
        page_key = _norm(args.get("page_key", ""))
        if page_key not in PAGE_MAP:
            return {"error": f"Unknown page_key={page_key}. Valid={sorted(PAGE_MAP.keys())}"}
        session_state["copilot_nav_to"] = PAGE_MAP[page_key]
        dash["page"] = page_key
        session_state["dash_state"] = dash
        return {"ok": True, "page_path": PAGE_MAP[page_key]}

    if tool_name == "auto_insights":
        return auto_insights(df, **args)

    if tool_name == "get_top_outlay_items":
        return tool_get_top_outlay_items(df, **args)


    if tool_name == "get_top_yoy_change":
        return tool_get_top_yoy_change(df, **args)

    return {"error": f"Unknown tool: {tool_name}"}



def _o_get(o, k, default=None):
    # Works for dict or SDK object
    if isinstance(o, dict):
        return o.get(k, default)
    return getattr(o, k, default)

def _is_tool_call(o) -> bool:
    t = _o_get(o, "type")
    return t in ("function_call", "tool_call")  # handle both

def _tool_name(o) -> str | None:
    # Most common: o.name
    name = _o_get(o, "name")
    if name:
        return str(name)

    # Some SDK shapes nest function info
    fn = _o_get(o, "function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    if fn is not None:
        fn_name = getattr(fn, "name", None)
        if fn_name:
            return str(fn_name)

    return None

def _tool_args(o):
    # Most common: o.arguments (string JSON or dict)
    args = _o_get(o, "arguments")
    if args is not None:
        return args

    # Nested shapes: o.function.arguments
    fn = _o_get(o, "function")
    if isinstance(fn, dict) and "arguments" in fn:
        return fn["arguments"]
    if fn is not None:
        return getattr(fn, "arguments", None)

    return None

def _tool_call_id(o) -> str | None:
    cid = _o_get(o, "call_id")
    return str(cid) if cid else None


def _extract_response_text(resp) -> str:
    """
    Robustly extract assistant text from a Responses API response across SDK/model variations.
    Handles:
      - resp.output_text shortcut
      - top-level output items with text
      - message content blocks with text
    """
    # 1) Shortcut
    txt = (getattr(resp, "output_text", None) or "").strip()
    if txt:
        return txt

    parts: List[str] = []
    out = getattr(resp, "output", None) or []

    # print([(o.type, getattr(o,'name',None)) for o in resp.output])

    for item in out:
        itype = getattr(item, "type", None)

        # 2) Some SDK builds/models return top-level "output_text"
        if itype in ("output_text", "text"):
            t = getattr(item, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
            continue

        # 3) Message container
        if itype == "message":
            content = getattr(item, "content", None) or []
            for c in content:
                # dict-like
                if isinstance(c, dict):
                    ctype = c.get("type")
                    if ctype in ("output_text", "text") and c.get("text"):
                        parts.append(str(c["text"]).strip())
                    continue

                # object-like
                ctype = getattr(c, "type", None)
                ctext = getattr(c, "text", None)

                if ctype in ("output_text", "text") and isinstance(ctext, str) and ctext.strip():
                    parts.append(ctext.strip())
                    continue

                # Fallbacks seen in some SDK builds
                for attr in ("value", "content"):
                    v = getattr(c, attr, None)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                        break

    return "\n".join([p for p in parts if p]).strip()





    
def _classify_request(user_text: str) -> str:
    t = (user_text or "").strip().lower()
    if re.search(r"\b(set|select|open|go to|navigate|switch|reset)\b", t):
        return "short"
    if re.search(r"\b(why|how|explain|compare|analy(ze|sis)|summari(ze|se)|recommend|insight|story)\b", t):
        return "long"
    n = len(t)
    if n <= 80:
        return "short"
    if n <= 220:
        return "medium"
    return "long"


def _budget(user_text: str, phase: str) -> int:
    """
    phase: "first" (tool planning) | "followup" (after tool outputs) | "force" (final answer)
    """
    kind = _classify_request(user_text)
    if phase == "first":
        return {"short": 180, "medium": 260, "long": 340}[kind]
    if phase == "followup":
        return {"short": 260, "medium": 420, "long": 700}[kind]
    # force
    return {"short": 260, "medium": 420, "long": 700}[kind]


def chat_turn(
    api_key: str,
    model: str,
    df: pd.DataFrame,
    session_state: Dict[str, Any],
    user_text: str,
    max_rounds: int = 3,
    history_limit: int = 6,
    max_output_tokens: Optional[int] = None,  # optional override
) -> str:
    client = OpenAI(api_key=api_key)
    ensure_dash_state(session_state)
    session_state.setdefault("copilot_history", [])

    # ---- Instructions (once)
    if "copilot_instructions" not in session_state:
        meta = tool_get_metadata(df)
        session_state["copilot_instructions"] = (
            "You are BudgetCopilot for a multi-page federal budget dashboard.\n"
            "- Use tools for dataset-backed numeric claims.\n"
            "- After tool results, answer the user directly (do not narrate tool use).\n"
            "- For changing dashboard view, call set_dashboard_state.\n"
            "- For opening a page, call navigate(page_key).\n"
            "- Keep answers concise; restate FY/range/bucket/item used.\n"
            "For questions like 'highest YoY change in FY ####', call get_top_yoy_change.\n"
            "For questions like 'biggest deficit increase', call get_deficit_series then compute YoY.\n"
            f"Dataset FY {meta['fy_min']}–{meta['fy_max']}; last actual FY {meta['fy_last_actual']}."
        )

    hist = session_state["copilot_history"][-history_limit:]
    base_messages = list(hist) + [{"role": "user", "content": user_text}]

    def _save_and_return(txt: str) -> str:
        txt = (txt or "").strip() or "(No text output.)"
        session_state["copilot_history"] = (
            hist + [{"role": "user", "content": user_text}, {"role": "assistant", "content": txt}]
        )[-history_limit:]
        return txt

    def _min_budget(phase: str) -> int:
        """
        IMPORTANT: Tool-calling needs headroom. If you set these too low, the model gets cut off
        before it emits the function_call object.
        """
        if max_output_tokens is not None:
            return int(max_output_tokens)

        # sensible defaults: fast but tool-safe
        if phase == "first":    # tool selection / tool call emission
            return max(_budget(user_text, "first"), 350)
        if phase == "followup": # after tool outputs, final answer usually
            return max(_budget(user_text, "followup"), 450)
        # force
        return max(_budget(user_text, "force"), 350)

    def _create(**kwargs):
        """
        Speed knobs that DON'T break tool calling.
        reasoning.effort='minimal' lowers latency; text.verbosity='low' reduces output. :contentReference[oaicite:1]{index=1}
        """
        return client.responses.create(
            **kwargs,
            reasoning={"effort": "minimal"},
            text={"verbosity": "low"},
        )

    # ---- 1) Initial call (tool planning / maybe immediate answer)
    cur = _create(
        model=model,
        instructions=session_state["copilot_instructions"],
        tools=TOOLS,
        input=base_messages,
        max_output_tokens=_min_budget("first"),
    )

    # ---- 2) Tool loop
    for round_i in range(max_rounds):
        out = getattr(cur, "output", None) or []
        tool_calls = [o for o in out if getattr(o, "type", None) == "function_call"]

        # If no tool calls, extract answer text
        if not tool_calls:
            text = _extract_response_text(cur)

            # Heuristic: if it looks like it started to call a tool but got cut off, retry once with bigger budget
            if (not text) or ("calling tool" in text.lower()) or text.strip().endswith(("FY 20", "FY 202", "FY 2025")):
                retry = _create(
                    model=model,
                    instructions=session_state["copilot_instructions"]
                    + "\nIf you need data, CALL THE TOOL directly. Do not narrate tool use.",
                    tools=TOOLS,
                    input=base_messages,
                    max_output_tokens=max(_min_budget("first"), 650),
                )
                # If retry produced tool calls, switch to it; else keep retry as current answer attempt
                retry_out = getattr(retry, "output", None) or []
                retry_tool_calls = [o for o in retry_out if getattr(o, "type", None) == "function_call"]
                cur = retry
                if retry_tool_calls:
                    tool_calls = retry_tool_calls
                else:
                    text = _extract_response_text(retry)
                    if text:
                        return _save_and_return(text)

            if text:
                return _save_and_return(text)

            # Still nothing: force a short answer (tools disabled)
            forced = _create(
                model=model,
                instructions=session_state["copilot_instructions"]
                + "\nAnswer succinctly in 1–2 sentences. Do not call tools.",
                tools=[],
                previous_response_id=cur.id,
                input=[{"role": "user", "content": "Answer now in 1–2 sentences."}],
                max_output_tokens=260,
            )
            return _save_and_return(_extract_response_text(forced))

        # Execute tools and send outputs back
        outputs: List[Dict[str, Any]] = []
        for call in tool_calls:
            args = call.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            result = run_tool(call.name, args or {}, df=df, session_state=session_state)
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        cur = _create(
            model=model,
            instructions=session_state["copilot_instructions"],
            tools=TOOLS,  # allow 1 more tool if needed
            previous_response_id=cur.id,
            input=outputs,
            max_output_tokens=_min_budget("followup"),
        )

    # ---- 3) If still tool-calling after limit, force final answer
    forced = _create(
        model=model,
        instructions=session_state["copilot_instructions"]
        + "\nNow provide the final answer using the tool results already provided. Do not call tools.",
        tools=[],
        previous_response_id=cur.id,
        input=[{"role": "user", "content": "Summarize and answer now."}],
        max_output_tokens=_min_budget("force"),
    )
    return _save_and_return(_extract_response_text(forced))




