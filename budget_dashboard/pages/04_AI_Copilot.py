import streamlit as st
import pandas as pd

from copilot_agent import chat_turn, ensure_dash_state, PAGE_MAP, auto_insights


st.title("🤖 BudgetCopilot (Dashboard-wide)")

if "df" not in st.session_state:
    st.warning("Load data on the main page first.")
    st.stop()

df = st.session_state["df"]
dash_state = ensure_dash_state(st.session_state)

@st.cache_data(show_spinner=False)
def cached_auto_insights(df: pd.DataFrame, fy0: int, fy1: int, k: int):
    return auto_insights(df, fy_start=fy0, fy_end=fy1, top_k=k)

with st.sidebar:
    st.subheader("LLM settings")

    # Prefer secrets on Streamlit Cloud
    default_key = st.secrets.get("OPENAI_API_KEY", "") or st.secrets.get("openai_api_key", "")
    api_key = default_key
    if not default_key:
        api_key = st.text_input("OpenAI API key", type="password", value="")
    else:
        st.caption("API key loaded from secrets ✅")

    model = st.text_input("Model", value="gpt-5-mini")
    debug = st.toggle("Show debug", value=False)
    st.session_state["copilot_debug"] = debug

    if st.button("🧹 Clear chat"):
        st.session_state["copilot_messages"] = []
        st.session_state["copilot_history"] = []
        st.session_state["copilot_nav_to"] = None
        st.rerun()

    if debug:
        st.subheader("dash_state (live)")
        st.json(st.session_state.get("dash_state", {}))

st.markdown("## ✨ Auto-Insights")
st.caption("Generated from your dataset (no LLM). Click **Apply view** to update filters and jump to the right page.")

info_min = int(df["fy"].min())
info_max = int(df["fy"].max())

dash = st.session_state.get("dash_state", {})
default_range = dash.get("fy_range")
if isinstance(default_range, (list, tuple)) and len(default_range) == 2:
    r0 = (max(info_min, int(default_range[0])), min(info_max, int(default_range[1])))
else:
    r0 = (info_min, info_max)

c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    rng = st.slider("Insight FY range", min_value=info_min, max_value=info_max, value=r0, key="ins_rng")
with c2:
    top_k = st.selectbox("How many cards", [3, 6, 8, 10, 12], index=0, key="ins_k")
with c3:
    auto_nav = st.toggle("Auto-navigate on apply", value=True, key="ins_nav")

gen = st.button("🚀 Generate insights", key="btn_gen_insights")

if gen:
    res = cached_auto_insights(df, int(rng[0]), int(rng[1]), int(top_k))
    st.session_state["auto_insights_res"] = res

res = st.session_state.get("auto_insights_res")
if res and isinstance(res, dict) and res.get("cards"):
    for card in res["cards"]:
        with st.container(border=True):
            st.markdown(f"### {card['title']}")
            st.markdown(card.get("subtitle", ""))

            why = card.get("why_it_matters")
            if why:
                st.markdown(f"**Why it matters:** {why}")

            conf = card.get("confidence_note")
            if conf:
                st.caption(f"Confidence note: {conf}")

            ev = card.get("evidence", [])
            if ev:
                st.dataframe(pd.DataFrame(ev), width="stretch", hide_index=True)

            colA, colB = st.columns([1, 1])
            with colA:
                if st.button("✅ Apply view", key=f"apply_{card['id']}"):
                    updates = card.get("apply_updates", {}) or {}
                    dash = ensure_dash_state(st.session_state)
                    for k, v in updates.items():
                        if k in dash:
                            dash[k] = v
                    st.session_state["dash_state"] = dash

                    # mirror to Line Item Explorer keys
                    if dash.get("outlay_bucket") is not None:
                        st.session_state["sel_outlay_bucket"] = dash["outlay_bucket"]
                    if dash.get("outlay_item") is not None:
                        st.session_state["sel_outlay_item"] = dash["outlay_item"]
                    if dash.get("receipt_item") is not None:
                        st.session_state["sel_receipt_item"] = dash["receipt_item"]

                    page_key = card.get("page_key")
                    if page_key in PAGE_MAP and auto_nav:
                        st.switch_page(PAGE_MAP[page_key])
                    elif page_key in PAGE_MAP:
                        st.session_state["copilot_nav_to"] = PAGE_MAP[page_key]
                        st.success("Applied. Use the prompt below to jump pages.")
                    else:
                        st.success("Applied.")
                    st.rerun()

            with colB:
                page_key = card.get("page_key")
                if page_key in PAGE_MAP:
                    st.page_link(PAGE_MAP[page_key], label="🔎 Open related page")

else:
    st.info("Click **Generate insights** to get story cards you can apply to the dashboard.")

st.markdown("---")
st.subheader("💬 Ask Copilot")

if "copilot_messages" not in st.session_state:
    st.session_state["copilot_messages"] = []

for m in st.session_state["copilot_messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about overview/deficit/line items…")
if prompt:
    if not api_key:
        st.error("Add your OpenAI API key in secrets or the sidebar.")
        st.stop()

    st.session_state["copilot_messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chat_turn(
                api_key=api_key,
                model=model,
                df=df,
                session_state=st.session_state,
                user_text=prompt,
            )
        st.markdown(answer)

        if st.session_state.get("copilot_debug") and answer.strip() == "(No text output.)":
            st.write("Agent history (compact):", st.session_state.get("copilot_history", []))
            st.write("Dash state:", st.session_state.get("dash_state", {}))
            st.warning("Assistant returned no text. This usually indicates a tool loop or response parsing issue.")

    st.session_state["copilot_messages"].append({"role": "assistant", "content": answer})

nav_to = st.session_state.get("copilot_nav_to")
if nav_to:
    st.info(f"Copilot suggests opening: `{nav_to}`")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("Go now"):
            st.session_state["copilot_nav_to"] = None
            st.switch_page(nav_to)
    with c2:
        if st.button("Dismiss"):
            st.session_state["copilot_nav_to"] = None
