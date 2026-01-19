# BudgetSight
Federal Financial Budget Dashboard for Presidential AI Challenge

## 👩‍⚖️ For Judges: How to Run the Federal Budget Dashboard

This project is a **multi-page Streamlit dashboard** that explores U.S. federal budget **outlays vs receipts** (historical + forecast), with:
- **Overview**: outlay buckets + receipts trends
- **Deficit Explorer**: outlays vs receipts and deficit over time (optionally vs GDP)
- **Line Item Explorer**: drill-down pies + time series + YoY changes
- **AI Copilot (GenAI)**: ask questions and apply “one-click” views to the dashboard

### Preloaded with Data (No Upload Needed)
1. Open the app link (provided by the team).
2. The dashboard loads a prepackaged dataset automatically.
3. Navigate using the sidebar: **Overview → Deficit Explorer → Line Item Explorer → AI Copilot**.

---

## 🤖 Using BudgetCopilot (GenAI Chat)
BudgetCopilot can answer questions like:
- “What year experienced the biggest deficit increase?”
- “What item had the highest YoY change in 2021?”
- “Set FY to 2025 and open the deficit explorer”
- “Show the time series for receipts: Individual Income Taxes”

### API Key
Depending on deployment settings:
- **If the app asks for an OpenAI API key:** paste your key in the sidebar.
- **If not:** the app is using a hosted key and is ready to use.

---

## ⭐ What to Try First (Quick Demo Flow)
1. Go to **Copilot** and click **Generate insights** → then click **Apply view** on any insight card.
2. Explore the selected view on the linked dashboard page.
3. Use **Line Item Explorer** to drill down with pies and view YoY bars.

---

## Notes / Assumptions
- Fiscal years are labeled as `FY` (integer).
- Values displayed are the dashboard’s **unified series**:
  - uses actual `value` when available
  - otherwise uses forecast `yhat`
- Forecasts were generated with Prophet per line item.

