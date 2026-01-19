# BudgetSight
Federal Financial Budget Dashboard for Presidential AI Challenge

**Live demo:** budgetsight.streamlit.app  
**Repo:** https://github.com/adityapaul1219/BudgetSight/

## What this project is
**BudgetSight** is an interactive Federal Budget dashboard that combines:
- **Historical actuals** and **Prophet-based forecasts**
- A unified `display_value` series (actuals where available, forecast afterward)
- A multi-page Streamlit UI for exploring:
  - Overview totals
  - Deficit/outlays/receipts
  - Line-item trends (with YoY change and forecast intervals)
  - AI Copilot (natural language) experience

## How to Run the Federal Budget Dashboard
Open the app link (provided by the team). **Preloaded with Data (No Upload Needed)** The dashboard loads a prepackaged dataset automatically.
Navigate using the sidebar: **Overview → Deficit Explorer → Line Item Explorer → AI Copilot**.
1. **Overview**: confirm totals and the “last actual FY” boundary (actual vs forecast).
2. **Deficit Explorer**: inspect deficit trend over time and deficit/GDP.
3. **Line Item Explorer**: pick an outlay bucket or receipts item and check YoY changes + forecast bands.
4. **AI Copilot (GenAI)**: ask questions or click **Generate insights** → **Apply view** to jump directly to the most interesting stories.  
   *These insights are computed from the dataset (no LLM required).*

## What to Try First (Quick Demo Flow)
1. Go to **Copilot** and click **Generate insights** → then click **Apply view** on any insight card.
2. Explore the selected view on the linked dashboard page.
3. Use **Line Item Explorer** to drill down with pies and view YoY bars.

---

## Using AI Copilot (GenAI Chat)
BudgetCopilot can answer questions like:
- “What year experienced the biggest deficit increase?”
- “What item had the highest YoY change in 2021?”
- “Set FY to 2025 and open the deficit explorer”
- “Show the time series for receipts: Individual Income Taxes”
- “What item had the highest year-over-year change in 2021?”
- “Show the top outlay items in FY 2023.”

---

## Methodology disclaimer
Forecast values are **model-based projections** (Prophet). They are not causal claims—use them as scenario guidance.
Data source: Office of Management and Budget / govinfo budget tables

---

### Notes / Assumptions
- Fiscal years are labeled as `FY` (integer).
- Values displayed are the dashboard’s **unified series**:
  - uses actual `value` when available
  - otherwise uses forecast `yhat`
- Forecasts were generated with Prophet per line item.

