import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Palantir (PLTR) – Financial Analysis",
    layout="wide"
)

st.title("📊 Palantir Technologies (PLTR) – Financial Analysis")
st.info(
    "This dashboard presents a personal financial analysis for educational purposes only. "
    "It does not constitute financial advice."
)

# ------------------------------------------------------------------
# CACHING
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_ticker(ticker):
    return yf.Ticker(ticker)

@st.cache_data(ttl=3600)
def load_price_data(ticker):
    return ticker.history(period="max")

# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
pltr = load_ticker("PLTR")
price_data = load_price_data(pltr)

financials = pltr.financials
balance_sheet = pltr.balance_sheet
cash_flow = pltr.cashflow

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Performance",
    "📊 Financial Health",
    "💰 Valuation (DCF)",
    "🧠 Investment Thesis"
])

# ==================================================================
# TAB 1 — PRICE & PERFORMANCE
# ==================================================================
with tab1:
    st.subheader("Stock Price Evolution")

    price_data["MA_50"] = price_data["Close"].rolling(50).mean()
    price_data["MA_200"] = price_data["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data["Close"],
        name="Close Price"
    ))
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data["MA_50"],
        name="50D MA"
    ))
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data["MA_200"],
        name="200D MA"
    ))

    fig.update_layout(
        height=450,
        yaxis_title="Price (USD)",
        xaxis_title="Date"
    )

    st.plotly_chart(fig, use_container_width=True)

    start_date = "2020-09-30"
    pltr_norm = price_data.loc[start_date:, "Close"]
    pltr_norm = pltr_norm / pltr_norm.iloc[0]

    sp500 = load_ticker("^GSPC")
    sp500_data = load_price_data(sp500).loc[start_date:, "Close"]
    sp500_norm = sp500_data / sp500_data.iloc[0]

    comparison = pd.DataFrame({
        "Date": pltr_norm.index,
        "Palantir": pltr_norm.values,
        "S&P 500": sp500_norm.values
    })

    st.subheader("Palantir vs S&P 500 (Normalized)")
    st.plotly_chart(
        px.line(
            comparison,
            x="Date",
            y=["Palantir", "S&P 500"],
            labels={"value": "Normalized Performance"}
        ),
        use_container_width=True
    )

# ==================================================================
# TAB 2 — FINANCIAL HEALTH
# ==================================================================
with tab2:
    st.subheader("Key Financial Ratios")

    revenue = financials.loc["Total Revenue"]
    net_income = financials.loc["Net Income"]
    gross_profit = financials.loc["Gross Profit"]
    equity = balance_sheet.loc["Stockholders Equity"]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Gross Margin",
        f"{(gross_profit / revenue * 100).iloc[0]:.1f}%"
    )
    col2.metric(
        "Net Margin",
        f"{(net_income / revenue * 100).iloc[0]:.1f}%"
    )
    col3.metric(
        "ROE",
        f"{(net_income / equity * 100).iloc[0]:.1f}%"
    )

    current_assets = balance_sheet.loc["Current Assets"]
    current_liabilities = balance_sheet.loc["Current Liabilities"]

    st.subheader("Liquidity & Leverage")

    col4, col5 = st.columns(2)

    col4.metric(
        "Current Ratio",
        f"{(current_assets / current_liabilities).iloc[0]:.2f}"
    )

    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"]
    col5.metric(
        "Debt to Equity",
        f"{(total_liabilities / equity).iloc[0]:.2f}"
    )

# ==================================================================
# TAB 3 — VALUATION (DCF)
# ==================================================================
with tab3:
    st.subheader("Discounted Cash Flow (DCF)")

    fcf_series = cash_flow.loc["Free Cash Flow"]
    base_fcf = float(fcf_series.iloc[0])

    col1, col2, col3 = st.columns(3)

    discount_rate = col1.slider("Discount Rate (%)", 5.0, 15.0, 10.0) / 100
    growth_rate = col2.slider("FCF Growth Rate (%)", 1.0, 10.0, 5.0) / 100
    years = col3.slider("Forecast Period (Years)", 3, 10, 5)

    def dcf(fcf, g, r, n):
        cashflows = [(fcf * (1 + g) ** t) / (1 + r) ** t for t in range(1, n + 1)]
        terminal = cashflows[-1] * (1 + g) / (r - g)
        return sum(cashflows) + terminal / (1 + r) ** n

    dcf_value = dcf(base_fcf, growth_rate, discount_rate, years)

    market_cap = pltr.info.get("marketCap")

    st.metric("DCF Equity Value", f"${dcf_value/1e9:.1f}B")
    if market_cap:
        st.metric("Market Capitalization", f"${market_cap/1e9:.1f}B")

# ==================================================================
# TAB 4 — INVESTMENT THESIS
# ==================================================================
with tab4:
    st.subheader("Personal Investment Perspective")

    purchase_price = 6.89
    purchase_date = "2022-05-11"
    current_price = price_data["Close"].iloc[-1]

    roi = (current_price / purchase_price - 1) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Entry Price", f"${purchase_price}")
    col2.metric("Current Price", f"${current_price:.2f}")
    col3.metric("ROI", f"{roi:.1f}%")

    st.markdown("""
**Why Palantir?**
- Strong positioning in mission-critical data analytics  
- High switching costs and long-term contracts  
- Increasing focus on commercial expansion  

**Key Risks**
- Valuation sensitivity  
- Customer concentration  
- Competitive pressure from large cloud providers
""")
