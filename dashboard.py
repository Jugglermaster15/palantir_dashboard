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
# CACHING - FIXED VERSION
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_financial_data(ticker_symbol):
    """Load and cache all financial data as serializable objects"""
    ticker = yf.Ticker(ticker_symbol)
    
    # Get price data
    price_data = ticker.history(period="max")
    
    # Get financial statements
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet
    cash_flow = ticker.cashflow
    info = ticker.info
    
    return {
        'price_data': price_data,
        'financials': financials,
        'balance_sheet': balance_sheet,
        'cash_flow': cash_flow,
        'info': info,
        'ticker_symbol': ticker_symbol
    }

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
data = load_financial_data("PLTR")

# Unpack the data
price_data = data['price_data']
financials = data['financials']
balance_sheet = data['balance_sheet']
cash_flow = data['cash_flow']
info = data['info']

# Load S&P 500 data separately
@st.cache_data(ttl=3600)
def load_sp500_data():
    sp500 = yf.Ticker("^GSPC")
    return sp500.history(period="max")

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

    # Calculate moving averages
    price_data_copy = price_data.copy()
    price_data_copy["MA_50"] = price_data_copy["Close"].rolling(50).mean()
    price_data_copy["MA_200"] = price_data_copy["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_data_copy.index,
        y=price_data_copy["Close"],
        name="Close Price"
    ))
    fig.add_trace(go.Scatter(
        x=price_data_copy.index,
        y=price_data_copy["MA_50"],
        name="50D MA"
    ))
    fig.add_trace(go.Scatter(
        x=price_data_copy.index,
        y=price_data_copy["MA_200"],
        name="200D MA"
    ))

    fig.update_layout(
        height=450,
        yaxis_title="Price (USD)",
        xaxis_title="Date"
    )

    st.plotly_chart(fig, use_container_width=True)

    start_date = "2020-09-30"
    pltr_norm = price_data_copy.loc[start_date:, "Close"]
    pltr_norm = pltr_norm / pltr_norm.iloc[0]

    sp500_data = load_sp500_data()
    sp500_norm = sp500_data.loc[start_date:, "Close"]
    sp500_norm = sp500_norm / sp500_norm.iloc[0]

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

    # Add error handling for missing data
    try:
        revenue = financials.loc["Total Revenue"]
        net_income = financials.loc["Net Income"]
        gross_profit = financials.loc["Gross Profit"]
        equity = balance_sheet.loc["Stockholders Equity"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Gross Margin",
            f"{(gross_profit.iloc[0] / revenue.iloc[0] * 100) if revenue.iloc[0] != 0 else 0:.1f}%"
        )
        col2.metric(
            "Net Margin",
            f"{(net_income.iloc[0] / revenue.iloc[0] * 100) if revenue.iloc[0] != 0 else 0:.1f}%"
        )
        col3.metric(
            "ROE",
            f"{(net_income.iloc[0] / equity.iloc[0] * 100) if equity.iloc[0] != 0 else 0:.1f}%"
        )

        current_assets = balance_sheet.loc["Current Assets"]
        current_liabilities = balance_sheet.loc["Current Liabilities"]

        st.subheader("Liquidity & Leverage")

        col4, col5 = st.columns(2)

        col4.metric(
            "Current Ratio",
            f"{(current_assets.iloc[0] / current_liabilities.iloc[0]) if current_liabilities.iloc[0] != 0 else 0:.2f}"
        )

        total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"]
        col5.metric(
            "Debt to Equity",
            f"{(total_liabilities.iloc[0] / equity.iloc[0]) if equity.iloc[0] != 0 else 0:.2f}"
        )
    except KeyError as e:
        st.error(f"Missing financial data: {e}. Some metrics may not be available.")
    except Exception as e:
        st.error(f"Error calculating financial ratios: {e}")

# ==================================================================
# TAB 3 — VALUATION (DCF)
# ==================================================================
with tab3:
    st.subheader("Discounted Cash Flow (DCF)")

    try:
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

        market_cap = info.get("marketCap")

        st.metric("DCF Equity Value", f"${dcf_value/1e9:.1f}B")
        if market_cap:
            st.metric("Market Capitalization", f"${market_cap/1e9:.1f}B")
    except KeyError:
        st.warning("Free Cash Flow data not available for DCF calculation.")
    except ZeroDivisionError:
        st.warning("Cannot calculate DCF with current parameters. Please adjust growth/discount rates.")
    except Exception as e:
        st.error(f"Error in DCF calculation: {e}")

# ==================================================================
# TAB 4 — INVESTMENT THESIS
# ==================================================================
with tab4:
    st.subheader("Personal Investment Perspective")

    try:
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
    except Exception as e:
        st.error(f"Error displaying investment thesis: {e}")