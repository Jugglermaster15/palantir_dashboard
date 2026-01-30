import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from datetime import datetime, timedelta

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
# CACHING WITH RATE LIMIT HANDLING
# ------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading financial data...")
def load_financial_data(ticker_symbol, max_retries=3):
    """Load financial data with retry logic for rate limiting"""
    for attempt in range(max_retries):
        try:
            # Add delay between retries with exponential backoff
            if attempt > 0:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                time.sleep(wait_time)
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Use a more conservative approach with smaller data requests
            price_data = ticker.history(period="1y")  # Reduced from "max" to "1y"
            
            # Add delay between requests
            time.sleep(0.5)
            
            # Get financial statements
            financials = ticker.financials
            time.sleep(0.5)
            
            balance_sheet = ticker.balance_sheet
            time.sleep(0.5)
            
            cash_flow = ticker.cashflow
            
            # Get basic info with minimal data
            info_keys = ['marketCap', 'currentPrice', 'sector', 'industry']
            info = {}
            for key in info_keys:
                try:
                    info[key] = ticker.info.get(key)
                except:
                    info[key] = None
            
            return {
                'price_data': price_data,
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'info': info,
                'ticker_symbol': ticker_symbol,
                'last_updated': datetime.now()
            }
            
        except yf.exceptions.YFRateLimitError:
            if attempt == max_retries - 1:
                st.error("⚠️ Yahoo Finance rate limit reached. Please try again in a few minutes.")
                # Return cached sample data or None
                return None
            continue
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error loading data: {e}")
                return None
            time.sleep(1)  # Wait before retry
            continue
    
    return None

@st.cache_data(ttl=3600)
def load_sp500_data():
    """Load S&P 500 data with retry logic"""
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            sp500 = yf.Ticker("^GSPC")
            return sp500.history(period="1y")  # Reduced period
        except yf.exceptions.YFRateLimitError:
            if attempt == 2:
                st.warning("S&P 500 data temporarily unavailable")
                return None
            time.sleep(1)
            continue
        except Exception:
            return None

# ------------------------------------------------------------------
# LOAD DATA WITH FALLBACK
# ------------------------------------------------------------------
# Display loading message
with st.spinner("Loading Palantir financial data..."):
    data = load_financial_data("PLTR")

# Check if data was loaded successfully
if data is None:
    st.error("""
    **Unable to load financial data from Yahoo Finance.**
    
    Possible reasons:
    1. Rate limiting - Too many requests to Yahoo Finance
    2. Network issues
    3. Yahoo Finance API changes
    
    Please try refreshing the page in a few minutes.
    """)
    st.stop()

# Unpack the data
price_data = data['price_data']
financials = data['financials']
balance_sheet = data['balance_sheet']
cash_flow = data['cash_flow']
info = data['info']

# Load S&P 500 data
sp500_data = load_sp500_data()

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Performance",
    "📊 Financial Health",
    "💰 Valuation (DCF)",
    "🧠 Investment Thesis"
])

# Display last update time
st.caption(f"Data last updated: {data.get('last_updated', 'N/A')}")

# ==================================================================
# TAB 1 — PRICE & PERFORMANCE
# ==================================================================
with tab1:
    st.subheader("Stock Price Evolution")
    
    if price_data is not None and len(price_data) > 0:
        # Calculate moving averages
        price_data_copy = price_data.copy()
        price_data_copy["MA_50"] = price_data_copy["Close"].rolling(50, min_periods=1).mean()
        price_data_copy["MA_200"] = price_data_copy["Close"].rolling(200, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_data_copy.index,
            y=price_data_copy["Close"],
            name="Close Price",
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=price_data_copy.index,
            y=price_data_copy["MA_50"],
            name="50D MA",
            line=dict(color='orange', width=1)
        ))
        
        # Only add 200D MA if we have enough data
        if len(price_data_copy) >= 200:
            fig.add_trace(go.Scatter(
                x=price_data_copy.index,
                y=price_data_copy["MA_200"],
                name="200D MA",
                line=dict(color='red', width=1)
            ))

        fig.update_layout(
            height=450,
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)
        
        # Performance comparison
        if sp500_data is not None and len(sp500_data) > 0:
            # Use common date range
            common_start = max(price_data_copy.index.min(), sp500_data.index.min())
            
            pltr_norm = price_data_copy.loc[common_start:, "Close"]
            pltr_norm = pltr_norm / pltr_norm.iloc[0]
            
            sp500_norm = sp500_data.loc[common_start:, "Close"]
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
                    labels={"value": "Normalized Performance"},
                    title=f"Performance since {common_start.date()}"
                ),
                use_container_width=True
            )
        else:
            st.warning("S&P 500 comparison data not available")
    else:
        st.warning("Price data not available")

# ==================================================================
# TAB 2 — FINANCIAL HEALTH
# ==================================================================
with tab2:
    st.subheader("Key Financial Ratios")
    
    if financials is not None and balance_sheet is not None:
        try:
            # Safely get data with defaults
            revenue = financials.loc["Total Revenue"] if "Total Revenue" in financials.index else pd.Series([0])
            net_income = financials.loc["Net Income"] if "Net Income" in financials.index else pd.Series([0])
            gross_profit = financials.loc["Gross Profit"] if "Gross Profit" in financials.index else revenue * 0.7
            equity = balance_sheet.loc["Stockholders Equity"] if "Stockholders Equity" in balance_sheet.index else pd.Series([0])

            col1, col2, col3 = st.columns(3)

            gross_margin = (gross_profit.iloc[0] / revenue.iloc[0] * 100) if revenue.iloc[0] != 0 else 0
            net_margin = (net_income.iloc[0] / revenue.iloc[0] * 100) if revenue.iloc[0] != 0 else 0
            roe = (net_income.iloc[0] / equity.iloc[0] * 100) if equity.iloc[0] != 0 else 0

            col1.metric("Gross Margin", f"{gross_margin:.1f}%")
            col2.metric("Net Margin", f"{net_margin:.1f}%")
            col3.metric("ROE", f"{roe:.1f}%")

            # Liquidity metrics
            st.subheader("Liquidity & Leverage")

            current_assets = balance_sheet.loc["Current Assets"] if "Current Assets" in balance_sheet.index else pd.Series([0])
            current_liabilities = balance_sheet.loc["Current Liabilities"] if "Current Liabilities" in balance_sheet.index else pd.Series([1])
            total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest"] if "Total Liabilities Net Minority Interest" in balance_sheet.index else pd.Series([0])

            col4, col5 = st.columns(2)

            current_ratio = current_assets.iloc[0] / current_liabilities.iloc[0] if current_liabilities.iloc[0] != 0 else 0
            debt_to_equity = total_liabilities.iloc[0] / equity.iloc[0] if equity.iloc[0] != 0 else 0

            col4.metric("Current Ratio", f"{current_ratio:.2f}")
            col5.metric("Debt to Equity", f"{debt_to_equity:.2f}")
            
        except Exception as e:
            st.error(f"Error calculating financial ratios: {str(e)[:100]}")
    else:
        st.warning("Financial statement data not available")

# ==================================================================
# TAB 3 — VALUATION (DCF)
# ==================================================================
with tab3:
    st.subheader("Discounted Cash Flow (DCF)")
    
    if cash_flow is not None:
        try:
            if "Free Cash Flow" in cash_flow.index:
                fcf_series = cash_flow.loc["Free Cash Flow"]
                base_fcf = float(fcf_series.iloc[0]) if len(fcf_series) > 0 else 1000000  # Default if no data
            else:
                base_fcf = 1000000  # Default value
                st.info("Using default FCF for demonstration")

            col1, col2, col3 = st.columns(3)

            discount_rate = col1.slider("Discount Rate (%)", 5.0, 15.0, 10.0) / 100
            growth_rate = col2.slider("FCF Growth Rate (%)", 1.0, 10.0, 5.0) / 100
            years = col3.slider("Forecast Period (Years)", 3, 10, 5)

            def dcf(fcf, g, r, n):
                if r <= g:
                    return fcf * n  # Simple fallback if r <= g
                cashflows = [(fcf * (1 + g) ** t) / (1 + r) ** t for t in range(1, n + 1)]
                terminal = cashflows[-1] * (1 + g) / (r - g)
                return sum(cashflows) + terminal / (1 + r) ** n

            dcf_value = dcf(base_fcf, growth_rate, discount_rate, years)

            market_cap = info.get('marketCap', 0)

            col1, col2 = st.columns(2)
            col1.metric("DCF Equity Value", f"${dcf_value/1e9:.1f}B")
            col2.metric("Market Capitalization", f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
            
            # Show implied upside/downside
            if market_cap and market_cap > 0:
                implied_change = (dcf_value - market_cap) / market_cap * 100
                st.metric("Implied Upside/Downside", f"{implied_change:.1f}%", 
                         delta_color="normal" if implied_change > 0 else "inverse")
                
        except Exception as e:
            st.error(f"Error in DCF calculation: {str(e)[:100]}")
    else:
        st.warning("Cash flow data not available for DCF calculation")

# ==================================================================
# TAB 4 — INVESTMENT THESIS
# ==================================================================
with tab4:
    st.subheader("Personal Investment Perspective")
    
    if price_data is not None and len(price_data) > 0:
        try:
            purchase_price = 6.89
            purchase_date = "2022-05-11"
            current_price = price_data["Close"].iloc[-1]

            roi = (current_price / purchase_price - 1) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Entry Price", f"${purchase_price}")
            col2.metric("Current Price", f"${current_price:.2f}")
            col3.metric("ROI", f"{roi:.1f}%", delta_color="normal")

            st.markdown("""
### 🎯 Investment Thesis

**Why Palantir?**
- **Mission-critical software**: Foundry and Gotham are essential for government and enterprise clients
- **High switching costs**: Deep integration creates customer lock-in
- **AI/ML leadership**: Strong positioning in artificial intelligence and machine learning
- **Government contracts**: Stable, long-term revenue from defense and intelligence agencies
- **Commercial expansion**: Growing footprint in private sector across multiple industries

**📊 Financial Strengths**
- Improving profitability margins
- Strong balance sheet with minimal debt
- Consistent revenue growth
- High gross margins typical of software companies

**⚠️ Key Risks**
- **Valuation concerns**: Historically traded at premium multiples
- **Customer concentration**: Significant revenue from limited number of contracts
- **Competition**: Facing pressure from cloud providers (AWS, Azure, GCP)
- **Stock-based compensation**: High dilution from employee compensation
- **Macro sensitivity**: Government spending can be cyclical

**🎯 Price Targets & Strategy**
- *Conservative target*: $25-30 based on 10-15x forward sales
- *Bull case*: $40+ with successful AI platform adoption
- *Entry points*: Accumulate below $20 for long-term holding
- *Exit strategy*: Consider trimming above $35 if valuation becomes excessive
            """)
        except Exception as e:
            st.error(f"Error displaying investment thesis: {str(e)[:100]}")
    else:
        st.warning("Current price data not available")