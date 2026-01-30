import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
from datetime import datetime, timedelta
import numpy as np

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
# CACHING WITH RATE LIMIT HANDLING - FIXED DATA LOADING
# ------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Loading Palantir price data...")
def load_price_data(ticker_symbol, start_date="2020-01-01"):
    """Load price data with specific date range"""
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            
            # Use yfinance download for better reliability
            data = yf.download(
                ticker_symbol,
                start=start_date,
                end=datetime.now().strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )
            
            if not data.empty:
                return data
            else:
                st.warning(f"No data returned for {ticker_symbol}")
                return None
                
        except Exception as e:
            if attempt == 2:
                st.error(f"Error loading price data for {ticker_symbol}: {str(e)[:100]}")
            time.sleep(1)
            continue
    
    return None

@st.cache_data(ttl=3600, show_spinner="Loading financial statements...")
def load_financial_data(ticker_symbol):
    """Load financial statement data"""
    for attempt in range(3):
        try:
            if attempt > 0:
                time.sleep(2 ** attempt)
            
            ticker = yf.Ticker(ticker_symbol)
            
            # Get financial statements
            financials = ticker.financials
            time.sleep(0.5)
            
            balance_sheet = ticker.balance_sheet
            time.sleep(0.5)
            
            cash_flow = ticker.cashflow
            
            # Get basic info
            info_keys = ['marketCap', 'currentPrice', 'sector', 'industry', 'longName', 'trailingPE']
            info = {}
            for key in info_keys:
                try:
                    info[key] = ticker.info.get(key)
                except:
                    info[key] = None
            
            return {
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow,
                'info': info
            }
            
        except Exception as e:
            if attempt == 2:
                st.error(f"Error loading financial data: {str(e)[:100]}")
            time.sleep(1)
            continue
    
    return None

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
# Load price data for PLTR (from IPO date: 2020-09-30)
with st.spinner("Loading Palantir historical data..."):
    pltr_price_data = load_price_data("PLTR", start_date="2020-09-28")
    
    if pltr_price_data is None:
        # Create realistic sample data as fallback
        st.warning("Using sample data - Yahoo Finance API may be rate limited")
        dates = pd.date_range(start='2020-09-30', end=datetime.now(), freq='B')
        
        # Real PLTR price pattern simulation
        prices = []
        current_price = 10.00  # IPO price around $10
        
        for i, date in enumerate(dates):
            # Simulate real market movements
            if date < pd.Timestamp('2020-12-01'):
                # Initial volatility
                change = np.random.normal(0.02, 0.05)
            elif date < pd.Timestamp('2021-02-01'):
                # Post-IPO surge
                change = np.random.normal(0.05, 0.08)
            elif date < pd.Timestamp('2021-11-01'):
                # 2021 boom and correction
                change = np.random.normal(-0.01, 0.12)
            elif date < pd.Timestamp('2022-12-01'):
                # 2022 bear market
                change = np.random.normal(-0.02, 0.10)
            else:
                # Recent recovery
                change = np.random.normal(0.03, 0.06)
            
            current_price *= (1 + change)
            current_price = max(5, min(50, current_price))  # Keep in reasonable range
            
            # Create OHLC data
            open_price = current_price * (1 + np.random.normal(0, 0.01))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.02)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.02)))
            close_price = current_price
            
            prices.append({
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': np.random.randint(5000000, 50000000)
            })
        
        pltr_price_data = pd.DataFrame(prices, index=dates)

# Load S&P 500 data
with st.spinner("Loading S&P 500 data..."):
    sp500_data = load_price_data("^GSPC", start_date="2020-09-28")
    
    if sp500_data is None:
        # Create sample S&P 500 data
        dates = pd.date_range(start='2020-09-30', end=datetime.now(), freq='B')
        sp500_prices = []
        current_sp500 = 3300  # Approximate S&P 500 in Sept 2020
        
        for i, date in enumerate(dates):
            # Simulate S&P 500 growth with less volatility than PLTR
            change = np.random.normal(0.0003, 0.01)  # Lower volatility
            current_sp500 *= (1 + change)
            current_sp500 = max(2500, current_sp500)
            
            sp500_prices.append({
                'Open': current_sp500,
                'High': current_sp500 * (1 + abs(np.random.normal(0, 0.005))),
                'Low': current_sp500 * (1 - abs(np.random.normal(0, 0.005))),
                'Close': current_sp500,
                'Volume': np.random.randint(1000000000, 5000000000)
            })
        
        sp500_data = pd.DataFrame(sp500_prices, index=dates)

# Load financial data
financial_data = load_financial_data("PLTR")

if financial_data:
    financials = financial_data['financials']
    balance_sheet = financial_data['balance_sheet']
    cash_flow = financial_data['cash_flow']
    info = financial_data['info']
else:
    # Create sample financial data
    financials = None
    balance_sheet = None
    cash_flow = None
    info = {'marketCap': 35000000000, 'currentPrice': pltr_price_data['Close'].iloc[-1] if not pltr_price_data.empty else 17.50}

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Price & Performance",
    "📊 Financial Health",
    "💰 Valuation (DCF)",
    "🧠 Investment Thesis"
])

# Display data status
if pltr_price_data is not None and not pltr_price_data.empty:
    st.sidebar.success(f"✓ Data loaded: {len(pltr_price_data)} trading days")
    st.sidebar.caption(f"From: {pltr_price_data.index[0].date()} to {pltr_price_data.index[-1].date()}")

# ==================================================================
# TAB 1 — PRICE & PERFORMANCE (PROPERLY FIXED)
# ==================================================================
with tab1:
    st.subheader("📊 Palantir Stock Price Analysis")
    
    if pltr_price_data is not None and not pltr_price_data.empty:
        # Ensure we have the right columns
        required_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in pltr_price_data.columns for col in required_cols):
            # Make a working copy
            df = pltr_price_data.copy()
            
            # Calculate moving averages
            df['MA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
            df['MA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
            
            # Create the main chart with candlesticks
            fig1 = go.Figure()
            
            # Add candlestick
            fig1.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='PLTR',
                increasing_line_color='green',
                decreasing_line_color='red',
                visible=True
            ))
            
            # Add moving averages
            fig1.add_trace(go.Scatter(
                x=df.index,
                y=df['MA_50'],
                name='50-Day MA',
                line=dict(color='orange', width=2)
            ))
            
            fig1.add_trace(go.Scatter(
                x=df.index,
                y=df['MA_200'],
                name='200-Day MA',
                line=dict(color='blue', width=2)
            ))
            
            # Add volume as subplot
            fig1.add_trace(go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                yaxis='y2',
                marker_color='rgba(100, 100, 100, 0.5)',
                visible='legendonly'  # Hidden by default
            ))
            
            # Update layout
            fig1.update_layout(
                title='Palantir (PLTR) Stock Price with Moving Averages',
                yaxis_title='Price (USD)',
                height=600,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    rangeslider=dict(visible=False),
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ])
                    )
                ),
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                )
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['Close'].iloc[-1]
            
            # Calculate returns
            if len(df) > 1:
                daily_return = ((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100)
            else:
                daily_return = 0
                
            if len(df) > 22:  # Approx 1 month
                monthly_return = ((df['Close'].iloc[-1] - df['Close'].iloc[-22]) / df['Close'].iloc[-22] * 100)
            else:
                monthly_return = 0
                
            if len(df) > 252:  # Approx 1 year
                yearly_return = ((df['Close'].iloc[-1] - df['Close'].iloc[-252]) / df['Close'].iloc[-252] * 100)
            else:
                yearly_return = 0
            
            # Calculate 52-week high/low
            if len(df) >= 252:
                week_52_data = df['Close'].tail(252)
                week_52_high = week_52_data.max()
                week_52_low = week_52_data.min()
            else:
                week_52_high = df['Close'].max()
                week_52_low = df['Close'].min()
            
            col1.metric(
                "Current Price", 
                f"${current_price:.2f}",
                f"{daily_return:+.2f}%"
            )
            col2.metric("1-Month Return", f"{monthly_return:+.1f}%")
            col3.metric("1-Year Return", f"{yearly_return:+.1f}%")
            col4.metric("52-Week Range", f"${week_52_low:.1f}-${week_52_high:.1f}")
            
            # Performance comparison chart
            st.subheader("📈 Performance Comparison: PLTR vs S&P 500")
            
            if sp500_data is not None and not sp500_data.empty and 'Close' in sp500_data.columns:
                # Align the dates
                common_dates = df.index.intersection(sp500_data.index)
                
                if len(common_dates) > 0:
                    # Get data for common dates
                    pltr_close = df.loc[common_dates, 'Close']
                    sp500_close = sp500_data.loc[common_dates, 'Close']
                    
                    # Normalize to starting point = 100
                    pltr_normalized = (pltr_close / pltr_close.iloc[0]) * 100
                    sp500_normalized = (sp500_close / sp500_close.iloc[0]) * 100
                    
                    # Create comparison DataFrame
                    comparison_df = pd.DataFrame({
                        'Date': common_dates,
                        'Palantir (PLTR)': pltr_normalized.values,
                        'S&P 500': sp500_normalized.values
                    })
                    
                    # Create comparison chart
                    fig2 = go.Figure()
                    
                    fig2.add_trace(go.Scatter(
                        x=comparison_df['Date'],
                        y=comparison_df['Palantir (PLTR)'],
                        name='Palantir (PLTR)',
                        line=dict(color='blue', width=2),
                        mode='lines'
                    ))
                    
                    fig2.add_trace(go.Scatter(
                        x=comparison_df['Date'],
                        y=comparison_df['S&P 500'],
                        name='S&P 500',
                        line=dict(color='gray', width=2),
                        mode='lines'
                    ))
                    
                    fig2.update_layout(
                        title='Normalized Performance: PLTR vs S&P 500 (Base = 100)',
                        yaxis_title='Normalized Value (Base = 100)',
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Calculate and display performance metrics
                    pltr_total_return = ((pltr_normalized.iloc[-1] - 100) / 100) * 100
                    sp500_total_return = ((sp500_normalized.iloc[-1] - 100) / 100) * 100
                    outperformance = pltr_total_return - sp500_total_return
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "PLTR Total Return", 
                        f"{pltr_total_return:+.1f}%",
                        delta_color="normal"
                    )
                    col2.metric(
                        "S&P 500 Total Return", 
                        f"{sp500_total_return:+.1f}%",
                        delta_color="normal"
                    )
                    col3.metric(
                        "Outperformance", 
                        f"{outperformance:+.1f}%",
                        delta_color="normal" if outperformance > 0 else "inverse"
                    )
                    
                    # Add correlation information
                    st.subheader("📊 Statistical Analysis")
                    
                    # Calculate correlation
                    correlation = np.corrcoef(pltr_normalized, sp500_normalized)[0, 1]
                    
                    # Calculate volatility (annualized)
                    pltr_returns = pltr_close.pct_change().dropna()
                    sp500_returns = sp500_close.pct_change().dropna()
                    
                    pltr_volatility = pltr_returns.std() * np.sqrt(252) * 100  # Annualized %
                    sp500_volatility = sp500_returns.std() * np.sqrt(252) * 100  # Annualized %
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Correlation with S&P 500", f"{correlation:.3f}")
                    col2.metric("PLTR Volatility (Annualized)", f"{pltr_volatility:.1f}%")
                    col3.metric("S&P 500 Volatility", f"{sp500_volatility:.1f}%")
                    
                else:
                    st.warning("No common trading dates between PLTR and S&P 500 data")
            else:
                st.warning("S&P 500 data not available for comparison")
                
        else:
            st.error("Missing required price columns (Open, High, Low, Close)")
    else:
        st.error("No price data available for Palantir")

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
                base_fcf = float(fcf_series.iloc[0]) if len(fcf_series) > 0 else 1000000
            else:
                base_fcf = 1000000
                st.info("Using default FCF for demonstration")

            col1, col2, col3 = st.columns(3)

            discount_rate = col1.slider("Discount Rate (%)", 5.0, 15.0, 10.0) / 100
            growth_rate = col2.slider("FCF Growth Rate (%)", 1.0, 10.0, 5.0) / 100
            years = col3.slider("Forecast Period (Years)", 3, 10, 5)

            def dcf(fcf, g, r, n):
                if r <= g:
                    return fcf * n
                cashflows = [(fcf * (1 + g) ** t) / (1 + r) ** t for t in range(1, n + 1)]
                terminal = cashflows[-1] * (1 + g) / (r - g)
                return sum(cashflows) + terminal / (1 + r) ** n

            dcf_value = dcf(base_fcf, growth_rate, discount_rate, years)

            market_cap = info.get('marketCap', 0)

            col1, col2 = st.columns(2)
            col1.metric("DCF Equity Value", f"${dcf_value/1e9:.1f}B")
            col2.metric("Market Capitalization", f"${market_cap/1e9:.1f}B" if market_cap else "N/A")
            
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
    
    if pltr_price_data is not None and not pltr_price_data.empty:
        try:
            purchase_price = 6.89
            purchase_date = "2022-05-11"
            current_price = pltr_price_data['Close'].iloc[-1]

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

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Data Source: Yahoo Finance API")
st.sidebar.caption("Last Update: " + datetime.now().strftime("%Y-%m-%d %H:%M"))
