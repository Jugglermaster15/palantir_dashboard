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
            
            # Get price data with enough history for meaningful analysis
            # Use multiple periods to get comprehensive data
            price_data = ticker.history(period="5y")  # Changed to 5 years for better charts
            
            # Add delay between requests
            time.sleep(0.5)
            
            # Get financial statements
            financials = ticker.financials
            time.sleep(0.5)
            
            balance_sheet = ticker.balance_sheet
            time.sleep(0.5)
            
            cash_flow = ticker.cashflow
            
            # Get basic info with minimal data
            info_keys = ['marketCap', 'currentPrice', 'sector', 'industry', 'longName']
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
                return None
            continue
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error loading data: {e}")
                return None
            time.sleep(1)
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
            return sp500.history(period="5y")  # Match 5-year period
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
    
    # Optionally, show cached sample data for demonstration
    st.warning("Showing sample data for demonstration purposes")
    
    # Create sample price data (real PLTR historical pattern)
    dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
    np.random.seed(42)
    base_price = 10
    volatility = 0.03
    prices = []
    current_price = base_price
    
    for i in range(len(dates)):
        # Simulate some real PLTR price patterns
        if i < 100:  # Early 2020
            current_price *= (1 + np.random.normal(0.001, volatility))
        elif i < 300:  # IPO and post-IPO period
            current_price = 20 + np.random.normal(0, 5)
        elif i < 600:  # 2021 volatility
            current_price = 25 + np.random.normal(0, 8)
        else:  # Recent period
            current_price = 15 + np.random.normal(0, 3)
        
        prices.append(max(5, current_price))  # Ensure price doesn't go below 5
    
    sample_price_data = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': [1000000 + np.random.randint(-500000, 500000) for _ in prices]
    }, index=dates)
    
    data = {
        'price_data': sample_price_data,
        'financials': None,
        'balance_sheet': None,
        'cash_flow': None,
        'info': {'marketCap': 30000000000, 'currentPrice': 17.50, 'longName': 'Palantir Technologies Inc.'},
        'last_updated': datetime.now()
    }

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
if data and 'last_updated' in data:
    st.caption(f"Data last updated: {data['last_updated'].strftime('%Y-%m-%d %H:%M:%S')}")

# ==================================================================
# TAB 1 — PRICE & PERFORMANCE (WITH CANDLESTICKS AND PROPER COMPARISON)
# ==================================================================
with tab1:
    st.subheader("Stock Price Evolution")
    
    if price_data is not None and len(price_data) > 100:  # Need enough data for meaningful charts
        # Make a copy to avoid modifying cached data
        price_data_copy = price_data.copy()
        
        # Calculate moving averages for 7, 50, and 200 days
        # Use min_periods=1 to start calculating from first available data point
        price_data_copy["MA_7"] = price_data_copy["Close"].rolling(window=7, min_periods=1).mean()
        price_data_copy["MA_50"] = price_data_copy["Close"].rolling(window=50, min_periods=1).mean()
        
        # For 200-day MA, we need at least 200 data points for accurate calculation
        if len(price_data_copy) >= 200:
            price_data_copy["MA_200"] = price_data_copy["Close"].rolling(window=200, min_periods=1).mean()
        else:
            # If not enough data, use available data for longer MA
            price_data_copy["MA_200"] = price_data_copy["Close"].rolling(window=min(100, len(price_data_copy)), min_periods=1).mean()
        
        # CHART 1: Palantir Candlestick with 7, 50, 200-day Moving Averages
        fig1 = go.Figure()
        
        # Add candlestick chart
        fig1.add_trace(go.Candlestick(
            x=price_data_copy.index,
            open=price_data_copy['Open'],
            high=price_data_copy['High'],
            low=price_data_copy['Low'],
            close=price_data_copy['Close'],
            name="PLTR Price",
            increasing_line_color='#26a69a',  # Green for up
            decreasing_line_color='#ef5350',  # Red for down
            visible=True
        ))
        
        # Add 7-day moving average
        fig1.add_trace(go.Scatter(
            x=price_data_copy.index,
            y=price_data_copy["MA_7"],
            name="7-Day MA",
            line=dict(color='yellow', width=1.5),
            visible=True
        ))
        
        # Add 50-day moving average
        fig1.add_trace(go.Scatter(
            x=price_data_copy.index,
            y=price_data_copy["MA_50"],
            name="50-Day MA",
            line=dict(color='orange', width=2),
            visible=True
        ))
        
        # Add 200-day moving average
        fig1.add_trace(go.Scatter(
            x=price_data_copy.index,
            y=price_data_copy["MA_200"],
            name="200-Day MA",
            line=dict(color='blue', width=2.5),
            visible=True
        ))
        
        # Add volume as a subplot (optional)
        fig1.add_trace(go.Bar(
            x=price_data_copy.index,
            y=price_data_copy['Volume'],
            name="Volume",
            yaxis="y2",
            marker_color='rgba(100, 100, 100, 0.3)',
            visible='legendonly'  # Hidden by default, can be toggled
        ))

        fig1.update_layout(
            title="Palantir (PLTR) - Candlestick Chart with Moving Averages",
            height=600,
            yaxis_title="Price (USD)",
            xaxis_title="Date",
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            xaxis_rangeslider_visible=True,  # Show range slider at bottom
            plot_bgcolor='rgba(240, 240, 240, 0.8)'
        )
        
        # Add time period selector buttons
        fig1.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(200, 200, 200, 0.3)'
            )
        )

        st.plotly_chart(fig1, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        if len(price_data_copy) > 0:
            current_price = price_data_copy["Close"].iloc[-1]
            current_date = price_data_copy.index[-1].strftime('%Y-%m-%d')
            
            # Calculate price changes
            price_1d_ago = price_data_copy["Close"].iloc[-2] if len(price_data_copy) > 1 else current_price
            price_1w_ago = price_data_copy["Close"].iloc[-7] if len(price_data_copy) > 7 else current_price
            price_1m_ago = price_data_copy["Close"].iloc[-22] if len(price_data_copy) > 22 else current_price
            price_1y_ago = price_data_copy["Close"].iloc[-252] if len(price_data_copy) > 252 else current_price
            
            daily_change = ((current_price - price_1d_ago) / price_1d_ago * 100) if price_1d_ago > 0 else 0
            weekly_change = ((current_price - price_1w_ago) / price_1w_ago * 100) if price_1w_ago > 0 else 0
            monthly_change = ((current_price - price_1m_ago) / price_1m_ago * 100) if price_1m_ago > 0 else 0
            yearly_change = ((current_price - price_1y_ago) / price_1y_ago * 100) if price_1y_ago > 0 else 0
            
            # Calculate 52-week high/low
            if len(price_data_copy) >= 252:
                week_52_data = price_data_copy["Close"].tail(252)
                week_52_high = week_52_data.max()
                week_52_low = week_52_data.min()
            else:
                week_52_high = price_data_copy["High"].max()
                week_52_low = price_data_copy["Low"].min()
            
            col1.metric(
                f"Current Price\n({current_date})", 
                f"${current_price:.2f}",
                f"{daily_change:+.2f}%",
                delta_color="normal"
            )
            col2.metric("Weekly Change", f"{weekly_change:+.1f}%")
            col3.metric("Monthly Change", f"{monthly_change:+.1f}%")
            col4.metric("Yearly Change", f"{yearly_change:+.1f}%")
            
            # Additional metrics
            st.subheader("Key Levels")
            col5, col6, col7, col8 = st.columns(4)
            
            # Current vs Moving Averages
            ma_7_current = price_data_copy["MA_7"].iloc[-1]
            ma_50_current = price_data_copy["MA_50"].iloc[-1]
            ma_200_current = price_data_copy["MA_200"].iloc[-1]
            
            vs_ma7 = ((current_price - ma_7_current) / ma_7_current * 100) if ma_7_current > 0 else 0
            vs_ma50 = ((current_price - ma_50_current) / ma_50_current * 100) if ma_50_current > 0 else 0
            vs_ma200 = ((current_price - ma_200_current) / ma_200_current * 100) if ma_200_current > 0 else 0
            
            col5.metric("vs 7-Day MA", f"{vs_ma7:+.1f}%")
            col6.metric("vs 50-Day MA", f"{vs_ma50:+.1f}%")
            col7.metric("vs 200-Day MA", f"{vs_ma200:+.1f}%")
            col8.metric("52-Week Range", f"${week_52_low:.1f} - ${week_52_high:.1f}")
        
        # Performance comparison with S&P 500
        st.subheader("Performance Comparison: PLTR vs S&P 500")
        
        if sp500_data is not None and len(sp500_data) > 0:
            # Filter data from July 2020 onwards
            start_date = pd.Timestamp("2020-07-01")
            
            # Filter PLTR data
            pltr_filtered = price_data_copy.loc[start_date:].copy()
            
            # Filter S&P 500 data
            sp500_filtered = sp500_data.loc[start_date:].copy()
            
            # Ensure we have data for both
            if len(pltr_filtered) > 0 and len(sp500_filtered) > 0:
                # Get common dates
                common_dates = pltr_filtered.index.intersection(sp500_filtered.index)
                
                if len(common_dates) > 0:
                    # Get closing prices for common dates
                    pltr_close = pltr_filtered.loc[common_dates, "Close"]
                    sp500_close = sp500_filtered.loc[common_dates, "Close"]
                    
                    # Normalize both series to 100 at the starting point
                    pltr_normalized = (pltr_close / pltr_close.iloc[0]) * 100
                    sp500_normalized = (sp500_close / sp500_close.iloc[0]) * 100
                    
                    # Create comparison DataFrame
                    comparison_df = pd.DataFrame({
                        'Date': common_dates,
                        'Palantir (PLTR)': pltr_normalized.values,
                        'S&P 500': sp500_normalized.values
                    })
                    
                    # CHART 2: Performance Comparison (using plotly.graph_objects for better control)
                    fig2 = go.Figure()
                    
                    # Add PLTR line
                    fig2.add_trace(go.Scatter(
                        x=comparison_df['Date'],
                        y=comparison_df['Palantir (PLTR)'],
                        name='Palantir (PLTR)',
                        mode='lines',
                        line=dict(color='blue', width=3),
                        hovertemplate='<b>PLTR</b><br>Date: %{x}<br>Normalized: %{y:.1f}<extra></extra>'
                    ))
                    
                    # Add S&P 500 line
                    fig2.add_trace(go.Scatter(
                        x=comparison_df['Date'],
                        y=comparison_df['S&P 500'],
                        name='S&P 500',
                        mode='lines',
                        line=dict(color='gray', width=2, dash='dash'),
                        hovertemplate='<b>S&P 500</b><br>Date: %{x}<br>Normalized: %{y:.1f}<extra></extra>'
                    ))
                    
                    # Add shaded area for outperformance/underperformance
                    # Find where PLTR outperforms S&P 500
                    pltr_higher = comparison_df['Palantir (PLTR)'] > comparison_df['S&P 500']
                    sp500_higher = comparison_df['S&P 500'] > comparison_df['Palantir (PLTR)']
                    
                    # Add fill between lines
                    fig2.add_trace(go.Scatter(
                        x=comparison_df['Date'],
                        y=comparison_df['Palantir (PLTR)'],
                        fill='tonexty',
                        fillcolor='rgba(0, 100, 255, 0.2)',
                        line=dict(width=0),
                        showlegend=False,
                        name='PLTR Outperforming'
                    ))
                    
                    fig2.update_layout(
                        title='Normalized Performance Comparison: PLTR vs S&P 500 (Base = 100)',
                        yaxis_title='Normalized Performance (Base = 100)',
                        height=500,
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
                                    dict(count=6, label="6M", step="month", stepmode="backward"),
                                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                                    dict(count=2, label="2Y", step="year", stepmode="backward"),
                                    dict(step="all", label="All")
                                ])
                            )
                        )
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Calculate and display performance statistics
                    pltr_total_return = ((pltr_normalized.iloc[-1] - 100) / 100) * 100
                    sp500_total_return = ((sp500_normalized.iloc[-1] - 100) / 100) * 100
                    outperformance = pltr_total_return - sp500_total_return
                    
                    # Additional metrics
                    pltr_high = pltr_normalized.max()
                    pltr_low = pltr_normalized.min()
                    sp500_high = sp500_normalized.max()
                    sp500_low = sp500_normalized.min()
                    
                    st.subheader("Performance Statistics (Since July 2020)")
                    col1, col2, col3, col4 = st.columns(4)
                    
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
                    col4.metric(
                        "Analysis Period", 
                        f"{len(common_dates):,} days"
                    )
                    
                    # More detailed statistics
                    st.subheader("Detailed Performance Analysis")
                    col5, col6, col7, col8 = st.columns(4)
                    
                    # Calculate annualized returns
                    years_elapsed = len(common_dates) / 252  # Approximate trading days per year
                    pltr_annualized = ((1 + pltr_total_return/100) ** (1/years_elapsed) - 1) * 100 if years_elapsed > 0 else 0
                    sp500_annualized = ((1 + sp500_total_return/100) ** (1/years_elapsed) - 1) * 100 if years_elapsed > 0 else 0
                    
                    col5.metric("PLTR Annualized", f"{pltr_annualized:+.1f}%")
                    col6.metric("S&P 500 Annualized", f"{sp500_annualized:+.1f}%")
                    col7.metric("PLTR Max (Normalized)", f"{pltr_high:.1f}")
                    col8.metric("S&P 500 Max", f"{sp500_high:.1f}")
                    
                    # Correlation analysis
                    correlation = pltr_normalized.corr(sp500_normalized)
                    
                    st.info(f"📊 **Correlation Analysis**: PLTR and S&P 500 have a correlation coefficient of **{correlation:.3f}** " +
                           f"({'highly correlated' if correlation > 0.7 else 'moderately correlated' if correlation > 0.3 else 'weakly correlated'}) " +
                           f"since July 2020.")
                    
                else:
                    st.warning("No overlapping trading dates between PLTR and S&P 500 data after July 2020")
            else:
                st.warning(f"Insufficient data for comparison since July 2020. PLTR: {len(pltr_filtered)} days, S&P 500: {len(sp500_filtered)} days")
        else:
            st.warning("S&P 500 comparison data not available")
            
    else:
        st.warning("Insufficient price data available for detailed analysis")
        if price_data is not None:
            st.write(f"Only {len(price_data)} data points available")

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

# Add requirements note at the bottom
st.sidebar.markdown("---")
st.sidebar.caption("Note: Data sourced from Yahoo Finance. May be delayed or incomplete.")

