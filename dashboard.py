import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Palantir Financial Analysis Dashboard")

# Fetch data
pltr = yf.Ticker("PLTR")
historical_data = pltr.history(period="max")
financials = pltr.financials
balance_sheet = pltr.balance_sheet
cash_flow = pltr.cashflow

# ------------------------------------------------------------------------------------
# 1. Historical Data
# ------------------------------------------------------------------------------------
st.subheader("Historical Stock Prices")
st.line_chart(historical_data['Close'])

# ------------------------------------------------------------------------------------
# 2. Financial Ratios
# ------------------------------------------------------------------------------------
st.subheader("Financial Ratios")

# Profitability Ratios
revenue = financials.loc['Total Revenue']
net_income = financials.loc['Net Income']
gross_profit = financials.loc['Gross Profit']
shareholder_equity = balance_sheet.loc['Stockholders Equity']

gross_profit_margin = (gross_profit / revenue) * 100
net_profit_margin = (net_income / revenue) * 100
roe = (net_income / shareholder_equity) * 100

# Liquidity Ratios
st.subheader("Liquidity Ratios")
current_assets = balance_sheet.loc['Current Assets']
current_liabilities = balance_sheet.loc['Current Liabilities']
inventory = balance_sheet.loc['Inventory'] if 'Inventory' in balance_sheet.index else 0

current_ratio = current_assets / current_liabilities
quick_ratio = (current_assets - inventory) / current_liabilities

st.write(f"Current Ratio: {current_ratio.iloc[0]:.2f}")
st.write(f"Quick Ratio: {quick_ratio.iloc[0]:.2f}")

# Efficiency Ratios
st.subheader("Efficiency Ratios")
total_assets = balance_sheet.loc['Total Assets']
receivables = balance_sheet.loc['Accounts Receivable']

asset_turnover = revenue / total_assets
receivables_turnover = revenue / receivables

st.write(f"Asset Turnover: {asset_turnover.iloc[0]:.2f}")
st.write(f"Receivables Turnover: {receivables_turnover.iloc[0]:.2f}")

# Leverage Ratios
st.subheader("Leverage Ratios")
total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest']
ebit = financials.loc['EBIT']
interest_expense = financials.loc['Interest Expense']

debt_to_equity = total_liabilities / shareholder_equity
interest_coverage = ebit / interest_expense

st.write(f"Debt-to-Equity Ratio: {debt_to_equity.iloc[0]:.2f}")
st.write(f"Interest Coverage Ratio: {interest_coverage.iloc[0]:.2f}")

# Market Ratios
st.subheader("Market Ratios")
market_cap = pltr.info['marketCap']
book_value = shareholder_equity.iloc[0]
shares_outstanding = pltr.info['sharesOutstanding']

pe_ratio = market_cap / net_income.iloc[0]
pb_ratio = market_cap / book_value

st.write(f"P/E Ratio: {pe_ratio:.2f}")
st.write(f"Price-to-Book (P/B) Ratio: {pb_ratio:.2f}")

# Conclusion
st.write("""
- **High-Margin Business Model**: Palantir has a high-margin business model, reflecting strong pricing power and operational efficiency.
- **Excellent Liquidity**: The company maintains excellent liquidity, reducing short-term financial risk.
- **Low Debt Levels**: Low debt levels enhance financial stability and reduce the risk of insolvency.
- **Future Focus**: While profitable, Palantir should focus on sustaining and improving net profit margins while efficiently deploying its excess liquidity to drive future growth.
""")

# Strengths Section
st.header("Strengths")
st.markdown("""
- **Excellent Liquidity**: Both current and quick ratios are strong, providing stability and effective risk management.
- **Low Debt Level**: This adds to the company’s financial stability.
- **Strong Receivables Turnover**: Reflects good operational efficiency in managing receivables.
""")

# Concerns Section
st.header("Concerns")
st.markdown("""
- **Low Asset Turnover**: Despite strong liquidity and low debt, Palantir’s asset turnover is relatively low, indicating that there's room for improvement in asset usage efficiency.
- **High P/E and P/B Ratios**: These ratios suggest that the stock may be overvalued, or that investors are expecting very high growth in the future.
""")

# ------------------------------------------------------------------------------------
# 3. Candlestick Charts with Moving Averages and Volume
# ------------------------------------------------------------------------------------
st.subheader("Candlestick Chart with Moving Averages")

# Calculate moving averages
historical_data['MA_50'] = historical_data['Close'].rolling(window=50).mean()
historical_data['MA_200'] = historical_data['Close'].rolling(window=200).mean()

# Create a candlestick chart
candlestick = go.Figure(data=[go.Candlestick(
    x=historical_data.index,
    open=historical_data['Open'],
    high=historical_data['High'],
    low=historical_data['Low'],
    close=historical_data['Close']
)])

# Add moving averages
candlestick.add_trace(go.Scatter(x=historical_data.index, y=historical_data['MA_50'], name="50-Day MA", line=dict(color='orange')))
candlestick.add_trace(go.Scatter(x=historical_data.index, y=historical_data['MA_200'], name="200-Day MA", line=dict(color='green')))

# Update layout
candlestick.update_layout(
    title="Palantir Candlestick Chart with Moving Averages",
    xaxis_title="Date",
    yaxis_title="Price (USD)"
)

# Display the chart
st.plotly_chart(candlestick)

# ------------------------------------------------------------------------------------
# 4. Market Comparison with S&P 500
# ------------------------------------------------------------------------------------
st.subheader("Palantir vs. S&P 500 Performance (Since September 30, 2020)")

# Fetch S&P 500 data
sp500 = yf.Ticker("^GSPC")
sp500_data = sp500.history(period="max")

# Filter data from September 30, 2020
start_date = "2020-09-30"
historical_data = historical_data.loc[start_date:]
sp500_data = sp500_data.loc[start_date:]

# Normalize prices to compare performance
historical_data['Normalized Close'] = historical_data['Close'] / historical_data['Close'].iloc[0]
sp500_data['Normalized Close'] = sp500_data['Close'] / sp500_data['Close'].iloc[0]

# Combine data into a single DataFrame
comparison_data = pd.DataFrame({
    "Date": historical_data.index,
    "Palantir (PLTR)": historical_data['Normalized Close'],
    "S&P 500": sp500_data['Normalized Close']
})

# Create an interactive line plot
fig = px.line(comparison_data, x="Date", y=["Palantir (PLTR)", "S&P 500"],
              title="Palantir vs. S&P 500 Performance (Since September 30, 2020)",
              labels={"value": "Normalized Price", "variable": "Index"})

# Display the plot
st.plotly_chart(fig)

st.write("""
Palantir’s stock has shown significant volatility since its IPO, with periods of strong outperformance during tech rallies but also underperformance when broader market sentiment shifts, especially in light of its business model and profitability challenges. Its performance relative to the S&P 500 is a reflection of these factors.
""")

# ------------------------------------------------------------------------------------
# 5. DCF Valuation
# ------------------------------------------------------------------------------------
st.subheader("Discounted Cash Flow (DCF) Valuation")

# Fetch free cash flow data (ensure this is dynamic and up-to-date)
free_cash_flow = cash_flow.loc['Free Cash Flow']

# Function to calculate DCF
def calculate_dcf(free_cash_flow, discount_rate, growth_rate, years):
    forecasted_fcf = []
    for t in range(1, years + 1):
        fcf = free_cash_flow * (1 + growth_rate) ** t
        forecasted_fcf.append(fcf)

    # Calculate terminal value
    terminal_value = forecasted_fcf[-1] * (1 + growth_rate) / (discount_rate - growth_rate)

    # Discount cash flows
    dcf_value = 0
    for t, fcf in enumerate(forecasted_fcf, start=1):
        dcf_value += fcf / (1 + discount_rate) ** t

    # Add terminal value
    dcf_value += terminal_value / (1 + discount_rate) ** years

    return dcf_value

# Streamlit app
st.title("Discounted Cash Flow (DCF) Calculator for Palantir")

# User inputs
free_cash_flow = st.number_input("Enter Free Cash Flow (in dollars):", value=1e9)
discount_rate = st.slider("Discount Rate (%):", 0.0, 30.0, 10.0) / 100  # Converting to decimal
growth_rate = st.slider("Growth Rate (%):", 0.0, 20.0, 5.0) / 100  # Converting to decimal
years = st.slider("Forecast Period (Years):", 1, 10, 5)

# Calculate DCF value
dcf_value = calculate_dcf(free_cash_flow, discount_rate, growth_rate, years)

# Get Palantir's market capitalization from Yahoo Finance
ticker = 'PLTR'
palantir = yf.Ticker(ticker)
market_cap = palantir.info['marketCap']

# Display results
st.subheader("Results:")
st.write(f"**Discounted Cash Flow (DCF) Value:** ${dcf_value / 1e9:.2f} Billion")
st.write(f"**Palantir's Current Market Capitalization:** ${market_cap / 1e9:.2f} Billion")

# Compare DCF and Market Cap
if dcf_value > market_cap:
    st.write("The DCF value suggests that Palantir is undervalued.")
elif dcf_value < market_cap:
    st.write("The DCF value suggests that Palantir is overvalued.")
else:
    st.write("The DCF value is in line with Palantir's market capitalization.")

st.write("""
Palantir's market price appears to be overvalued relative to the DCF-based valuation. While the company may have strong growth potential, investors should be cautious and aware of the risks associated with high valuations that are not backed by current cash flow projections.
""")

# ------------------------------------------------------------------------------------
# 6. Stock Price Prediction
# ------------------------------------------------------------------------------------
st.subheader("Stock Price Prediction")

# Prepare data for machine learning
historical_data['Returns'] = historical_data['Close'].pct_change()
historical_data = historical_data.dropna()
X = historical_data[['Open', 'High', 'Low', 'Volume']]
y = historical_data['Close']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display evaluation metrics
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R-squared: {r2:.2f}")

# Plot actual vs predicted prices
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_test, predictions, color='blue', label="Predictions")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label="Ideal Line")
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted Prices")
ax.legend()
ax.grid()
st.pyplot(fig)

st.title("Real-Time Market Share Comparison")

# Define the tickers for Palantir and its competitors
tickers = {
    "Palantir": "PLTR",
    "Salesforce": "CRM",  
    "Microsoft": "MSFT",  
    "Google": "GOOGL",
    "Amazon": "AMZN",   
    "IBM": "IBM",    
    "Oracle": "ORCL",  
    "SAP": "SAP",   
    "Snowflake": "SNOW", 
    "Slack": "WORK"   
}

# Fetch real-time market caps
market_caps = {}
for company, ticker in tickers.items():
    stock = yf.Ticker(ticker)
    market_cap = stock.info.get('marketCap', None)
    if market_cap:
        market_caps[company] = market_cap

# Convert market caps to a DataFrame
df = pd.DataFrame(list(market_caps.items()), columns=['Company', 'Market Cap'])

# Calculate market share percentages
total_market_cap = df['Market Cap'].sum()
df['Market Share'] = (df['Market Cap'] / total_market_cap) * 100

# Sort by market share
df = df.sort_values(by='Market Share', ascending=True)  # Sort ascending for better horizontal bar visualization

# Plot the horizontal bar chart
plt.figure(figsize=(10, 6))
colors = ['red' if company == 'Palantir' else 'blue' for company in df['Company']]
plt.barh(df['Company'], df['Market Share'], color=colors)
plt.title('Real-Time Market Share Comparison (Public Companies)')
plt.xlabel('Market Share (%)')
plt.ylabel('Company')
plt.tight_layout()

# Display the chart in Streamlit
st.pyplot(plt)

# Display the raw data in a table (optional)
st.subheader("Raw Data")
st.dataframe(df)

st.title("Insights on Palantir's Market Position")

# Niche Market Presence
st.header("Niche Market Presence")
st.write("""
Palantir is a specialized data analytics and AI-driven software company, primarily serving government agencies and large enterprises.  
Its smaller market share reflects its focus on high-value, security-sensitive contracts rather than mass-market cloud solutions.
""")

# Competition with Established Giants
st.header("Competition with Established Giants")
st.write("""
Companies like **Microsoft, Amazon, and Google** dominate the market with their broad cloud computing and enterprise solutions,  
making it challenging for Palantir to capture a significant portion of the market.
""")

# Potential for Growth
st.header("Potential for Growth")
st.write("""
Despite its smaller share, Palantir has a strong reputation for advanced AI analytics and big data solutions,  
which could drive future expansion, particularly in **defense, finance, and healthcare** sectors.
""")

# Investment Perspective
st.header("Investment Perspective")
st.write("""
Investors looking at Palantir should consider its unique positioning in the AI and data analytics space,  
but also recognize the challenge it faces in scaling its commercial sector presence against well-established cloud competitors.
""")


st.title("Personal Investment Section - Palantir Performance")

st.write("""
I have been closely monitoring Palantir's performance since before its public listing in 2020. After observing a favorable market correction, I decided to make an investment on May 11, 2022, recognizing it as an optimal entry point. I purchased shares at a price of $6.89 USD. Below is the performance of my investment to date.
""")

# Define investment details
purchase_price = 6.89  # My purchase price
purchase_date = "2022-05-11"  # My purchase date
current_price = historical_data['Close'].iloc[-1]  # Latest closing price
roi = ((current_price - purchase_price) / purchase_price) * 100  # Calculate ROI

# Display investment details
st.write(f"Purchased at: ${purchase_price} on {purchase_date}")
st.write(f"Current Price: ${current_price:.2f}")
st.write(f"ROI: {roi:.2f}%")

# Plot investment performance
investment_data = historical_data.loc[purchase_date:]
st.line_chart(investment_data['Close'])

# ------------------------------------------------------------------------------------
# Additional Analysis
# ------------------------------------------------------------------------------------
st.subheader("SWOT Analysis")

# Display SWOT Analysis
st.write("""
### Strengths
1. **Strong Government Contracts:** Palantir has long-standing relationships with government agencies, providing a stable revenue stream.
2. **Advanced Data Analytics:** The company’s platforms (e.g., Foundry, Gotham) are highly sophisticated and tailored for complex data analysis.
3. **High Barriers to Entry:** Palantir’s proprietary technology and expertise create significant barriers for competitors.

### Weaknesses
1. **Dependence on Government Contracts:** A large portion of revenue comes from government contracts, making the company vulnerable to policy changes.
2. **High Customer Acquisition Costs:** Acquiring new clients is expensive and time-consuming.
3. **Limited Profitability:** Palantir has struggled to achieve consistent profitability.

### Opportunities
1. **Expansion into Commercial Markets:** There is significant potential for growth in industries like healthcare, finance, and manufacturing.
2. **International Growth:** Palantir can expand its presence in international markets.
3. **AI and Machine Learning Integration:** Leveraging AI/ML can enhance its platforms and attract more clients.

### Threats
1. **Competition:** Competitors like Snowflake and Microsoft are entering the data analytics space.
2. **Regulatory Risks:** Increased scrutiny on data privacy and security could impact operations.
3. **Economic Downturns:** Reduced spending by governments and enterprises during economic downturns could hurt revenue.
""")

st.subheader("Conclusion")

# Display Conclusion
st.write("""
Palantir Technologies (PLTR) is a unique player in the data analytics space, with a strong focus on government contracts and complex data solutions. While the company has demonstrated significant strengths, such as its advanced technology and high barriers to entry, it also faces challenges like dependence on government revenue and limited profitability.

The financial analysis highlights Palantir’s potential for growth, particularly in commercial markets and international expansion. However, risks such as competition, regulatory changes, and economic downturns must be carefully managed.

From an investment perspective, Palantir’s stock has shown volatility, but its long-term potential remains promising, especially if it can diversify its revenue streams and achieve consistent profitability. Investors should closely monitor the company’s ability to execute its growth strategies and adapt to market changes.
""")