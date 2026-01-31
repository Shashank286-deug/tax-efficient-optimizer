import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Tax-Efficient Quant Optimizer", layout="wide")
st.title("📈 Tax-Aware Portfolio Optimizer")
st.markdown("""
This tool uses **Modern Portfolio Theory (MPT)** to find the optimal asset allocation, 
but adjusts for **Capital Gains Tax** to show true realized returns.
""")

# --- 2. SIDEBAR INPUTS (The "Tax" & "Quant" Parameters) ---
st.sidebar.header("Portfolio Settings")

# User inputs tickers (e.g., Reliance, TCS, Infosys)
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", "RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS")
tickers = [t.strip() for t in tickers_input.split(',')]

# Tax Inputs (Your "Tax Analyst" Speciality)
st.sidebar.header("Tax Regime Assumptions")
tax_rate = st.sidebar.slider("Est. Effective Tax Rate (%)", 0, 30, 10, help="e.g. 12.5% for LTCG or 20% for STCG")
tax_impact = tax_rate / 100

# Date Range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- 3. DATA FETCHING ENGINE ---
@st.cache_data # Caches data so it doesn't reload on every click
def get_stock_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

if st.sidebar.button("Run Optimization"):
    with st.spinner('Fetching market data and running Monte Carlo simulations...'):
        
        # Get Data
        try:
            df = get_stock_data(tickers, start_date, end_date)
            
            # Check if data is empty
            if df.empty:
                st.error("No data found. Please check the ticker symbols.")
            else:
                # Calculate Daily Returns
                daily_returns = df.pct_change()
                
                # --- 4. THE QUANT MATH (Monte Carlo Simulation) ---
                # We will generate 5,000 random portfolios to see which is best
                num_portfolios = 5000
                results = np.zeros((4, num_portfolios)) # Storing: Return, Volatility, Sharpe, Tax-Adj Return
                weights_record = []

                # Annualizing factors (252 trading days in a year)
                mean_daily_returns = daily_returns.mean()
                cov_matrix = daily_returns.cov()

                for i in range(num_portfolios):
                    # Generate random weights for each stock
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights) # Normalize so sum = 1 (100%)
                    weights_record.append(weights)

                    # Portfolio Expected Return (Annualized)
                    portfolio_return = np.sum(mean_daily_returns * weights) * 252
                    
                    # Portfolio Volatility (Risk)
                    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                    
                    # Portfolio Sharpe Ratio (assuming 0% risk-free for simplicity)
                    sharpe_ratio = portfolio_return / portfolio_std_dev

                    # --- THE "TAX EDGE" ---
                    # Calculate Post-Tax Return
                    # We assume the gains are realized and taxed at the user's rate
                    post_tax_return = portfolio_return * (1 - tax_impact)

                    results[0,i] = portfolio_return
                    results[1,i] = portfolio_std_dev
                    results[2,i] = sharpe_ratio
                    results[3,i] = post_tax_return

                # Create a DataFrame for visualization
                sim_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe', 'Post_Tax_Return'])
                
                # Find the "Max Sharpe" Portfolio (The mathematically "best" one)
                max_sharpe_idx = sim_df['Sharpe'].idxmax()
                max_sharpe_port = sim_df.iloc[max_sharpe_idx]
                optimal_weights = weights_record[max_sharpe_idx]

                # --- 5. VISUALIZATION ---
                
                # Layout: 2 Columns
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Efficient Frontier (Risk vs Return)")
                    # Plotting the "cloud" of portfolios
                    fig = px.scatter(
                        sim_df, x='Risk', y='Return', color='Sharpe',
                        title="Monte Carlo Simulation (5,000 Portfolios)",
                        labels={'Risk': 'Volatility (Risk)', 'Return': 'Annualized Return'},
                        color_continuous_scale='Viridis'
                    )
                    # Add a red marker for the optimal portfolio
                    fig.add_scatter(x=[max_sharpe_port['Risk']], y=[max_sharpe_port['Return']], 
                                    mode='markers', marker=dict(color='red', size=15, symbol='star'),
                                    name='Optimal Portfolio')
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Optimal Allocation")
                    # Create a pie chart of the best weights
                    allocation_df = pd.DataFrame({'Asset': tickers, 'Weight': optimal_weights})
                    fig_pie = px.pie(allocation_df, values='Weight', names='Asset', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)

                    st.info(f"""
                    **Performance Metrics:**
                    - 🟢 Pre-Tax Return: {max_sharpe_port['Return']:.2%}
                    - 🔴 Tax Drag: {(max_sharpe_port['Return'] - max_sharpe_port['Post_Tax_Return']):.2%}
                    - 🟢 **Post-Tax Return: {max_sharpe_port['Post_Tax_Return']:.2%}**
                    - ⚠️ Risk (Vol): {max_sharpe_port['Risk']:.2%}
                    """)

                # --- 6. RAW DATA ---
                with st.expander("See Raw Data (Correlation Matrix)"):
                    st.write("Correlation between assets (important for diversification):")
                    st.dataframe(daily_returns.corr().style.background_gradient(cmap='coolwarm'))

        except Exception as e:
            st.error(f"An error occurred: {e}")

else:
    st.info("👈 Enter stock tickers in the sidebar (e.g., RELIANCE.NS for NSE) and hit 'Run Optimization'")
