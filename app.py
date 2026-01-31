import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Tax-Efficient Quant Optimizer", layout="wide")
st.title("🤖 AI Tax-Aware Portfolio Optimizer")
st.markdown("""
This tool uses **Modern Portfolio Theory (MPT)** to find the mathematical "sweet spot" for your investments.
It balances **Risk vs. Reward** while accounting for **Indian Capital Gains Taxes**.
""")

# --- 2. SIDEBAR INPUTS (The "Tax" & "Quant" Parameters) ---
st.sidebar.header("⚙️ Portfolio Settings")

# PRE-DEFINED LIST OF TOP INDIAN STOCKS (Nifty 50 Giants)
# This prevents typos and ensures data exists.
nifty_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", 
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS",
    "WIPRO.NS", "NESTLEIND.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS"
]

# DROPDOWN MULTI-SELECT
selected_tickers = st.sidebar.multiselect(
    "Select Assets for Portfolio", 
    options=nifty_stocks,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"] # Default selection
)

# Tax Inputs
st.sidebar.header("Tax Regime Assumptions")
tax_rate = st.sidebar.slider("Est. Effective Tax Rate (%)", 0, 30, 10, help="Approx 12.5% for Long Term Gains > 1.25L")
tax_impact = tax_rate / 100

# Date Range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- 3. DATA FETCHING ENGINE ---
@st.cache_data
def get_stock_data(tickers, start, end):
    if not tickers: return pd.DataFrame() # Handle empty selection
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

# --- MAIN APP LOGIC ---
if st.sidebar.button("Run Optimization"):
    if not selected_tickers:
        st.error("⚠️ Please select at least 2 stocks from the sidebar to create a portfolio.")
    else:
        with st.spinner('🤖 Crunching numbers, simulating 5,000 market scenarios...'):
            
            try:
                # Get Data
                df = get_stock_data(selected_tickers, start_date, end_date)
                
                if df.empty:
                    st.error("No data returned. Try different dates.")
                else:
                    # Daily Returns
                    daily_returns = df.pct_change()
                    
                    # --- 4. MONTE CARLO SIMULATION ---
                    num_portfolios = 5000
                    results = np.zeros((4, num_portfolios)) 
                    weights_record = []

                    mean_daily_returns = daily_returns.mean()
                    cov_matrix = daily_returns.cov()

                    for i in range(num_portfolios):
                        weights = np.random.random(len(selected_tickers))
                        weights /= np.sum(weights)
                        weights_record.append(weights)

                        # Annualized Return & Risk
                        portfolio_return = np.sum(mean_daily_returns * weights) * 252
                        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                        
                        # Sharpe Ratio (Assuming 6% Risk Free Rate for India)
                        sharpe_ratio = (portfolio_return - 0.06) / portfolio_std_dev

                        # Tax Calc
                        post_tax_return = portfolio_return * (1 - tax_impact)

                        results[0,i] = portfolio_return
                        results[1,i] = portfolio_std_dev
                        results[2,i] = sharpe_ratio
                        results[3,i] = post_tax_return

                    sim_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe', 'Post_Tax_Return'])
                    
                    # Optimal Portfolio
                    max_sharpe_idx = sim_df['Sharpe'].idxmax()
                    max_sharpe_port = sim_df.iloc[max_sharpe_idx]
                    optimal_weights = weights_record[max_sharpe_idx]

                    # --- 5. VISUALIZATION ---
                    
                    # TOP ROW: METRICS
                    st.success("Optimization Complete! Here is your optimal strategy:")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("💰 Expected Return (Yearly)", f"{max_sharpe_port['Return']:.1%}")
                    m2.metric("📉 Risk (Volatility)", f"{max_sharpe_port['Risk']:.1%}")
                    m3.metric("💸 Tax Drag", f"-{(max_sharpe_port['Return'] - max_sharpe_port['Post_Tax_Return']):.1%}", help="Return lost to taxes")
                    m4.metric("✅ Post-Tax Return", f"{max_sharpe_port['Post_Tax_Return']:.1%}")

                    # MIDDLE ROW: CHARTS
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("1️⃣ The Efficient Frontier")
                        fig = px.scatter(
                            sim_df, x='Risk', y='Return', color='Sharpe',
                            title="Risk vs Return of 5,000 Possible Portfolios",
                            color_continuous_scale='RdYlGn' # Red to Green color scale
                        )
                        fig.add_scatter(x=[max_sharpe_port['Risk']], y=[max_sharpe_port['Return']], 
                                        mode='markers', marker=dict(color='red', size=20, symbol='star'),
                                        name='★ Optimal Portfolio')
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("2️⃣ Ideal Allocation")
                        allocation_df = pd.DataFrame({'Asset': selected_tickers, 'Weight': optimal_weights})
                        # Filter out tiny weights (<1%) for cleaner chart
                        allocation_df = allocation_df[allocation_df['Weight'] > 0.01]
                        
                        fig_pie = px.pie(allocation_df, values='Weight', names='Asset', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                        st.plotly_chart(fig_pie, use_container_width=True)

                    # --- 6. AI ANALYST COMMENTARY ---
                    st.markdown("---")
                    st.subheader("🧠 The AI Analyst's Review")
                    
                    # Logic to generate text
                    best_asset = allocation_df.sort_values(by="Weight", ascending=False).iloc[0]
                    risk_level = "High" if max_sharpe_port['Risk'] > 0.20 else "Moderate" if max_sharpe_port['Risk'] > 0.12 else "Low"
                    
                    st.write(f"""
                    Based on the simulation of **{start_date.year} to {end_date.year}** data:
                    
                    * **Winning Strategy:** The model suggests heavily weighting **{best_asset['Asset']}** ({best_asset['Weight']:.1%} allocation). This stock has historically provided the best returns relative to its stability.
                    * **Risk Profile:** This portfolio has a **{risk_level}** risk profile ({max_sharpe_port['Risk']:.1%} volatility). 
                    * **Tax Efficiency:** You are losing approximately **{(max_sharpe_port['Return'] - max_sharpe_port['Post_Tax_Return']):.1%}** of your gains to taxes annually. 
                    """)
                    
                    with st.expander("📚 Terminology Guide (Click to Learn)"):
                        st.write("""
                        * **Sharpe Ratio:** The 'Efficiency Score' of a portfolio. A higher number means you are getting more return for every unit of risk you take.
                        * **Volatility:** How much the portfolio value bounces up and down. Lower is generally better for peace of mind.
                        * **Efficient Frontier:** The curve on the graph. Any portfolio ON the line is 'Efficient'. Anything below it is 'Bad' (taking risk without getting enough return).
                        """)

            except Exception as e:
                st.error(f"Something went wrong: {e}")

else:
    st.info("👈 Select your stocks from the dropdown and click 'Run Optimization' to start.")
