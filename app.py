import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

# --- 1. CONFIGURATION & TITLE ---
st.set_page_config(page_title="Tax-Efficient Quant Optimizer", layout="wide")
st.title("🤖 AI Tax-Aware Portfolio Optimizer")
st.markdown("### The Bridge Between Quantitative Finance & Tax Strategy")

# --- 2. SIDEBAR INPUTS ---
st.sidebar.header("⚙️ Portfolio Settings")

nifty_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LICI.NS", 
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS",
    "WIPRO.NS", "NESTLEIND.NS", "TATASTEEL.NS", "NTPC.NS", "POWERGRID.NS",
    "M&M.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS"
]

selected_tickers = st.sidebar.multiselect(
    "Select Assets (Min 2)", 
    options=nifty_stocks,
    default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
)

tax_rate = st.sidebar.slider("Est. Effective Tax Rate (%)", 0, 30, 10)
tax_impact = tax_rate / 100
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# --- 3. DATA ENGINE ---
@st.cache_data
def get_stock_data(tickers, start, end):
    if not tickers: return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end)['Close']
    return data

# --- TABS LAYOUT ---
tab1, tab2 = st.tabs(["🚀 Run Optimizer", "📘 Quant Concepts & Formulas"])

with tab1:
    if st.sidebar.button("Run Optimization"):
        if len(selected_tickers) < 2:
            st.error("⚠️ Please select at least 2 stocks.")
        else:
            with st.spinner('Calculating Efficient Frontier, Risk Metrics & Correlations...'):
                try:
                    df = get_stock_data(selected_tickers, start_date, end_date)
                    
                    if df.empty:
                        st.error("No data returned. Try different dates.")
                    else:
                        daily_returns = df.pct_change().dropna()
                        
                        # --- MONTE CARLO SIMULATION ---
                        num_portfolios = 5000
                        results = np.zeros((6, num_portfolios))
                        weights_record = []

                        mean_daily_returns = daily_returns.mean()
                        cov_matrix = daily_returns.cov()

                        for i in range(num_portfolios):
                            weights = np.random.random(len(selected_tickers))
                            weights /= np.sum(weights)
                            weights_record.append(weights)

                            # Portfolio Return & Volatility
                            port_return = np.sum(mean_daily_returns * weights) * 252
                            port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                            
                            # Sharpe Ratio
                            sharpe_ratio = (port_return - 0.06) / port_std_dev

                            # Advanced Risk Metrics (VaR & CVaR)
                            sim_returns = daily_returns.dot(weights)
                            var_95 = np.percentile(sim_returns, 5) # 5th percentile
                            cvar_95 = sim_returns[sim_returns <= var_95].mean()

                            # Tax Impact
                            post_tax_return = port_return * (1 - tax_impact)

                            results[0,i] = port_return
                            results[1,i] = port_std_dev
                            results[2,i] = sharpe_ratio
                            results[3,i] = post_tax_return
                            results[4,i] = var_95
                            results[5,i] = cvar_95

                        # Store results
                        columns = ['Return', 'Risk', 'Sharpe', 'Post_Tax_Return', 'VaR_95', 'CVaR_95']
                        sim_df = pd.DataFrame(results.T, columns=columns)
                        
                        # Find Optimal Portfolio
                        max_sharpe_idx = sim_df['Sharpe'].idxmax()
                        max_sharpe_port = sim_df.iloc[max_sharpe_idx]
                        optimal_weights = weights_record[max_sharpe_idx]

                        # --- DISPLAY METRICS ---
                        st.success("Optimization Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("💰 Annual Return", f"{max_sharpe_port['Return']:.1%}", delta=f"Post-Tax: {max_sharpe_port['Post_Tax_Return']:.1%}")
                        col2.metric("📉 Volatility (Risk)", f"{max_sharpe_port['Risk']:.1%}")
                        col3.metric("⚠️ 95% VaR", f"{max_sharpe_port['VaR_95']:.2%}", help="Minimum loss on the worst 5% of days")
                        col4.metric("☢️ 95% CVaR", f"{max_sharpe_port['CVaR_95']:.2%}", help="Average loss when things go really bad")

                        # --- CHARTS ---
                        c1, c2 = st.columns([2,1])
                        
                        with c1:
                            st.subheader("Efficient Frontier")
                            fig = px.scatter(
                                sim_df, x='Risk', y='Return', color='Sharpe',
                                title="Risk vs Return Landscape", color_continuous_scale='Viridis'
                            )
                            fig.add_scatter(x=[max_sharpe_port['Risk']], y=[max_sharpe_port['Return']], 
                                            mode='markers', marker=dict(color='red', size=20, symbol='star'),
                                            name='Optimal Portfolio')
                            st.plotly_chart(fig, use_container_width=True)

                        with c2:
                            st.subheader("Optimal Allocation")
                            alloc_df = pd.DataFrame({'Asset': selected_tickers, 'Weight': optimal_weights})
                            alloc_df = alloc_df[alloc_df['Weight'] > 0.01] # Filter small weights
                            
                            # Sort for better summary later
                            alloc_df = alloc_df.sort_values(by="Weight", ascending=False)
                            
                            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True)

                        # --- 6. EXPLANATION & REAL WORLD EXAMPLE ---
                        st.divider()
                        st.subheader("📝 Analysis & Real-World Example")
                        
                        ex_col1, ex_col2 = st.columns(2)
                        
                        with ex_col1:
                            st.markdown("#### 1. What is happening?")
                            top_stock = alloc_df.iloc[0]
                            st.write(f"""
                            The algorithm ran **5,000 simulations** of potential portfolios. 
                            It found that to get the highest return for the lowest risk, you should allocate **{top_stock['Weight']:.1%}** of your money to **{top_stock['Asset']}**.
                            """)
                            
                            st.markdown("#### 2. Risk Summary")
                            st.write(f"""
                            * **Volatility ({max_sharpe_port['Risk']:.1%}):** This is the "bounce." A higher number means your account balance swings wildly.
                            * **VaR ({max_sharpe_port['VaR_95']:.2%}):** This is the "Safety Net." It means on 95 out of 100 days, your daily loss will NOT exceed {max_sharpe_port['VaR_95']:.2%}.
                            """)

                        with ex_col2:
                            st.markdown("#### 3. The ₹1 Lakh Example")
                            st.info("If you invested **₹1,00,000** in this portfolio today:")
                            
                            investment = 100000
                            profit_pre_tax = investment * max_sharpe_port['Return']
                            tax_bill = profit_pre_tax * tax_impact
                            profit_post_tax = profit_pre_tax - tax_bill
                            worst_day_loss = investment * max_sharpe_port['VaR_95'] # VaR is negative
                            
                            st.write(f"""
                            * **Expected Profit (Pre-Tax):** ₹{profit_pre_tax:,.0f}
                            * **Tax Bill (@{tax_rate}%):** -₹{tax_bill:,.0f} (This is the drag!)
                            * **Net Profit (In Pocket):** **₹{profit_post_tax:,.0f}**
                            """)
                            
                            st.warning(f"""
                            **🚨 The Crash Scenario:**
                            If the market crashes tomorrow (a "VaR Event"), your portfolio value might drop by roughly **₹{abs(worst_day_loss):,.0f}** in a single day. 
                            """)

                        # --- 7. DEEP DIVE VISUALS ---
                        st.divider()
                        st.subheader("🔬 Deep Dive: Correlation & Drawdown")
                        
                        d_col1, d_col2 = st.columns(2)

                        with d_col1:
                            st.markdown("#### Correlation Matrix")
                            st.caption("How closely do these assets move together? (Light = High Correlation, Dark = Low)")
                            
                            # Calculate Correlation
                            corr_matrix = daily_returns.corr()
                            
                            # Plot Heatmap using Plotly
                            fig_corr = px.imshow(
                                corr_matrix, 
                                text_auto=True, 
                                aspect="auto",
                                color_continuous_scale='RdBu_r', # Red = High Corr (Bad for diversity)
                                origin='lower'
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                            
                            st.info("""
                            **Quant Tip:** You want a portfolio with **lower** correlation numbers (darker colors). 
                            If everything is '1.0', your portfolio is not diversified!
                            """)

                        with d_col2:
                            st.markdown("#### Max Drawdown (The 'Pain' Chart)")
                            st.caption("Percentage fall from the highest peak value over time.")
                            
                            # Calculate Cumulative Return of the Optimized Portfolio
                            cumulative_returns = (1 + daily_returns.dot(optimal_weights)).cumprod()
                            
                            # Calculate Drawdown
                            peak = cumulative_returns.cummax()
                            drawdown = (cumulative_returns - peak) / peak
                            
                            # Plot Drawdown
                            fig_dd = px.area(
                                drawdown, 
                                title="Portfolio Drawdown History",
                                labels={'value': 'Drawdown %', 'Date': 'Year'},
                                color_discrete_sequence=['red']
                            )
                            fig_dd.update_layout(yaxis_tickformat='.0%')
                            st.plotly_chart(fig_dd, use_container_width=True)

                            max_dd = drawdown.min()
                            st.error(f"**Max Historical Drawdown:** {max_dd:.1%} (The worst crash this portfolio faced)")

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("👈 Use the sidebar to select assets and click 'Run Optimization'")

with tab2:
    st.header("🧠 The Math Behind the Model")
    st.markdown("""
    This section explains the Quantitative Finance concepts used in this tool. 
    Reviewing this will help you answer technical interview questions.
    """)

    st.markdown("---")

    # CONCEPT 1: SHARPE RATIO
    st.subheader("1. Sharpe Ratio (The Efficiency Score)")
    st.markdown("""
    **Why use it?** Returns alone are meaningless without knowing the risk taken to get them. The Sharpe Ratio tells us 
    "how much return we are getting per unit of risk."
    """)
    st.latex(r'''
    Sharpe = \frac{R_p - R_f}{\sigma_p}
    ''')
    st.markdown("""
    * $R_p$: Expected Portfolio Return
    * $R_f$: Risk-Free Rate (we assumed 6.0%)
    * $\sigma_p$: Portfolio Standard Deviation (Volatility)
    """)

    st.markdown("---")

    # CONCEPT 2: VALUE AT RISK (VaR)
    st.subheader("2. Value at Risk (VaR 95%)")
    st.markdown("""
    **Why use it?** Volatility ($\sigma$) assumes risk is symmetrical (upside is the same as downside). 
    Banks care about **Downside Risk**. VaR answers: *"On a really bad day (worst 5%), how much could I lose?"*
    """)
    st.latex(r'''
    VaR_{\alpha} = \mu + z_{\alpha} \cdot \sigma
    ''')
    st.write("In our Monte Carlo simulation, we calculate this empirically by sorting the returns of 5,000 portfolios and finding the 5th percentile.")

    st.markdown("---")

    # CONCEPT 3: TAX DRAG
    st.subheader("3. Tax Drag (The Analyst's Edge)")
    st.markdown("""
    **Why use it?** Most algorithms optimize for *Pre-Tax* returns. In the real world, what matters is what you keep.
    We introduce a simple linear scalar to adjust the Efficient Frontier.
    """)
    st.latex(r'''
    R_{post-tax} = R_{pre-tax} \times (1 - TaxRate)
    ''')
