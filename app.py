import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from GoogleNews import GoogleNews # New reliable news source

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

# --- 4. SENTIMENT ENGINE (UPDATED TO GOOGLE NEWS) ---
def get_sentiment(tickers):
    sentiment_data = []
    googlenews = GoogleNews(lang='en', region='IN', period='1d') # Look for news from the last 24h
    
    for ticker in tickers:
        try:
            # Clean the ticker: "RELIANCE.NS" -> "RELIANCE stock" for better search results
            clean_name = ticker.replace(".NS", "")
            search_term = f"{clean_name} stock news"
            
            # Fetch News
            googlenews.clear() # Clear past results
            googlenews.search(search_term)
            result = googlenews.result()
            
            if result:
                # Combine top 3 headlines for analysis
                top_3_headlines = " ".join([item['title'] for item in result[:3]])
                latest_headline = result[0]['title']
                
                # Analyze Sentiment
                analysis = TextBlob(top_3_headlines)
                score = analysis.sentiment.polarity
            else:
                score = 0
                latest_headline = "No recent news found."
                
            sentiment_data.append({
                "Asset": ticker,
                "Sentiment Score": score,
                "Latest News": latest_headline
            })
            
        except Exception as e:
             sentiment_data.append({"Asset": ticker, "Sentiment Score": 0, "Latest News": "Could not fetch data"})
             
    return pd.DataFrame(sentiment_data)

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚀 Run Optimizer", "📰 AI News Sentiment", "📘 Quant Concepts"])

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

                            port_return = np.sum(mean_daily_returns * weights) * 252
                            port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                            sharpe_ratio = (port_return - 0.06) / port_std_dev

                            sim_returns = daily_returns.dot(weights)
                            var_95 = np.percentile(sim_returns, 5)
                            cvar_95 = sim_returns[sim_returns <= var_95].mean()

                            post_tax_return = port_return * (1 - tax_impact)

                            results[0,i] = port_return
                            results[1,i] = port_std_dev
                            results[2,i] = sharpe_ratio
                            results[3,i] = post_tax_return
                            results[4,i] = var_95
                            results[5,i] = cvar_95

                        columns = ['Return', 'Risk', 'Sharpe', 'Post_Tax_Return', 'VaR_95', 'CVaR_95']
                        sim_df = pd.DataFrame(results.T, columns=columns)
                        
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
                            alloc_df = alloc_df[alloc_df['Weight'] > 0.01] 
                            alloc_df = alloc_df.sort_values(by="Weight", ascending=False)
                            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True)

                        # --- SCENARIO ANALYSIS ---
                        st.divider()
                        st.subheader("📝 Analysis & Real-World Example")
                        
                        ex_col1, ex_col2 = st.columns(2)
                        
                        with ex_col1:
                            st.markdown("#### 1. What is happening?")
                            top_stock = alloc_df.iloc[0]
                            st.write(f"""
                            The algorithm ran **5,000 simulations** and suggests allocating **{top_stock['Weight']:.1%}** to **{top_stock['Asset']}**.
                            This balance maximizes return for your accepted level of risk.
                            """)
                        
                        with ex_col2:
                            st.markdown("#### 2. The ₹1 Lakh Example")
                            investment = 100000
                            profit_pre_tax = investment * max_sharpe_port['Return']
                            tax_bill = profit_pre_tax * tax_impact
                            profit_post_tax = profit_pre_tax - tax_bill
                            
                            st.write(f"""
                            * **Expected Profit (Pre-Tax):** ₹{profit_pre_tax:,.0f}
                            * **Tax Bill (@{tax_rate}%):** -₹{tax_bill:,.0f}
                            * **Net Profit:** **₹{profit_post_tax:,.0f}**
                            """)

                        # --- DEEP DIVE ---
                        st.divider()
                        st.subheader("🔬 Deep Dive: Correlation & Drawdown")
                        d_col1, d_col2 = st.columns(2)

                        with d_col1:
                            st.markdown("#### Correlation Matrix")
                            corr_matrix = daily_returns.corr()
                            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
                            st.plotly_chart(fig_corr, use_container_width=True)

                        with d_col2:
                            st.markdown("#### Max Drawdown")
                            cumulative_returns = (1 + daily_returns.dot(optimal_weights)).cumprod()
                            drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
                            fig_dd = px.area(drawdown, title="Drawdown History", color_discrete_sequence=['red'])
                            fig_dd.update_layout(yaxis_tickformat='.0%')
                            st.plotly_chart(fig_dd, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("👈 Use the sidebar to select assets and click 'Run Optimization'")

with tab2:
    st.header("📰 AI News Analysis (Live)")
    st.markdown("This tool fetches the latest news headlines using **Google News** and uses **Natural Language Processing (NLP)** to score the sentiment.")
    
    if st.button("Analyze News Sentiment"):
        with st.spinner("Searching Google News & analyzing sentiment..."):
            sent_df = get_sentiment(selected_tickers)
            
            # Display Key Metrics
            avg_sentiment = sent_df['Sentiment Score'].mean()
            sentiment_label = "Positive 🟢" if avg_sentiment > 0.05 else "Negative 🔴" if avg_sentiment < -0.05 else "Neutral ⚪"
            
            st.metric("Overall Portfolio Mood", sentiment_label, delta=f"{avg_sentiment:.3f} Score")
            
            # Display detailed Dataframe with styling
            st.dataframe(
                sent_df.style.background_gradient(subset=['Sentiment Score'], cmap='RdYlGn', vmin=-0.5, vmax=0.5),
                use_container_width=True
            )
            
            st.info("""
            **How it works:**
            * **Score > 0:** The news language is generally positive/optimistic.
            * **Score < 0:** The news language is generally negative/pessimistic.
            * We search Google News for headlines (e.g., 'TCS stock news') and analyze the text.
            """)

with tab3:
    st.header("🧠 The Math Behind the Model")
    st.markdown("### 1. Sharpe Ratio")
    st.latex(r'''Sharpe = \frac{R_p - R_f}{\sigma_p}''')
    st.markdown("### 2. Value at Risk (VaR)")
    st.latex(r'''VaR_{\alpha} = \mu + z_{\alpha} \cdot \sigma''')
    st.markdown("### 3. Tax Drag")
    st.latex(r'''R_{post-tax} = R_{pre-tax} \times (1 - TaxRate)''')import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from GoogleNews import GoogleNews # New reliable news source

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

# --- 4. SENTIMENT ENGINE (UPDATED TO GOOGLE NEWS) ---
def get_sentiment(tickers):
    sentiment_data = []
    googlenews = GoogleNews(lang='en', region='IN', period='1d') # Look for news from the last 24h
    
    for ticker in tickers:
        try:
            # Clean the ticker: "RELIANCE.NS" -> "RELIANCE stock" for better search results
            clean_name = ticker.replace(".NS", "")
            search_term = f"{clean_name} stock news"
            
            # Fetch News
            googlenews.clear() # Clear past results
            googlenews.search(search_term)
            result = googlenews.result()
            
            if result:
                # Combine top 3 headlines for analysis
                top_3_headlines = " ".join([item['title'] for item in result[:3]])
                latest_headline = result[0]['title']
                
                # Analyze Sentiment
                analysis = TextBlob(top_3_headlines)
                score = analysis.sentiment.polarity
            else:
                score = 0
                latest_headline = "No recent news found."
                
            sentiment_data.append({
                "Asset": ticker,
                "Sentiment Score": score,
                "Latest News": latest_headline
            })
            
        except Exception as e:
             sentiment_data.append({"Asset": ticker, "Sentiment Score": 0, "Latest News": "Could not fetch data"})
             
    return pd.DataFrame(sentiment_data)

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚀 Run Optimizer", "📰 AI News Sentiment", "📘 Quant Concepts"])

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

                            port_return = np.sum(mean_daily_returns * weights) * 252
                            port_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
                            sharpe_ratio = (port_return - 0.06) / port_std_dev

                            sim_returns = daily_returns.dot(weights)
                            var_95 = np.percentile(sim_returns, 5)
                            cvar_95 = sim_returns[sim_returns <= var_95].mean()

                            post_tax_return = port_return * (1 - tax_impact)

                            results[0,i] = port_return
                            results[1,i] = port_std_dev
                            results[2,i] = sharpe_ratio
                            results[3,i] = post_tax_return
                            results[4,i] = var_95
                            results[5,i] = cvar_95

                        columns = ['Return', 'Risk', 'Sharpe', 'Post_Tax_Return', 'VaR_95', 'CVaR_95']
                        sim_df = pd.DataFrame(results.T, columns=columns)
                        
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
                            alloc_df = alloc_df[alloc_df['Weight'] > 0.01] 
                            alloc_df = alloc_df.sort_values(by="Weight", ascending=False)
                            fig_pie = px.pie(alloc_df, values='Weight', names='Asset', hole=0.4)
                            st.plotly_chart(fig_pie, use_container_width=True)

                        # --- SCENARIO ANALYSIS ---
                        st.divider()
                        st.subheader("📝 Analysis & Real-World Example")
                        
                        ex_col1, ex_col2 = st.columns(2)
                        
                        with ex_col1:
                            st.markdown("#### 1. What is happening?")
                            top_stock = alloc_df.iloc[0]
                            st.write(f"""
                            The algorithm ran **5,000 simulations** and suggests allocating **{top_stock['Weight']:.1%}** to **{top_stock['Asset']}**.
                            This balance maximizes return for your accepted level of risk.
                            """)
                        
                        with ex_col2:
                            st.markdown("#### 2. The ₹1 Lakh Example")
                            investment = 100000
                            profit_pre_tax = investment * max_sharpe_port['Return']
                            tax_bill = profit_pre_tax * tax_impact
                            profit_post_tax = profit_pre_tax - tax_bill
                            
                            st.write(f"""
                            * **Expected Profit (Pre-Tax):** ₹{profit_pre_tax:,.0f}
                            * **Tax Bill (@{tax_rate}%):** -₹{tax_bill:,.0f}
                            * **Net Profit:** **₹{profit_post_tax:,.0f}**
                            """)

                        # --- DEEP DIVE ---
                        st.divider()
                        st.subheader("🔬 Deep Dive: Correlation & Drawdown")
                        d_col1, d_col2 = st.columns(2)

                        with d_col1:
                            st.markdown("#### Correlation Matrix")
                            corr_matrix = daily_returns.corr()
                            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', origin='lower')
                            st.plotly_chart(fig_corr, use_container_width=True)

                        with d_col2:
                            st.markdown("#### Max Drawdown")
                            cumulative_returns = (1 + daily_returns.dot(optimal_weights)).cumprod()
                            drawdown = (cumulative_returns - cumulative_returns.cummax()) / cumulative_returns.cummax()
                            fig_dd = px.area(drawdown, title="Drawdown History", color_discrete_sequence=['red'])
                            fig_dd.update_layout(yaxis_tickformat='.0%')
                            st.plotly_chart(fig_dd, use_container_width=True)

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("👈 Use the sidebar to select assets and click 'Run Optimization'")

with tab2:
    st.header("📰 AI News Analysis (Live)")
    st.markdown("This tool fetches the latest news headlines using **Google News** and uses **Natural Language Processing (NLP)** to score the sentiment.")
    
    if st.button("Analyze News Sentiment"):
        with st.spinner("Searching Google News & analyzing sentiment..."):
            sent_df = get_sentiment(selected_tickers)
            
            # Display Key Metrics
            avg_sentiment = sent_df['Sentiment Score'].mean()
            sentiment_label = "Positive 🟢" if avg_sentiment > 0.05 else "Negative 🔴" if avg_sentiment < -0.05 else "Neutral ⚪"
            
            st.metric("Overall Portfolio Mood", sentiment_label, delta=f"{avg_sentiment:.3f} Score")
            
            # Display detailed Dataframe with styling
            st.dataframe(
                sent_df.style.background_gradient(subset=['Sentiment Score'], cmap='RdYlGn', vmin=-0.5, vmax=0.5),
                use_container_width=True
            )
            
            st.info("""
            **How it works:**
            * **Score > 0:** The news language is generally positive/optimistic.
            * **Score < 0:** The news language is generally negative/pessimistic.
            * We search Google News for headlines (e.g., 'TCS stock news') and analyze the text.
            """)

with tab3:
    st.header("🧠 The Math Behind the Model")
    st.markdown("### 1. Sharpe Ratio")
    st.latex(r'''Sharpe = \frac{R_p - R_f}{\sigma_p}''')
    st.markdown("### 2. Value at Risk (VaR)")
    st.latex(r'''VaR_{\alpha} = \mu + z_{\alpha} \cdot \sigma''')
    st.markdown("### 3. Tax Drag")
    st.latex(r'''R_{post-tax} = R_{pre-tax} \times (1 - TaxRate)''')
