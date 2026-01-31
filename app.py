import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
from GoogleNews import GoogleNews

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

# --- 4. SENTIMENT ENGINE (GOOGLE NEWS) ---
def get_sentiment(tickers):
    sentiment_data = []
    googlenews = GoogleNews(lang='en', region='IN', period='1d') 
    
    for ticker in tickers:
        try:
            clean_name = ticker.replace(".NS", "")
            search_term = f"{clean_name} stock news"
            
            googlenews.clear()
            googlenews.search(search_term)
            result = googlenews.result()
            
            if result:
                top_3 = " ".join([item['title'] for item in result[:3]])
                latest_headline = result[0]['title']
                analysis = TextBlob(top_3)
                score = analysis.sentiment.polarity
            else:
                score = 0
                latest_headline = "No recent news found."
                
            sentiment_data.append({
                "Asset": ticker,
                "Sentiment Score": score,
                "Latest News": latest_headline
            })
            
        except Exception:
             sentiment_data.append({"Asset": ticker, "Sentiment Score": 0, "Latest News": "Could not fetch data"})
             
    return pd.DataFrame(sentiment_data)

# --- TABS LAYOUT ---
tab1, tab2, tab3 = st.tabs(["🚀 Run Optimizer", "📰 AI News Sentiment", "📘 Quant Concepts"])

with tab1:
    if st.sidebar.button("Run Optimization"):
        if len(selected_tickers) < 2:
            st.error("⚠️ Please select at least 2 stocks.")
        else:
            with st.spinner('Calculating Efficient Frontier...'):
                try:
                    df = get_stock_data(selected_tickers, start_date, end_date)
                    
                    if df.empty:
                        st.error("No data returned. Try different dates.")
                    else:
                        daily_returns = df.pct_change().dropna()
                        
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

                        st.success("Optimization Complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("💰 Annual Return", f"{max_sharpe_port['Return']:.1%}", delta=f"Post-Tax: {max_sharpe_port['Post_Tax_Return']:.1%}")
                        col2.metric("📉 Volatility", f"{max_sharpe_port['Risk']:.1%}")
                        col3.metric("⚠️ 95% VaR", f"{max_sharpe_port['VaR_95']:.2%}")
                        col4.metric("☢️ 95% CVaR", f"{max_sharpe_port['CVaR_95']:.2%}")

                        c1, c2 = st.columns([2,1])
                        with c1:
                            st.subheader("Efficient Frontier")
                            fig = px.scatter(sim_df, x='Risk', y='Return', color='Sharpe', title="Risk vs Return Landscape")
                            fig.add_scatter(x=[max_sharpe_port['Risk']], y=[max_sharpe_port['Return']], mode='markers', marker=dict(color='red', size=20, symbol='star'), name='Optimal Portfolio')
                            st.plotly_chart(fig, use_container_width=True)
