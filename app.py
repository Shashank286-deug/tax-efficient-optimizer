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
    try:
        data = yf.download(tickers, start=start, end=end)['Close']
        return data
    except Exception:
        return pd.DataFrame()

# --- 4. SENTIMENT ENGINE (ROBUST GOOGLE NEWS) ---
def get_sentiment(tickers):
    sentiment_data = []
    
    for ticker in tickers:
        try:
            # Clean ticker for better search (e.g., "TCS.NS" -> "TCS India stock news")
            clean_name = ticker.replace(".NS", "")
            search_term = f"{clean_name} India stock news"
            
            googlenews = GoogleNews(lang='en', region='IN', period='1d')
            googlenews.clear()
            googlenews.search(search_term)
            result = googlenews.result()
            
            if result and len(result) > 0:
                # Analyze top 2 headlines
                top_headlines = " ".join([item['title'] for item in result[:2]])
                latest_headline = result[0]['title']
                analysis = TextBlob(top_headlines)
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
             # Fallback if Google News fails (prevents app crash)
             sentiment_data.append({
                 "Asset": ticker, 
                 "Sentiment Score": 0, 
                 "Latest News": "News unavailable (Connection Limit)"
             })
             
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
                        st.error("No data returned. Try different dates or tickers.")
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
                            
                            # Safe division for Sharpe
                            if port_std_dev == 0:
                                sharpe_ratio = 0
                            else:
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
                        sim_df =
