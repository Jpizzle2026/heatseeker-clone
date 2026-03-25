import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Heatseeker Clone - Free", layout="wide")
st.title("🔥 Heatseeker Clone (Free yfinance Version)")
st.markdown("**Dealer Exposure Heatmap • GEX Nodes • Trinity Mode**  \n*Fully free using Yahoo Finance data • Approximates Skylit AI's Heatseeker*")

# Sidebar
with st.sidebar:
    st.header("Settings")
    ticker_options = ["SPY", "QQQ"]
    selected_tickers = st.multiselect("Tickers (Trinity Mode)", ticker_options, default=ticker_options)
    st.info("Data is delayed ~15 min. Refresh during market hours for best results.")

# Black-Scholes Gamma
def bs_gamma(S, K, T, r=0.05, sigma=None):
    if T <= 0 or sigma is None or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Fetch GEX data using yfinance
@st.cache_data(ttl=60)
def get_gex_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        spot = tk.history(period="1d")['Close'].iloc[-1]
        
        # Get all expirations and fetch chains
        data = []
        for exp in tk.options[:5]:  # Limit to first 5 expirations to keep it fast
            chain = tk.option_chain(exp)
            calls = chain.calls
            puts = chain.puts
            
            for df, opt_type in [(calls, "call"), (puts, "put")]:
                for _, row in df.iterrows():
                    K = row['strike']
                    T = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365.25
                    iv = row['impliedVolatility'] or 0.2
                    gamma = bs_gamma(spot, K, T, sigma=iv)
                    oi = row['openInterest'] or 0
                    gex = gamma * oi * 100 * spot
                    if opt_type == "put":
                        gex = -gex
                    
                    data.append({
                        "strike": K,
                        "expiration": exp,
                        "type": opt_type,
                        "oi": oi,
                        "gamma": gamma,
                        "gex": gex,
                        "iv": iv
                    })
        
        df = pd.DataFrame(data)
        if df.empty:
            return pd.DataFrame(), spot
        
        # Aggregate per strike + expiration
        df_agg = df.groupby(["expiration", "strike"]).agg({
            "gex": "sum",
            "oi": "sum",
            "gamma": "mean",
            "iv": "mean"
        }).reset_index()
        
        df_agg["abs_gex"] = df_agg["gex"].abs()
        df_agg["color"] = np.where(df_agg["gex"] > 0, "#FFEA00", "#6B00B6")
        df_agg["size"] = (df_agg["abs_gex"] / df_agg["abs_gex"].max() * 60 + 10).clip(10, 80)
        
        return df_agg, spot
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame(), None

# Main UI
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

st.subheader("Live Dealer Exposure Heatmap (Trinity Mode)")

cols = st.columns(len(selected_tickers))

for idx, ticker in enumerate(selected_tickers):
    with cols[idx]:
        with st.spinner(f"Loading {ticker}..."):
            df, spot = get_gex_data(ticker)
            if df.empty or spot is None:
                st.warning(f"No data for {ticker}")
                continue
            
            # King Node
            king_idx = df["abs_gex"].idxmax()
            king = df.loc[king_idx]
            
            fig = go.Figure()
            
            # Nodes
            fig.add_trace(go.Scatter(
                x=df["strike"],
                y=df["gex"],
                mode="markers",
                marker=dict(size=df["size"], color=df["color"], line=dict(width=2, color="white"), opacity=0.85),
                text=df.apply(lambda row: f"Strike {row['strike']}<br>GEX: {row['gex']:.0f}<br>OI: {row['oi']}", axis=1),
                hoverinfo="text",
                name=f"{ticker} Nodes"
            ))
            
            # Current price
            fig.add_vline(x=spot, line=dict(color="red", width=3, dash="dash"),
                         annotation_text=f"Current: ${spot:.2f}", annotation_position="top left")
            
            # King Node
            fig.add_trace(go.Scatter(
                x=[king["strike"]],
                y=[king["gex"]],
                mode="markers+text",
                marker=dict(size=25, color="lime", symbol="star"),
                text=[f"👑 KING NODE\n{king['strike']}"],
                textposition="top center",
                name="King Node"
            ))
            
            fig.update_layout(
                title=f"{ticker} Dealer Exposure • {len(df)} Nodes",
                xaxis_title="Strike Price",
                yaxis_title="Gamma Exposure (GEX)",
                height=520,
                template="plotly_dark",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

with st.sidebar:
    st.subheader("How to Trade")
    st.markdown("""
    **Magnets** — Biggest bubbles = strongest support/resistance  
    **King Node** 👑 — Dealers heavily exposed here  
    **Refresh often** — Nodes shift with new OI/IV
    """)
    st.caption("100% Free • Powered by yfinance (Yahoo Finance) • Built by Grok")

st.success("✅ Running on free data! No API key needed.")
