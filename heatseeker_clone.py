import streamlit as st
import pandas as pd
import numpy as np
from polygon import RESTClient
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime
import time

st.set_page_config(page_title="Heatseeker Clone", layout="wide")
st.title("🔥 Heatseeker Clone by Grok")
st.markdown("**Dealer Exposure Heatmap • Real-time GEX Nodes • Trinity Mode**  \n*Approximates Skylit AI's Heatseeker using public options data*")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Polygon API Key", type="password", value="DEMO")  # Replace with your key or leave as DEMO for testing
    ticker_options = ["SPY", "QQQ"]
    selected_tickers = st.multiselect("Tickers (Trinity Mode)", ticker_options, default=ticker_options)
    refresh_interval = st.slider("Auto-refresh (seconds)", 30, 300, 60)
    st.info("Market hours recommended for live data")

client = RESTClient(api_key) if api_key != "DEMO" else RESTClient()

# Black-Scholes Gamma (same for call/put)
def bs_gamma(S, K, T, r=0.05, sigma=None):
    if T <= 0 or sigma is None or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Fetch & compute GEX for one ticker
@st.cache_data(ttl=60)
def get_gex_data(ticker):
    try:
        options_iter = client.list_snapshot_options_chain(
            ticker,
            params={"limit": 500, "expiration_date.gte": datetime.now().strftime("%Y-%m-%d")}
        )
        options = list(options_iter)
    except:
        st.error(f"Could not fetch data for {ticker}. Check API key/market hours.")
        return pd.DataFrame()

    data = []
    spot = None
    for opt in options:
        if not spot:
            spot = opt.underlying_asset.price
        details = opt.details
        if not details:
            continue
        K = details.strike_price
        exp_date = datetime.strptime(details.expiration_date, "%Y-%m-%d")
        T = (exp_date - datetime.now()).days / 365.25
        iv = opt.implied_volatility or 0.2
        gamma = bs_gamma(spot, K, T, sigma=iv)
        oi = opt.open_interest or 0
        gex = gamma * oi * 100 * spot
        if details.contract_type == "put":
            gex = -gex  # Standard dealer GEX convention
        
        data.append({
            "strike": K,
            "expiration": details.expiration_date,
            "type": details.contract_type,
            "oi": oi,
            "gamma": gamma,
            "gex": gex,
            "iv": iv
        })
    
    df = pd.DataFrame(data)
    if not df.empty:
        # Aggregate per strike + expiration
        df_agg = df.groupby(["expiration", "strike"]).agg({
            "gex": "sum",
            "oi": "sum",
            "gamma": "mean",
            "iv": "mean"
        }).reset_index()
        df_agg["abs_gex"] = df_agg["gex"].abs()
        df_agg["color"] = np.where(df_agg["gex"] > 0, "#FFEA00", "#6B00B6")  # Pika yellow / Barney purple
        df_agg["size"] = df_agg["abs_gex"] / df_agg["abs_gex"].max() * 60 + 10  # bubble size
    return df_agg, spot

# Main app
if st.button("🔄 Refresh Data Now"):
    st.cache_data.clear()

col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Live Dealer Exposure Heatmap (Trinity Mode)")
    
    figs = []
    for ticker in selected_tickers:
        with st.spinner(f"Loading {ticker}..."):
            df, spot = get_gex_data(ticker)
            if df.empty:
                continue
            
            # Find King Node
            king = df.loc[df["abs_gex"].idxmax()]
            
            # Create figure
            fig = go.Figure()
            
            # Bubbles (nodes)
            fig.add_trace(go.Scatter(
                x=df["strike"],
                y=df["gex"],
                mode="markers",
                marker=dict(
                    size=df["size"],
                    color=df["color"],
                    line=dict(width=2, color="white"),
                    opacity=0.85
                ),
                text=df.apply(lambda row: f"Strike {row['strike']}<br>GEX: {row['gex']:.0f}<br>OI: {row['oi']}", axis=1),
                hoverinfo="text",
                name=f"{ticker} Nodes"
            ))
            
            # Current price line
            fig.add_vline(x=spot, line=dict(color="red", width=3, dash="dash"), annotation_text=f"Current: ${spot:.2f}", annotation_position="top left")
            
            # King Node highlight
            fig.add_trace(go.Scatter(
                x=[king["strike"]],
                y=[king["gex"]],
                mode="markers+text",
                marker=dict(size=20, color="lime", symbol="star"),
                text=[f"👑 KING NODE<br>{king['strike']}"],
                textposition="top center",
                name="King Node"
            ))
            
            fig.update_layout(
                title=f"{ticker} Dealer Exposure • {len(df)} Nodes",
                xaxis_title="Strike Price",
                yaxis_title="Gamma Exposure (GEX)",
                height=500,
                template="plotly_dark",
                showlegend=False
            )
            figs.append(fig)
    
    # Display in grid
    for i, fig in enumerate(figs):
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("How to Trade Like Heatseeker")
    st.markdown("""
    **1. Identify Magnets**  
    Largest bubbles = strongest support/resistance.
    
    **2. King Node** ⭐  
    Dealers want price here by EOD/EOW.
    
    **3. Define Range**  
    Fade edges (high probability).
    
    **4. Gatekeepers**  
    Watch rejections near big nodes.
    
    **5. Map Flow**  
    Refresh often — nodes reshuffle!
    """)
    st.caption("Prototype approximates Skylit’s proprietary data. For production, add vanna, real-time alerts, Discord bots, etc.")

st.success("✅ App ready! Customize the code to add Swing Mode, alerts, or backtesting.")
st.markdown("Built in minutes by Grok • Powered by Polygon.io + Plotly")
