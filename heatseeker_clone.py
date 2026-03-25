import streamlit as st
import pandas as pd
import numpy as np
from massive import RESTClient
from scipy.stats import norm
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Heatseeker Clone", layout="wide")
st.title("🔥 Heatseeker Clone by Grok")
st.markdown("**Dealer Exposure Heatmap • Real-time GEX Nodes • Trinity Mode**  \n*Approximates Skylit AI's Heatseeker using Massive.com (ex-Polygon) data*")

# Sidebar
with st.sidebar:
    st.header("Settings")
    # Use secrets if available, otherwise fallback to text input
    if "POLYGON_API_KEY" in st.secrets:
        api_key = st.secrets["POLYGON_API_KEY"]
        st.success("✅ API key loaded from secrets")
    else:
        api_key = st.text_input("Massive / Polygon API Key", type="password", value="DEMO")
    
    ticker_options = ["SPY", "QQQ"]
    selected_tickers = st.multiselect("Tickers (Trinity Mode)", ticker_options, default=ticker_options)
    refresh_interval = st.slider("Auto-refresh (seconds)", 30, 300, 60)
    st.info("Market hours recommended for best data")

# Initialize client safely
@st.cache_resource
def get_client(api_key):
    try:
        if api_key and api_key != "DEMO":
            return RESTClient(api_key=api_key)
        else:
            return RESTClient()  # demo mode if supported
    except Exception as e:
        st.error(f"Failed to initialize client: {e}")
        return None

client = get_client(api_key)

# Black-Scholes Gamma
def bs_gamma(S, K, T, r=0.05, sigma=None):
    if T <= 0 or sigma is None or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

# Fetch GEX data
@st.cache_data(ttl=60)
def get_gex_data(ticker):
    if not client:
        st.error("Client not initialized. Check API key.")
        return pd.DataFrame(), None
    
    try:
        options_iter = client.list_snapshot_options_chain(
            ticker,
            params={
                "limit": 500,
                "expiration_date.gte": datetime.now().strftime("%Y-%m-%d")
            }
        )
        options = list(options_iter)
    except Exception as e:
        st.error(f"Could not fetch data for {ticker}: {str(e)[:200]}")
        return pd.DataFrame(), None

    data = []
    spot = None
    for opt in options:
        try:
            if not spot and hasattr(opt, 'underlying_asset') and opt.underlying_asset:
                spot = opt.underlying_asset.price
            
            details = opt.details if hasattr(opt, 'details') else None
            if not details:
                continue
                
            K = details.strike_price
            exp_date = datetime.strptime(details.expiration_date, "%Y-%m-%d")
            T = (exp_date - datetime.now()).days / 365.25
            iv = getattr(opt, 'implied_volatility', 0.2) or 0.2
            gamma = bs_gamma(spot, K, T, sigma=iv)
            oi = getattr(opt, 'open_interest', 0) or 0
            gex = gamma * oi * 100 * spot
            if details.contract_type == "put":
                gex = -gex
            
            data.append({
                "strike": K,
                "expiration": details.expiration_date,
                "type": details.contract_type,
                "oi": oi,
                "gamma": gamma,
                "gex": gex,
                "iv": iv
            })
        except:
            continue  # skip bad entries
    
    df = pd.DataFrame(data)
    if df.empty:
        return df, spot
    
    # Aggregate
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

# Sidebar help
with st.sidebar:
    st.subheader("How to Trade")
    st.markdown("""
    **Magnets** — Biggest bubbles = strongest support/resistance  
    **King Node** 👑 — Dealers heavily exposed here  
    **Range** — Fade the edges  
    **Refresh often** — Nodes move!
    """)
    st.caption("Data via Massive.com • Prototype by Grok")

st.success("✅ Heatseeker Clone is running! Add your API key in Streamlit Secrets for best results.")
