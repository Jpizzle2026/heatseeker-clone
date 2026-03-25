import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Heatseeker Clone - Heatmap Style", layout="wide")
st.title("🔥 Heatseeker Clone (Heatmap Style - Free)")
st.markdown("**Trinity Mode Heatmap • Dealer GEX Bars • Approximates Skylit Heatseeker**")

with st.sidebar:
    st.header("Settings")
    ticker_options = ["SPXW", "SPY", "QQQ"]  # SPXW works via ^SPX in yfinance for index
    selected_tickers = st.multiselect("Panels", ticker_options, default=ticker_options)
    max_expirations = st.slider("Max expirations to include", 1, 10, 5)

def bs_gamma(S, K, T, sigma=0.2):
    if T <= 0 or sigma <= 0:
        return 0.0
    from scipy.stats import norm
    d1 = (np.log(S / K) + (0.05 + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

@st.cache_data(ttl=60)
def get_gex_heatmap_data(ticker_symbol):
    try:
        if ticker_symbol == "SPXW":
            tk = yf.Ticker("^SPX")  # Index for SPX options
        else:
            tk = yf.Ticker(ticker_symbol)
        
        spot = tk.history(period="1d")['Close'].iloc[-1]
        data = []
        
        for exp in list(tk.options)[:max_expirations]:
            chain = tk.option_chain(exp)
            for df_type, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                for _, row in df_type.iterrows():
                    K = row['strike']
                    T = (datetime.strptime(exp, "%Y-%m-%d") - datetime.now()).days / 365.25
                    iv = row.get('impliedVolatility', 0.2) or 0.2
                    gamma = bs_gamma(spot, K, T, iv)
                    oi = row.get('openInterest', 0) or 0
                    gex = gamma * oi * 100 * spot
                    if opt_type == "put":
                        gex = -gex
                    data.append({"strike": K, "gex": gex, "exp": exp, "oi": oi})
        
        df = pd.DataFrame(data)
        if df.empty:
            return None, spot
        
        # Aggregate by strike
        df_agg = df.groupby("strike").agg({"gex": "sum", "oi": "sum"}).reset_index()
        df_agg = df_agg.sort_values("strike")
        
        return df_agg, spot
    except Exception as e:
        st.error(f"Error for {ticker_symbol}: {e}")
        return None, None

if st.button("🔄 Refresh Now"):
    st.cache_data.clear()

cols = st.columns(len(selected_tickers))

for idx, ticker in enumerate(selected_tickers):
    with cols[idx]:
        with st.spinner(f"Loading {ticker} heatmap..."):
            df, spot = get_gex_heatmap_data(ticker)
            if df is None or spot is None:
                st.warning(f"No data for {ticker}")
                continue
            
            # King Node (strongest absolute GEX)
            king_row = df.loc[df["gex"].abs().idxmax()]
            
            # Create heatmap figure (horizontal bars via heatmap)
            fig = go.Figure()
            
            # Heatmap trace - strikes on y, single column on x for bars
            fig.add_trace(go.Heatmap(
                z=[[g] for g in df["gex"]],  # vertical bars
                y=df["strike"],
                x=["GEX"],
                colorscale=[[0, "#6B00B6"], [0.5, "#00BFA5"], [1, "#FFEA00"]],  # purple -> teal -> yellow
                showscale=True,
                colorbar=dict(title="Net GEX"),
                hovertemplate="Strike: %{y}<br>GEX: %{z:.0f}<extra></extra>"
            ))
            
            # Text labels for exposure values
            for i, row in df.iterrows():
                fig.add_annotation(
                    x=0.5,
                    y=row["strike"],
                    text=f"{row['gex']/1000:.0f}K",
                    showarrow=False,
                    font=dict(size=10, color="white" if abs(row["gex"]) > 1e6 else "black")
                )
            
            # Current price line
            fig.add_hline(y=spot, line=dict(color="red", width=3, dash="dash"),
                          annotation_text=f"Current ${spot:.2f}", annotation_position="left")
            
            # King Node highlight
            fig.add_hrect(y0=king_row["strike"]-2, y1=king_row["strike"]+2,
                          fillcolor="lime", opacity=0.3, line_width=0,
                          annotation_text="👑 KING", annotation_position="top right")
            
            fig.update_layout(
                title=f"{ticker} Dealer GEX Heatmap",
                height=700,
                template="plotly_dark",
                xaxis_visible=False,
                yaxis_title="Strike Price",
                margin=dict(l=60, r=60)
            )
            
            st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("""
**How to read it (like Skylit):**
- **Yellow/Lime** = Positive GEX (support, low vol)
- **Purple** = Negative GEX (resistance, high vol)
- **Longer/brighter bars** = Stronger magnet
- **King Node** = Dealers want price here
""")

st.success("✅ Closer to Skylit Heatseeker look! Still free with yfinance (~15 min delay).")
