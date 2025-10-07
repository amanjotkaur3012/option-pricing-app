import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math
import pandas as pd

# -------------------------------
# Helper Functions
# -------------------------------

def get_live_price(symbol):
    """Fetch the latest live price from Yahoo Finance"""
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if data.empty:
            data = yf.Ticker(symbol).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception:
        st.warning("⚠️ Could not fetch live data, using default ₹100")
        return 100.0

# --- Black-Scholes Model ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price, d1, d2

# --- Greeks ---
def greeks(S, K, T, r, sigma, option_type='call'):
    price, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2)
    return price, delta, gamma, theta, vega, rho

# --- Binomial Model ---
def binomial_option_price(S, K, T, r, sigma, steps=100, option_type='call'):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    prices = [S * (u**j) * (d**(steps-j)) for j in range(steps+1)]
    payoffs = [max(pS - K, 0) if option_type=='call' else max(K - pS, 0) for pS in prices]
    for i in range(steps-1, -1, -1):
        payoffs = [discount * (p*payoffs[j+1] + (1-p)*payoffs[j]) for j in range(i+1)]
    return payoffs[0]

# -------------------------------
# Streamlit App Layout
# -------------------------------

st.set_page_config(page_title="Option Pricing Model", page_icon="📈", layout="wide")
st.title("📊 Option Pricing and Greeks Calculator")
st.caption("Developed by Amanjot Kaur | MSc Finance & Analytics | Christ University")

st.markdown("""
This interactive tool prices **Index or Equity Options** using:
- **Black-Scholes Model**
- **Binomial Tree Model**
It also visualizes how volatility, time, and spot price affect option values and Greeks.
""")

# -------------------------------
# Inputs
# -------------------------------
symbol = st.text_input("Enter a Stock or Index Symbol (e.g., RELIANCE.NS, INFY.NS, ^NSEI):", value="RELIANCE.NS")

if st.button("📥 Fetch Live Price"):
    S = get_live_price(symbol)
    st.success(f"✅ Live Price for {symbol}: ₹{S}")
    st.session_state['S'] = S

if 'S' in st.session_state:
    S = st.session_state['S']

    st.subheader("🔧 Option Input Parameters")
    K = st.number_input("Strike Price (₹)", value=S, step=10.0)
    days = st.number_input("Days to Expiry", value=30, step=5)
    r = st.number_input("Risk-free Interest Rate (%)", value=6.0, step=0.1) / 100
    sigma = st.number_input("Volatility (%)", value=25.0, step=1.0) / 100
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    model = st.radio("Select Model", ["Black-Scholes", "Binomial"], horizontal=True)
    steps = st.slider("Binomial Steps (if using Binomial Model)", 10, 200, 100)
    T = days / 365

    if st.button("🧮 Calculate Option Price"):
        if model == "Black-Scholes":
            price, delta, gamma, theta, vega, rho = greeks(S, K, T, r, sigma, option_type)
        else:
            price = binomial_option_price(S, K, T, r, sigma, steps, option_type)
            delta = gamma = theta = vega = rho = np.nan

        # --- Display Results ---
        st.subheader("📈 Option Valuation Results")
        st.metric("Option Price (₹)", f"{price:.2f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Delta", f"{delta:.4f}")
        c2.metric("Gamma", f"{gamma:.6f}")
        c3.metric("Theta", f"{theta:.4f}")

        c4, c5 = st.columns(2)
        c4.metric("Vega", f"{vega:.4f}")
        c5.metric("Rho", f"{rho:.4f}")

        # -------------------------------
        # Scenario Analysis
        # -------------------------------
        st.subheader("📊 Scenario Analysis Charts")

        # --- Option Price vs Volatility ---
        vols = np.linspace(0.05, 0.6, 20)
        prices_vol = [black_scholes(S, K, T, r, v, option_type)[0] for v in vols]
        fig1, ax1 = plt.subplots()
        ax1.plot(vols*100, prices_vol, color='purple', marker='o')
        ax1.set_title("Option Price vs Volatility")
        ax1.set_xlabel("Volatility (%)")
        ax1.set_ylabel("Option Price (₹)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

        # --- Option Price vs Spot Price ---
        spots = np.linspace(S*0.8, S*1.2, 20)
        prices_spot = [black_scholes(s, K, T, r, sigma, option_type)[0] for s in spots]
        fig2, ax2 = plt.subplots()
        ax2.plot(spots, prices_spot, color='teal', marker='o')
        ax2.set_title("Option Price vs Spot Price")
        ax2.set_xlabel("Spot Price (₹)")
        ax2.set_ylabel("Option Price (₹)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        # --- Option Price vs Time to Expiry ---
        times = np.linspace(5/365, 90/365, 20)
        prices_time = [black_scholes(S, K, t, r, sigma, option_type)[0] for t in times]
        fig3, ax3 = plt.subplots()
        ax3.plot(times*365, prices_time, color='darkorange', marker='o')
        ax3.set_title("Option Price vs Days to Expiry")
        ax3.set_xlabel("Days to Expiry")
        ax3.set_ylabel("Option Price (₹)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig3)

        # --- Summary Table ---
        df = pd.DataFrame({
            "Volatility (%)": vols*100,
            "Option Price (₹)": prices_vol
        })
        st.dataframe(df, use_container_width=True)

        st.info("""
        **Key Insights:**
        - 📈 Higher volatility → higher option price (Vega effect)
        - ⏳ Longer time to expiry → higher option value (Theta effect)
        - 📊 For calls: higher spot price → higher option value  
        - 📉 For puts: higher spot price → lower option value
        """)
