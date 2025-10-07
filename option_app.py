import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
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

# --- Binomial Model ---
def binomial_option_price(S, K, T, r, sigma, steps=100, option_type='call'):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # terminal prices
    prices = [S * (u**j) * (d**(steps-j)) for j in range(steps+1)]
    payoffs = [max(pS - K, 0) if option_type == 'call' else max(K - pS, 0) for pS in prices]

    # backward induction
    for i in range(steps-1, -1, -1):
        payoffs = [discount * (p * payoffs[j+1] + (1 - p) * payoffs[j]) for j in range(i+1)]

    return payoffs[0]

# --- Greeks (Numerical Approximations for Binomial Model) ---
def calculate_greeks_binomial(S, K, T, r, sigma, steps=100, option_type='call', h=0.01):
    price = binomial_option_price(S, K, T, r, sigma, steps, option_type)
    
    # Delta
    delta = (binomial_option_price(S+h, K, T, r, sigma, steps, option_type) -
             binomial_option_price(S-h, K, T, r, sigma, steps, option_type)) / (2*h)
    
    # Gamma
    price_up = binomial_option_price(S+h, K, T, r, sigma, steps, option_type)
    price_down = binomial_option_price(S-h, K, T, r, sigma, steps, option_type)
    gamma = (price_up - 2*price + price_down) / (h**2)
    
    # Theta
    dt = 1/365  # 1 day in years
    if T > dt:
        theta = (binomial_option_price(S, K, T, r, sigma, steps, option_type) -
                 binomial_option_price(S, K, T-dt, r, sigma, steps, option_type)) / dt
    else:
        theta = np.nan
    
    # Vega
    vega = (binomial_option_price(S, K, T, r, sigma+h, steps, option_type) -
            binomial_option_price(S, K, T, r, sigma-h, steps, option_type)) / (2*h)
    
    # Rho
    rho = (binomial_option_price(S, K, T, r+h, sigma, steps, option_type) -
           binomial_option_price(S, K, T, r-h, sigma, steps, option_type)) / (2*h)
    
    return price, delta, gamma, theta, vega, rho

# -------------------------------
# Streamlit App Layout
# -------------------------------

st.set_page_config(page_title="Option Pricing (Binomial Model)", page_icon="📈", layout="wide")
st.title("📊 Option Pricing using Binomial Model")
st.caption("Developed by Amanjot Kaur | MSc Finance & Analytics | Christ University")

st.markdown("""
This interactive tool prices **Equity or Index Options** using the **Binomial Tree Model**  
and analyzes how volatility, time, and spot price affect option values.
""")

# -------------------------------
# User Inputs
# -------------------------------
symbol = st.text_input(
    "Enter a Stock or Index Symbol (e.g., NIFTY_MID_SELECT.NS, DLF.NS, RELIANCE.NS):",
    value="NIFTY_MID_SELECT.NS"
)

if st.button("📥 Fetch Live Price"):
    S = get_live_price(symbol)
    st.success(f"✅ Live Price for {symbol}: ₹{S:.2f}")
    st.session_state['S'] = S

if 'S' in st.session_state:
    S = st.session_state['S']

    st.subheader("🔧 Option Input Parameters")
    K = st.number_input("Strike Price (₹)", value=S, step=10.0)
    days = st.number_input("Days to Expiry", value=30, step=5)
    r = st.number_input("Risk-free Interest Rate (%)", value=6.0, step=0.1) / 100
    sigma = st.number_input("Volatility (%)", value=25.0, step=1.0) / 100
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    steps = st.slider("Binomial Steps", 10, 200, 100)
    T = days / 365

    # -------------------------------
    # Theory / Explanation Box
    # -------------------------------
    with st.expander("ℹ️ Option Greeks Explained"):
        st.markdown("""
        **Delta (Δ):** Measures how much the option price changes for a small change in the underlying stock price.  
        - Positive for calls, negative for puts.  

        **Gamma (Γ):** Measures how much Delta changes as the underlying price changes.  
        - High Gamma → Delta changes quickly, option is sensitive to spot price movements.  

        **Theta (Θ):** Measures time decay — how much the option loses value as time passes.  
        - Negative for long options (they lose value as expiry approaches).  

        **Vega (ν):** Measures sensitivity of option price to volatility.  
        - Higher volatility → higher option price.  

        **Rho (ρ):** Measures sensitivity to the risk-free interest rate.  
        - Call options increase in value if rates rise, puts decrease.  
        """)

    if st.button("🧮 Calculate Option Price"):
        # --- Calculate Price & Greeks ---
        price, delta, gamma, theta, vega, rho = calculate_greeks_binomial(S, K, T, r, sigma, steps, option_type)

        # --- Display Results ---
        st.subheader("📈 Option Valuation Results (Binomial Model)")
        st.metric("Option Price (₹)", f"{price:.2f}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Delta", f"{delta:.4f}")
        c2.metric("Gamma", f"{gamma:.6f}")
        c3.metric("Theta", f"{theta:.4f}")

        c4, c5 = st.columns(2)
        c4.metric("Vega", f"{vega:.4f}")
        c5.metric("Rho", f"{rho:.4f}")

        # -------------------------------
        # Scenario Analysis Charts
        vols = np.linspace(0.05, 0.6, 20)
        prices_vol = [binomial_option_price(S, K, T, r, v, steps, option_type) for v in vols]
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(vols*100, prices_vol, color='purple', marker='o')
        ax1.set_title("Option Price vs Volatility (Binomial Model)")
        ax1.set_xlabel("Volatility (%)")
        ax1.set_ylabel("Option Price (₹)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

        spots = np.linspace(S*0.8, S*1.2, 20)
        prices_spot = [binomial_option_price(s, K, T, r, sigma, steps, option_type) for s in spots]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(spots, prices_spot, color='teal', marker='o')
        ax2.set_title("Option Price vs Spot Price (Binomial Model)")
        ax2.set_xlabel("Spot Price (₹)")
        ax2.set_ylabel("Option Price (₹)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        times = np.linspace(5/365, 90/365, 20)
        prices_time = [binomial_option_price(S, K, t, r, sigma, steps, option_type) for t in times]
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(times*365, prices_time, color='darkorange', marker='o')
        ax3.set_title("Option Price vs Days to Expiry (Binomial Model)")
        ax3.set_xlabel("Days to Expiry")
        ax3.set_ylabel("Option Price (₹)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig3)

        strike_range = np.linspace(S*0.8, S*1.2, 20)
        prices_strike = [binomial_option_price(S, k, T, r, sigma, steps, option_type) for k in strike_range]
        fig_strike, ax_strike = plt.subplots(figsize=(6, 4))
        ax_strike.plot(strike_range, prices_strike, color='green', marker='o')
        ax_strike.set_title("Option Price vs Strike Price (Binomial Model)")
        ax_strike.set_xlabel("Strike Price (₹)")
        ax_strike.set_ylabel("Option Price (₹)")
        ax_strike.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_strike)

        # --- Summary Table ---
        df = pd.DataFrame({
            "Volatility (%)": vols*100,
            "Option Price (₹)": prices_vol
        })
        st.dataframe(df, use_container_width=True)

        st.info("""
        **Key Insights:**
        - 📈 Higher volatility → higher option price  
        - ⏳ Longer time to expiry → higher option value  
        - 📊 For calls: higher spot price → higher option value  
        - 📉 For puts: higher spot price → lower option value  
        """)
