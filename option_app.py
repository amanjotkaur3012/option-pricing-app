honestly --- this is what i did but the thing is we are not tought blacl scholes model till now so --- only binomial is taught to us -- ---- the python code is this -- import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
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

# --- Estimate Delta for Binomial using Finite Differences ---
def delta_binomial(S, K, T, r, sigma, steps=100, option_type='call', h=0.01):
    price_up = binomial_option_price(S+h, K, T, r, sigma, steps, option_type)
    price_down = binomial_option_price(S-h, K, T, r, sigma, steps, option_type)
    return (price_up - price_down) / (2*h)

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
symbol = st.text_input(
    "Enter a Stock or Index Symbol (e.g., NIFTY_MID_SELECT.NS, DLF.NS, RELIANCE.NS):",
    value="NIFTY_MID_SELECT.NS"
)

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
        # Calculate option price and Greeks
        if model == "Black-Scholes":
            price, delta, gamma, theta, vega, rho = greeks(S, K, T, r, sigma, option_type)
            price_bs = price
            delta_bs = delta
            price_bin = binomial_option_price(S, K, T, r, sigma, steps, option_type)
            delta_bin = delta_binomial(S, K, T, r, sigma, steps, option_type)
        else:
            price = binomial_option_price(S, K, T, r, sigma, steps, option_type)
            delta = delta_binomial(S, K, T, r, sigma, steps, option_type)
            gamma = theta = vega = rho = np.nan
            price_bin = price
            delta_bin = delta
            price_bs, delta_bs, gamma_bs, theta_bs, vega_bs, rho_bs = greeks(S, K, T, r, sigma, option_type)

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
        # Comparative Table: BS vs Binomial
        df_compare = pd.DataFrame({
            "Model": ["Black-Scholes", "Binomial"],
            "Option Price (₹)": [price_bs, price_bin],
            "Delta": [delta_bs, delta_bin],
            "Gamma": [gamma_bs if 'gamma_bs' in locals() else np.nan, np.nan],
            "Theta": [theta_bs if 'theta_bs' in locals() else np.nan, np.nan],
            "Vega": [vega_bs if 'vega_bs' in locals() else np.nan, np.nan],
            "Rho": [rho_bs if 'rho_bs' in locals() else np.nan, np.nan]
        })
        st.subheader("📊 Comparative Table: Black-Scholes vs Binomial")
        st.dataframe(df_compare, use_container_width=True)

        # -------------------------------
        # Scenario Analysis Charts

        # Option Price vs Volatility
        vols = np.linspace(0.05, 0.6, 20)
        prices_vol = [black_scholes(S, K, T, r, v, option_type)[0] for v in vols]
        fig1, ax1 = plt.subplots()
        ax1.plot(vols*100, prices_vol, color='purple', marker='o')
        ax1.set_title("Option Price vs Volatility")
        ax1.set_xlabel("Volatility (%)")
        ax1.set_ylabel("Option Price (₹)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

        # Option Price vs Spot Price
        spots = np.linspace(S*0.8, S*1.2, 20)
        prices_spot = [black_scholes(s, K, T, r, sigma, option_type)[0] for s in spots]
        fig2, ax2 = plt.subplots()
        ax2.plot(spots, prices_spot, color='teal', marker='o')
        ax2.set_title("Option Price vs Spot Price")
        ax2.set_xlabel("Spot Price (₹)")
        ax2.set_ylabel("Option Price (₹)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig2)

        # Option Price vs Time to Expiry
        times = np.linspace(5/365, 90/365, 20)
        prices_time = [black_scholes(S, K, t, r, sigma, option_type)[0] for t in times]
        fig3, ax3 = plt.subplots()
        ax3.plot(times*365, prices_time, color='darkorange', marker='o')
        ax3.set_title("Option Price vs Days to Expiry")
        ax3.set_xlabel("Days to Expiry")
        ax3.set_ylabel("Option Price (₹)")
        ax3.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig3)

        # Option Price vs Strike Price
        strike_range = np.linspace(S*0.8, S*1.2, 20)
        prices_strike = [black_scholes(S, k, T, r, sigma, option_type)[0] for k in strike_range]
        fig_strike, ax_strike = plt.subplots()
        ax_strike.plot(strike_range, prices_strike, color='green', marker='o')
        ax_strike.set_title("Option Price vs Strike Price (Black-Scholes)")
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
        - 📈 Higher volatility → higher option price (Vega effect)
        - ⏳ Longer time to expiry → higher option value (Theta effect)
        - 📊 For calls: higher spot price → higher option value  
        - 📉 For puts: higher spot price → lower option value
        """)---------------- nd the assignment ques is this -- now help me rectify this --Assignment 1 Simulation (CIA III- Sum of Parts Pedagogy) Marks 20
Pricing Index / Equity Options and Developing a Working Model for Financial
Decision-Making Pricing of Equity Options and Index option
 Prepare a Python or Excel-based program to price index options and equity option,
focusing on Equity derivatives.
 Analyze factors affecting option pricing, including the Greeks.
 Compare the results from different scenarios and prepare a comparative report.
Instructions:
1. Develop the Program: (5 Marks)
o Create a program in Python or Excel to price Equity options.
o Incorporate models such as Binominal Option model to calculate option
prices.
o Include the calculation of Greeks (Delta, Gamma, Theta, Vega, Rho) to
understand how different factors affect option prices.

2. Factors Affecting Option Pricing: (5 Marks)
o Analyze the impact of underlying asset price, strike price, volatility, time to
expiration, and risk-free interest rate on option pricing.
o Use the program to simulate different scenarios and observe changes in option
prices and Greeks.

3. Comparative Report and Viva : (12 Marks)
o Compare the results obtained from different scenarios.
o Prepare a report detailing the analysis, including graphs and charts to illustrate
findings.
o Discuss the implications of the results and how the factors and Greeks
influence option pricing. 
