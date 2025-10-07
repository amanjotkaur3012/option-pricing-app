import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import math
from io import BytesIO

# -------------------------------
# Helper Functions
# -------------------------------

# Fetch live price
def get_live_price(symbol):
    try:
        data = yf.download(symbol, period="1d", interval="1m", progress=False)
        if data.empty:
            data = yf.Ticker(symbol).history(period="1d")
        return float(data['Close'].iloc[-1])
    except Exception:
        st.warning("⚠️ Could not fetch live data, using default ₹100")
        return 100.0

# Black-Scholes Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price, d1, d2

# Greeks Calculation
def greeks(S, K, T, r, sigma, option_type='call'):
    price, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2)
    return price, delta, gamma, theta, vega, rho

# Binomial Model
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
# Streamlit App
# -------------------------------

st.set_page_config(page_title="Option Pricing Simulator", page_icon="📈", layout="wide")

st.title("📊 Option Pricing & Greeks Simulator")
st.caption("Developed by Amanjot Kaur | MSc Finance & Analytics | Christ University")

st.markdown("""
This advanced simulator prices **Equity or Index Options** using:
- **Black-Scholes** and **Binomial Models**
- Computes all key **Greeks** (Δ, Γ, Θ, Vega, Rho)
- Runs **multi-factor scenario analysis** with visualization and report download.
""")

# -------------------------------
# User Inputs
# -------------------------------
symbol = st.text_input("Enter Stock/Index Symbol (e.g., RELIANCE.NS, ^NSEI):", value="RELIANCE.NS")

if st.button("📥 Fetch Live Price"):
    S = get_live_price(symbol)
    st.success(f"✅ Live Price for {symbol}: ₹{S}")
    st.session_state['S'] = S

if 'S' in st.session_state:
    S = st.session_state['S']
    st.subheader("🔧 Option Parameters")

    K = st.number_input("Strike Price (₹)", value=S, step=10.0)
    days = st.number_input("Days to Expiry", value=30, step=5)
    T = days / 365
    r = st.number_input("Risk-Free Interest Rate (%)", value=6.0, step=0.1) / 100
    sigma = st.number_input("Volatility (%)", value=25.0, step=1.0) / 100
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    model = st.radio("Model", ["Black-Scholes", "Binomial"], horizontal=True)
    steps = st.slider("Binomial Steps (if Binomial Model)", 10, 200, 100)

    if st.button("🧮 Calculate"):
        if model == "Black-Scholes":
            price, delta, gamma, theta, vega, rho = greeks(S, K, T, r, sigma, option_type)
        else:
            price = binomial_option_price(S, K, T, r, sigma, steps, option_type)
            delta = gamma = theta = vega = rho = np.nan

        st.subheader("💰 Option Valuation Results")
        st.metric("Option Price (₹)", f"{price:.2f}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Delta (Δ)", f"{delta:.4f}")
        c2.metric("Gamma (Γ)", f"{gamma:.6f}")
        c3.metric("Theta (Θ)", f"{theta:.4f}")
        c4, c5 = st.columns(2)
        c4.metric("Vega", f"{vega:.4f}")
        c5.metric("Rho", f"{rho:.4f}")

        # -------------------------------
        # Tabs for Scenario Analysis
        # -------------------------------
        st.subheader("📊 Scenario Analysis")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Volatility Impact", "Spot Price Impact", "Strike Price Impact", "Time to Expiry", "Interest Rate Impact"
        ])

        # 1️⃣ Volatility Impact
        with tab1:
            vols = np.linspace(0.05, 0.6, 15)
            prices = []
            vegas = []
            for v in vols:
                p, d1, d2 = black_scholes(S, K, T, r, v, option_type)
                prices.append(p)
                vegas.append(S*norm.pdf(d1)*np.sqrt(T))
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(vols*100, prices, color='blue', label='Option Price')
            ax2.plot(vols*100, vegas, color='green', linestyle='--', label='Vega')
            ax1.set_xlabel("Volatility (%)")
            ax1.set_ylabel("Option Price (₹)")
            ax2.set_ylabel("Vega")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)
            st.info("Higher volatility increases both **option price** and **Vega**, indicating greater sensitivity to volatility.")

        # 2️⃣ Spot Price Impact
        with tab2:
            spots = np.linspace(S*0.8, S*1.2, 15)
            prices = [black_scholes(s, K, T, r, sigma, option_type)[0] for s in spots]
            deltas = [black_scholes(s, K, T, r, sigma, option_type)[1] for s in spots]
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(spots, prices, color='blue', label='Option Price')
            ax2.plot(spots, [norm.cdf(d1) for d1 in deltas], color='orange', linestyle='--', label='Delta')
            ax1.set_xlabel("Spot Price (₹)")
            ax1.set_ylabel("Option Price (₹)")
            ax2.set_ylabel("Delta")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)
            st.info("As spot price rises, **call options gain value (positive Delta)** while **put options lose value (negative Delta)**.")

        # 3️⃣ Strike Price Impact
        with tab3:
            Ks = np.linspace(S*0.8, S*1.2, 15)
            prices = [black_scholes(S, k, T, r, sigma, option_type)[0] for k in Ks]
            fig, ax = plt.subplots()
            ax.plot(Ks, prices, color='purple', marker='o')
            ax.set_xlabel("Strike Price (₹)")
            ax.set_ylabel("Option Price (₹)")
            ax.set_title("Option Price vs Strike Price")
            st.pyplot(fig)
            st.info("Options with lower strike (ITM) are more expensive. Price declines as strike increases (OTM).")

        # 4️⃣ Time to Expiry Impact
        with tab4:
            times = np.array([10, 30, 60, 90, 120, 180])
            prices = [black_scholes(S, K, t/365, r, sigma, option_type)[0] for t in times]
            fig, ax = plt.subplots()
            ax.plot(times, prices, color='teal', marker='o')
            ax.set_xlabel("Days to Expiry")
            ax.set_ylabel("Option Price (₹)")
            ax.set_title("Option Price vs Time to Expiry")
            st.pyplot(fig)
            st.info("Longer time to expiry increases option value due to higher **time value**. As expiry nears, **Theta decay** accelerates.")

        # 5️⃣ Interest Rate Impact
        with tab5:
            rates = np.linspace(0.01, 0.15, 10)
            prices = [black_scholes(S, K, T, r_, sigma, option_type)[0] for r_ in rates]
            fig, ax = plt.subplots()
            ax.plot(rates*100, prices, color='red', marker='o')
            ax.set_xlabel("Risk-Free Rate (%)")
            ax.set_ylabel("Option Price (₹)")
            ax.set_title("Option Price vs Risk-Free Rate")
            st.pyplot(fig)
            st.info("Higher risk-free rates slightly increase **call prices** and decrease **put prices** due to present value effects (Rho).")

        # -------------------------------
        # Comparative Scenario Table
        # -------------------------------
        st.subheader("📋 Comparative Scenario Table")

        vol_scenarios = [0.1, 0.2, 0.3, 0.4]
        data = []
        for v in vol_scenarios:
            price, delta, gamma, theta, vega, rho = greeks(S, K, T, r, v, option_type)
            data.append([v*100, price, delta, gamma, theta, vega, rho])

        df = pd.DataFrame(data, columns=["Volatility (%)", "Price (₹)", "Delta", "Gamma", "Theta", "Vega", "Rho"])
        st.dataframe(df.style.format("{:.4f}"))

        # Download button
        excel = BytesIO()
        df.to_excel(excel, index=False)
        st.download_button("📥 Download Analysis (Excel)", data=excel.getvalue(), file_name="option_analysis.xlsx")

        # -------------------------------
        # Summary Insights
        # -------------------------------
        st.subheader("🧠 Key Insights Summary")
        st.markdown(f"""
        - **Volatility (σ):** Higher volatility ⇒ higher option price and Vega sensitivity.  
        - **Spot Price (S):** For calls, price rises with S (Δ > 0); for puts, price falls (Δ < 0).  
        - **Strike Price (K):** Lower K ⇒ deeper ITM ⇒ higher intrinsic value.  
        - **Time to Expiry (T):** Longer maturity increases value due to time premium.  
        - **Risk-Free Rate (r):** Positive correlation with call prices; negative with put prices.  
        """)

        st.success("✅ Complete! You can now export your analysis and include charts in your final report.")
