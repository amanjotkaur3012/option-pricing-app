import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import math
import matplotlib.pyplot as plt

def get_live_price(symbol):
    data = yf.download(symbol, period="1d", interval="1m", progress=False)
    if data.empty:
        data = yf.Ticker(symbol).history(period="1d")
    return round(data['Close'].iloc[-1], 2)

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
    return price, d1, d2

def greeks(S,K,T,r,sigma,option_type='call'):
    price,d1,d2 = black_scholes(S,K,T,r,sigma,option_type)
    delta = norm.cdf(d1) if option_type=='call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2 if option_type=='call' else -d2)
    return price,delta,gamma,theta,vega,rho

st.set_page_config(page_title="Option Pricing Prototype", page_icon="📈")
st.title("📊 Option Pricing Prototype App")
st.caption("Developed by Amanjot Kaur | MSc Finance & Analytics")

st.write("""
Use this app to price options using live data (Yahoo Finance).
It calculates the **Option Price** and **Greeks** using the Black-Scholes model.
""")

symbol = st.text_input("Enter Stock Symbol (e.g. RELIANCE.NS, INFY.NS):", "RELIANCE.NS")

if st.button("Fetch Live Price"):
    try:
        S = get_live_price(symbol)
        st.success(f"✅ Live Spot Price for {symbol}: ₹{S}")
        st.session_state['S'] = S
    except Exception as e:
        st.error("Could not fetch live price. Try another symbol.")

if 'S' in st.session_state:
    S = st.session_state['S']
    K = st.number_input("Strike Price (₹)", value=S, step=10.0)
    days = st.number_input("Days to Expiry", value=30, step=5)
    r = st.number_input("Risk-free Rate (%)", value=6.0, step=0.1)/100
    sigma = st.number_input("Volatility (%)", value=25.0, step=1.0)/100
    option_type = st.selectbox("Option Type", ["call", "put"])
    T = days/365

    if st.button("Calculate Option Price & Greeks"):
        price,delta,gamma,theta,vega,rho = greeks(S,K,T,r,sigma,option_type)
        st.subheader("Results")
        st.metric("Option Price (₹)", f"{price:.2f}")
        col1,col2,col3 = st.columns(3)
        col1.metric("Delta", f"{delta:.4f}")
        col2.metric("Gamma", f"{gamma:.6f}")
        col3.metric("Theta", f"{theta:.4f}")
        col4,col5 = st.columns(2)
        col4.metric("Vega", f"{vega:.4f}")
        col5.metric("Rho", f"{rho:.4f}")

        # Charts
        st.subheader("📈 Scenario Analysis")

        # 1️⃣ Price vs Volatility
        vols = np.linspace(0.05, 0.6, 12)
        prices_vol = [black_scholes(S,K,T,r,v,option_type)[0] for v in vols]
        fig1, ax1 = plt.subplots()
        ax1.plot(vols*100, prices_vol, color="purple", marker="o")
        ax1.set_title("Option Price vs Volatility")
        ax1.set_xlabel("Volatility (%)")
        ax1.set_ylabel("Option Price (₹)")
        st.pyplot(fig1)

        # 2️⃣ Price vs Spot
        spots = np.linspace(S*0.7, S*1.3, 12)
        prices_spot = [black_scholes(s,K,T,r,sigma,option_type)[0] for s in spots]
        fig2, ax2 = plt.subplots()
        ax2.plot(spots, prices_spot, color="teal", marker="o")
        ax2.set_title("Option Price vs Spot Price")
        ax2.set_xlabel("Spot Price (₹)")
        ax2.set_ylabel("Option Price (₹)")
        st.pyplot(fig2)

        st.info("""
        ✅ **Observations**
        - Higher volatility → higher option price (Vega effect)  
        - For calls: higher spot → higher price (Delta effect)  
        - For puts: higher spot → lower price  
        """)
