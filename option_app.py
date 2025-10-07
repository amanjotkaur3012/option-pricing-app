# app.py
import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

# -------------------------------
# Helper Functions
# -------------------------------

def get_live_price(symbol):
    """Fetch latest close price from Yahoo Finance. Returns float price or None."""
    try:
        data = yf.download(symbol, period="5d", interval="1d", progress=False)
        if data is None or data.empty:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if hist is None or hist.empty:
                return None
            return float(hist['Close'].iloc[-1])
        return float(data['Close'].iloc[-1])
    except Exception:
        return None

# --- Binomial Option Price (CRR) ---
def binomial_option_price(S, K, T, r, sigma, steps=100, option_type='call', american=False):
    """
    CRR binomial model (backwards induction).
    S: spot
    K: strike
    T: time in years
    r: risk-free rate (annual, continuous assumed via exp(r*dt))
    sigma: volatility (annual)
    steps: number of binomial steps
    option_type: 'call' or 'put'
    american: if True, allow early exercise (for American options)
    """
    if T <= 0 or steps <= 0:
        # immediate expiry: payoff at maturity
        return max(S - K, 0) if option_type=='call' else max(K - S, 0)

    dt = T / steps
    # Avoid zero sigma
    if sigma <= 0:
        # deterministic up/down (u=1, d=1) — option value reduces to discounted payoff expectation
        payoff = max(S - K, 0) if option_type=='call' else max(K - S, 0)
        return np.exp(-r*T) * payoff

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    # Risk-neutral probability
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # Build terminal prices
    prices = np.array([S * (u**j) * (d**(steps-j)) for j in range(steps+1)])
    # Terminal payoff
    if option_type == 'call':
        payoffs = np.maximum(prices - K, 0.0)
    else:
        payoffs = np.maximum(K - prices, 0.0)

    # Backward induction
    for i in range(steps-1, -1, -1):
        payoffs = disc * (p * payoffs[1:i+2] + (1 - p) * payoffs[0:i+1])
        if american:
            # If American, compare to immediate exercise at nodes:
            prices = np.array([S * (u**j) * (d**(i-j)) for j in range(i+1)])
            if option_type == 'call':
                payoffs = np.maximum(payoffs, prices - K)
            else:
                payoffs = np.maximum(payoffs, K - prices)
    return float(payoffs[0])

# --- Greeks via finite differences (based on binomial model) ---
def greeks_binomial(S, K, T, r, sigma, steps=200, option_type='call'):
    """
    Compute Delta, Gamma, Theta, Vega, Rho using finite differences for the binomial price.
    Uses central differences where appropriate and safe relative bumps.
    """
    # Small relative bumps
    eps_S = max(0.01, 0.001 * S)        # absolute bump for spot
    eps_sigma = max(1e-4, 0.001 * sigma) # bump for vol
    eps_r = 1e-4                         # bump for rate (~0.01% absolute)
    eps_T_days = 1                       # theta computed per 1 day

    # Price baseline
    price = binomial_option_price(S, K, T, r, sigma, steps, option_type)

    # Delta (central)
    price_up = binomial_option_price(S + eps_S, K, T, r, sigma, steps, option_type)
    price_down = binomial_option_price(S - eps_S, K, T, r, sigma, steps, option_type)
    delta = (price_up - price_down) / (2 * eps_S)

    # Gamma (second derivative wrt S)
    gamma = (price_up - 2 * price + price_down) / (eps_S ** 2)

    # Theta: use one-day forward (so negative T). Theta per day
    dt_day = eps_T_days / 365.0
    T_minus = max(1/365.0, T - dt_day)  # avoid zero or negative T
    price_Tminus = binomial_option_price(S, K, T_minus, r, sigma, steps, option_type)
    theta_per_day = (price_Tminus - price) / dt_day  # change per day (note sign convention)
    # Many texts define theta per day negative as price decays; here we return per-day change

    # Vega (w.r.t sigma)
    price_sigma_up = binomial_option_price(S, K, T, r, sigma + eps_sigma, steps, option_type)
    price_sigma_down = binomial_option_price(S, K, T, r, max(1e-8, sigma - eps_sigma), steps, option_type)
    vega = (price_sigma_up - price_sigma_down) / (2 * eps_sigma)

    # Rho (w.r.t r) - per 1% (0.01 absolute) or per unit? We'll return per 1% point: scale accordingly
    price_r_up = binomial_option_price(S, K, T, r + eps_r, sigma, steps, option_type)
    price_r_down = binomial_option_price(S, K, T, r - eps_r, sigma, steps, option_type)
    rho = (price_r_up - price_r_down) / (2 * eps_r)  # per absolute rate (i.e., per 1.0 = 100%)
    # To express rho per 1% (i.e., change for 1 percentage point), you can multiply by 0.01 outside

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta_per_day": theta_per_day,
        "vega": vega,
        "rho_per_abs": rho
    }

# Optional: Black-Scholes (only for comparison / illustration)
def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return float(max(S - K, 0)) if option_type=='call' else float(max(K - S, 0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return float(price)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Binomial Option Pricing — Assignment", layout="wide")
st.title("📌 Binomial Option Pricing & Greeks — Assignment")
st.caption("Prepared by Amanjot | Focus on Binomial model (CRR) — Greeks via finite differences")

st.markdown("""
**What this tool does (aligned to your assignment):**
- Prices equity/index options using the **Cox-Ross-Rubinstein (binomial)** tree.
- Computes Greeks (Delta, Gamma, Theta per day, Vega, Rho) using finite differences from the binomial price.
- Allows scenario simulations (varying spot, volatility, time, strike) and produces charts you can include in your report.
""")

# Inputs
col1, col2 = st.columns([2, 1])
with col1:
    symbol = st.text_input("Enter ticker (Yahoo Finance), e.g. RELIANCE.NS or NIFTY 50 use '^NSEI' for index", value="RELIANCE.NS")
    if st.button("Fetch live price"):
        p = get_live_price(symbol)
        if p is None:
            st.warning("Could not fetch live price. Enter spot manually.")
            st.session_state['S'] = None
        else:
            st.success(f"Live price fetched: ₹{p:.2f}")
            st.session_state['S'] = p

with col2:
    use_live = st.checkbox("Use fetched live price when available", value=True)

# Spot: if live exists, prefill, else allow manual
S_default = st.session_state.get('S', None)
S = st.number_input("Spot Price (₹)", value=float(S_default) if S_default else 100.0, step=1.0)

K = st.number_input("Strike Price (₹)", value=round(S), step=1.0)
days = st.number_input("Days to Expiry", min_value=1, max_value=3650, value=30, step=1)
r_pct = st.number_input("Risk-free rate (annual %)", value=6.0, step=0.01)
sigma_pct = st.number_input("Volatility (annual %, e.g. 20 for 20%)", value=25.0, step=0.1)
option_type = st.radio("Option type", ("call", "put"))
steps = st.slider("Binomial steps", min_value=20, max_value=2000, value=200, step=10)
american = st.checkbox("American option (allow early exercise)?", value=False)
compare_bs = st.checkbox("Also compute Black–Scholes price for comparison (optional)", value=False)

# Convert units
r = float(r_pct) / 100.0
sigma = float(sigma_pct) / 100.0
T = float(days) / 365.0

if st.button("Calculate (Binomial)"):
    with st.spinner("Computing binomial price and Greeks..."):
        price_bin = binomial_option_price(S, K, T, r, sigma, steps, option_type, american)
        g = greeks_binomial(S, K, T, r, sigma, steps, option_type)

        if compare_bs:
            price_bs = black_scholes_price(S, K, T, r, sigma, option_type)
        else:
            price_bs = None

    # Display results
    st.subheader("Result — Binomial Model")
    c1, c2, c3 = st.columns(3)
    c1.metric("Option Price (₹)", f"{g['price']:.4f}")
    c2.metric("Delta", f"{g['delta']:.6f}")
    c3.metric("Gamma", f"{g['gamma']:.6f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Theta (per day)", f"{g['theta_per_day']:.6f}")
    c5.metric("Vega (per 1 vol unit)", f"{g['vega']:.6f}")
    # Convert rho to per 1% point for readability:
    rho_per_pct = g['rho_per_abs'] * 0.01
    c6.metric("Rho (per 1%)", f"{rho_per_pct:.6f}")

    # Comparison table (optional)
    df_rows = {
        "Model": ["Binomial (CRR)"],
        "Price (₹)": [g['price']],
        "Delta": [g['delta']],
        "Gamma": [g['gamma']],
        "Theta/day": [g['theta_per_day']],
        "Vega": [g['vega']],
        "Rho (per 1%)": [rho_per_pct]
    }
    if price_bs is not None:
        df_rows["Model"].append("Black–Scholes (analytical)")
        df_rows["Price (₹)"].append(price_bs)
        df_rows["Delta"].append(np.nan)
        df_rows["Gamma"].append(np.nan)
        df_rows["Theta/day"].append(np.nan)
        df_rows["Vega"].append(np.nan)
        df_rows["Rho (per 1%)"].append(np.nan)

    df_compare = pd.DataFrame(df_rows)
    st.subheader("Comparison Table")
    st.dataframe(df_compare, use_container_width=True)

    # -------------------------------
    # Scenario Charts (suggested for report)
    # -------------------------------
    st.subheader("Suggested scenario charts")
    # 1) Price vs Volatility
    vols = np.linspace(max(0.01, sigma*0.5), sigma*2.0, 20)
    prices_vol = [binomial_option_price(S, K, T, r, v, steps, option_type, american) for v in vols]
    fig1, ax1 = plt.subplots()
    ax1.plot(vols*100, prices_vol, marker='o')
    ax1.set_title("Option Price vs Volatility (Binomial)")
    ax1.set_xlabel("Volatility (%)")
    ax1.set_ylabel("Option Price (₹)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)

    # 2) Price vs Spot
    spots = np.linspace(S*0.7, S*1.3, 20)
    prices_spot = [binomial_option_price(s, K, T, r, sigma, steps, option_type, american) for s in spots]
    fig2, ax2 = plt.subplots()
    ax2.plot(spots, prices_spot, marker='o')
    ax2.set_title("Option Price vs Spot Price (Binomial)")
    ax2.set_xlabel("Spot Price (₹)")
    ax2.set_ylabel("Option Price (₹)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)

    # 3) Greeks vs Volatility (example with Delta)
    deltas_vol = []
    for v in vols:
        gtmp = greeks_binomial(S, K, T, r, v, steps, option_type)
        deltas_vol.append(gtmp['delta'])
    fig3, ax3 = plt.subplots()
    ax3.plot(vols*100, deltas_vol, marker='o')
    ax3.set_title("Delta vs Volatility (Binomial)")
    ax3.set_xlabel("Volatility (%)")
    ax3.set_ylabel("Delta")
    ax3.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig3)

    st.info("""
    **Report tips (what to include for full marks)**:
    1. Describe models used (CRR binomial). State assumptions (no dividends, continuous compounding of r).
    2. Show results table comparing prices and Greeks under different scenarios:
       - Vary spot (±20%), strike (ITM/ATM/OTM), volatility (low/medium/high), days to expiry (short/medium/long), and interest rate.
    3. Include the charts above (price vs vol, price vs spot, delta vs vol, theta decay vs days).
    4. Discuss observations: e.g., higher volatility → higher option price; delta increases with spot for calls, etc.
    5. For viva: explain how Greeks were computed (finite differences) and why we used them for the binomial model.
    """)

st.markdown("---")
st.caption("If you want, I can now: (1) generate the graphs and a short comparative report (word/PDF) ready to submit, or (2) convert this to an Excel workbook with scenarios. Tell me which one and I'll create it from these results.")
