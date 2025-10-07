# binomial_greeks_app.py
# Streamlit app: CRR Binomial Option Pricing + Greeks (Delta, Gamma, Theta, Vega, Rho)
# Prepared for: Assignment 1 Simulation (CIA III)
# Author: Amanjot (adapted by ChatGPT)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------------------------------
# Binomial pricing (CRR) and Greeks via finite differences
# -------------------------------

def binomial_option_price(S, K, T, r, sigma, steps=100, option_type='call', american=False):
    """
    Cox-Ross-Rubinstein binomial pricing (backward induction).
    Returns option price (float).
    """
    if T <= 0 or steps <= 0:
        return float(max(S - K, 0)) if option_type == 'call' else float(max(K - S, 0))

    dt = T / steps
    # handle sigma ~= 0
    if sigma <= 0:
        payoff = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        return float(np.exp(-r * T) * payoff)

    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp(r * dt) - d) / (u - d)

    # terminal stock prices
    prices = np.array([S * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])

    # terminal payoffs
    if option_type == 'call':
        payoffs = np.maximum(prices - K, 0.0)
    else:
        payoffs = np.maximum(K - prices, 0.0)

    # backward induction
    for i in range(steps - 1, -1, -1):
        payoffs = disc * (p * payoffs[1:i + 2] + (1 - p) * payoffs[0:i + 1])
        if american:
            prices_i = np.array([S * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
            if option_type == 'call':
                payoffs = np.maximum(payoffs, prices_i - K)
            else:
                payoffs = np.maximum(payoffs, K - prices_i)

    return float(payoffs[0])


def greeks_binomial(S, K, T, r, sigma, steps=200, option_type='call'):
    """
    Compute Greeks using centered finite differences based on the CRR price.
    Returns dict: price, delta, gamma, theta_per_day, vega, rho_per_pct
    """
    # baseline price
    price = binomial_option_price(S, K, T, r, sigma, steps, option_type)

    # choose bumps
    eps_S = max(0.01, 0.001 * S)           # spot bump
    eps_sigma = max(1e-4, 0.001 * max(sigma, 0.01))  # vol bump
    eps_r = 1e-4                           # rate bump (absolute)
    eps_days = 1                           # theta per 1 day

    # Delta: central difference
    price_up = binomial_option_price(S + eps_S, K, T, r, sigma, steps, option_type)
    price_down = binomial_option_price(S - eps_S, K, T, r, sigma, steps, option_type)
    delta = (price_up - price_down) / (2 * eps_S)

    # Gamma: second derivative wrt S
    gamma = (price_up - 2 * price + price_down) / (eps_S ** 2)

    # Theta per day: use T - 1 day
    dt_day = eps_days / 365.0
    T_minus = max(1/365.0, T - dt_day)
    price_Tminus = binomial_option_price(S, K, T_minus, r, sigma, steps, option_type)
    # theta_per_day = (price_Tminus - price) / dt_day  (price drops as time passes -> negative)
    theta_per_day = (price_Tminus - price) / dt_day

    # Vega: central difference in sigma
    price_sigma_up = binomial_option_price(S, K, T, r, sigma + eps_sigma, steps, option_type)
    price_sigma_down = binomial_option_price(S, K, T, r, max(1e-8, sigma - eps_sigma), steps, option_type)
    vega = (price_sigma_up - price_sigma_down) / (2 * eps_sigma)

    # Rho: central difference in rate (per absolute 1.0). Convert to per 1% (multiply by 0.01)
    price_r_up = binomial_option_price(S, K, T, r + eps_r, sigma, steps, option_type)
    price_r_down = binomial_option_price(S, K, T, r - eps_r, sigma, steps, option_type)
    rho_per_abs = (price_r_up - price_r_down) / (2 * eps_r)
    rho_per_pct = rho_per_abs * 0.01

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta_per_day': theta_per_day,
        'vega': vega,
        'rho_per_pct': rho_per_pct
    }


# -------------------------------
# Streamlit UI (focused on Binomial + Greeks)
# -------------------------------

st.set_page_config(page_title='Binomial Pricing + Greeks', layout='wide')
st.title('📊 Binomial Option Pricing — CRR + Greeks')
st.caption('Simplified app: computes binomial option price and Greeks via finite differences')

st.markdown('''
This tool prices **European or American** options using the Cox–Ross–Rubinstein (CRR) binomial tree
and estimates the Greeks (Delta, Gamma, Theta per day, Vega, Rho) using finite differences.

Use this for scenario analysis required by your assignment.
''')

# Input panel
with st.sidebar.form('inputs'):
    st.header('Inputs')
    S = st.number_input('Spot Price (S)', value=150.0, step=1.0)
    K = st.number_input('Strike Price (K)', value=160.0, step=1.0)
    days = st.number_input('Days to expiry', min_value=1, max_value=3650, value=30, step=1)
    r_pct = st.number_input('Risk-free rate (annual %)', value=6.0, step=0.01)
    sigma_pct = st.number_input('Volatility (annual %)', value=25.0, step=0.1)
    option_type = st.selectbox('Option type', ('call', 'put'))
    steps = st.slider('Binomial steps', min_value=1, max_value=2000, value=200, step=1)
    american = st.checkbox('American option (allow early exercise)', value=False)
    submit = st.form_submit_button('Calculate')

r = r_pct / 100.0
sigma = sigma_pct / 100.0
T = days / 365.0

if submit:
    # compute
    price = binomial_option_price(S, K, T, r, sigma, steps, option_type, american)
    g = greeks_binomial(S, K, T, r, sigma, steps, option_type)

    # display results
    st.subheader('Result — Binomial Model')
    c1, c2, c3 = st.columns(3)
    c1.metric('Option Price (₹)', f"{g['price']:.6f}")
    c2.metric('Delta', f"{g['delta']:.6f}")
    c3.metric('Gamma', f"{g['gamma']:.6f}")

    c4, c5, c6 = st.columns(3)
    c4.metric('Theta (per day)', f"{g['theta_per_day']:.6f}")
    c5.metric('Vega', f"{g['vega']:.6f}")
    c6.metric('Rho (per 1%)', f"{g['rho_per_pct']:.6f}")

    # quick comparison: small scenario table
    st.subheader('Scenario Table — change one input at a time')
    scenarios = []
    # vary spot
    for s in [S * 0.9, S, S * 1.1]:
        p = binomial_option_price(s, K, T, r, sigma, steps, option_type, american)
        gtmp = greeks_binomial(s, K, T, r, sigma, steps, option_type)
        scenarios.append({'Scenario': f'Spot={s:.2f}', 'Price': p, 'Delta': gtmp['delta'], 'Theta/day': gtmp['theta_per_day']})
    # vary sigma
    for sp in [sigma * 0.5, sigma, sigma * 1.5]:
        p = binomial_option_price(S, K, T, r, sp, steps, option_type, american)
        gtmp = greeks_binomial(S, K, T, r, sp, steps, option_type)
        scenarios.append({'Scenario': f'Vol={sp*100:.1f}%', 'Price': p, 'Delta': gtmp['delta'], 'Theta/day': gtmp['theta_per_day']})

    df_scen = pd.DataFrame(scenarios)
    st.dataframe(df_scen, use_container_width=True)

    # plots for report
    st.subheader('Charts for report')
    # Price vs Volatility
    vols = np.linspace(max(0.01, sigma * 0.5), sigma * 2.0, 20)
    prices_vol = [binomial_option_price(S, K, T, r, v, steps, option_type, american) for v in vols]
    fig1, ax1 = plt.subplots()
    ax1.plot(vols * 100, prices_vol, marker='o')
    ax1.set_xlabel('Volatility (%)')
    ax1.set_ylabel('Option Price (₹)')
    ax1.set_title('Option Price vs Volatility (Binomial)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # Price vs Spot
    spots = np.linspace(S * 0.7, S * 1.3, 20)
    prices_spot = [binomial_option_price(s, K, T, r, sigma, steps, option_type, american) for s in spots]
    fig2, ax2 = plt.subplots()
    ax2.plot(spots, prices_spot, marker='o')
    ax2.set_xlabel('Spot Price (₹)')
    ax2.set_ylabel('Option Price (₹)')
    ax2.set_title('Option Price vs Spot Price (Binomial)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    # Delta vs Vol
    deltas_vol = [greeks_binomial(S, K, T, r, v, steps, option_type)['delta'] for v in vols]
    fig3, ax3 = plt.subplots()
    ax3.plot(vols * 100, deltas_vol, marker='o')
    ax3.set_xlabel('Volatility (%)')
    ax3.set_ylabel('Delta')
    ax3.set_title('Delta vs Volatility (Binomial)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

    st.markdown('''
    **Report tips:** include the scenario table, charts above, and short interpretations:
    - How price changes with volatility, spot and time
    - How Greeks change and what that implies for hedging
    - Mention assumptions (no dividends by default, continuous compounding of r in tree)
    ''')

    # export option: CSV of scenario table
    csv = df_scen.to_csv(index=False).encode('utf-8')
    st.download_button(label='Download scenario table (CSV)', data=csv, file_name='binomial_scenarios.csv', mime='text/csv')

st.markdown('---')
st.caption('This app focuses on the CRR binomial model and computes Greeks required for your assignment.')
