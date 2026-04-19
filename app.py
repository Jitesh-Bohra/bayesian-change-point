import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Bayesian Change-Point", layout="wide")

# Persistent data storage
if 'y_history' not in st.session_state:
    st.session_state.y_history = []

st.title("Sequential Bayesian Change-Point Detection")

# --- Sidebar Sliders ---
st.sidebar.header("Model Parameters")
u1 = st.sidebar.slider("μ1 (Regime 1)", -10.0, 10.0, 5.0)
u2 = st.sidebar.slider("μ2 (Regime 2)", -10.0, 10.0, 10.0)
sigma = st.sidebar.slider("σ (Common Variance)", 0.1, 5.0, 1.0)

# --- Data Generation Buttons ---
st.write("### 1. Generate Data")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button(f"Generate y_k from Regime 1 (μ={u1})"):
        st.session_state.y_history.append(np.random.normal(u1, sigma))
with c2:
    if st.button(f"Generate y_k from Regime 2 (μ={u2})"):
        st.session_state.y_history.append(np.random.normal(u2, sigma))
with c3:
    if st.button("Reset Data"):
        st.session_state.y_history = []

# --- Posterior Calculation ---
y = np.array(st.session_state.y_history)
k = len(y)

if k > 1:
    # Prior: Discrete Uniform on {1, ..., 99}
    ms = np.arange(1, 100)
    log_post = np.zeros(len(ms))
    
    for i, m in enumerate(ms):
        if m >= k:
            # Entire sequence so far belongs to Regime 1
            log_post[i] = np.sum(norm.logpdf(y, loc=u1, scale=sigma))
        else:
            # Sequence splits at m
            ll1 = np.sum(norm.logpdf(y[:m], loc=u1, scale=sigma))
            ll2 = np.sum(norm.logpdf(y[m:], loc=u2, scale=sigma))
            log_post[i] = ll1 + ll2
    
    # Normalize probabilities using the log-sum-exp trick
    probs = np.exp(log_post - np.max(log_post))
    probs /= np.sum(probs)

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(y, 'o-', color='teal', markersize=4)
    ax1.set_title(f"Observations ($y_1$ to $y_{k}$)")
    ax1.set_ylabel("Value")
    
    ax2.bar(ms, probs, color='coral', alpha=0.8, width=1.0)
    ax2.set_xlim(0, 100)
    ax2.set_title("Posterior Probability of Change-Point ($t_0$)")
    ax2.set_xlabel("Time Step (m)")
    
    st.pyplot(fig)
else:
    st.info("Add points using the buttons to begin detection.")
