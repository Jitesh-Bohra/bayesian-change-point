import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Bayesian Change-Point Detection", layout="wide")

# --- App Description ---
st.title("Bayesian Change-Point Detection Dashboard")
st.markdown("""
**Welcome!** This app demonstrates how a Bayesian model detects a "change-point" ($t_0$) in a data sequence where the mean shifts from $\mu_1$ to $\mu_2$.
* **Sidebar**: Set your **Prior Beliefs** (what the model assumes) and **Actual Values** (how data is generated).
* **Buttons**: Add data points from either the first or second regime.
* **Plots**: Watch the data sequence update and see the model's posterior distribution for the change-point $t_0$.
""")

# --- Initialization ---
if 'y_vals' not in st.session_state:
    st.session_state.y_vals = []
if 'regime_labels' not in st.session_state:
    st.session_state.regime_labels = []

# --- Sidebar: Parameters ---
st.sidebar.header("1. Prior Hyperparameters")
u1_tilde = st.sidebar.slider("Prior Mean μ1 (~u1)", -10.0, 10.0, 5.0)
u2_tilde = st.sidebar.slider("Prior Mean μ2 (~u2)", -10.0, 10.0, 10.0)
sigma0_sq = st.sidebar.slider("Prior σ0² (Data Variance)", 0.1, 10.0, 1.0)
sigma1_sq = st.sidebar.slider("Prior σ1² (μ1 Variance)", 0.1, 10.0, 1.0)
sigma2_sq = st.sidebar.slider("Prior σ2² (μ2 Variance)", 0.1, 10.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Actual Generation Values")
u1_actual = st.sidebar.slider("Actual μ1", -10.0, 10.0, 5.0)
u2_actual = st.sidebar.slider("Actual μ2", -10.0, 10.0, 10.0)
gen_sigma = st.sidebar.slider("Generation σ", 0.1, 5.0, 1.0)

# --- Interaction Buttons ---
st.write("### Data Generation")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(f"Generate Regime 1 (μ={u1_actual})"):
        st.session_state.y_vals.append(np.random.normal(u1_actual, gen_sigma))
        st.session_state.regime_labels.append(1)
with col2:
    if st.button(f"Generate Regime 2 (μ={u2_actual})"):
        st.session_state.y_vals.append(np.random.normal(u2_actual, gen_sigma))
        st.session_state.regime_labels.append(2)
with col3:
    if st.button("Reset Experiment"):
        st.session_state.y_vals = []
        st.session_state.regime_labels = []

# --- Analysis & Visualisation ---
y = np.array(st.session_state.y_vals)
regimes = np.array(st.session_state.regime_labels)
n = len(y)

if n > 1:
    # Posterior Calculation for t0 (Discrete Uniform Prior on {1...99})
    # Using log-likelihood for stability [cite: 2535]
    t0_range = np.arange(1, 100)
    log_post = np.zeros(len(t0_range))
    
    for i, t0 in enumerate(t0_range):
        if t0 >= n:
            # Case: Change hasn't happened yet (All points ~ N(u1_tilde, sigma0_sq))
            log_post[i] = np.sum(norm.logpdf(y, loc=u1_tilde, scale=np.sqrt(sigma0_sq)))
        else:
            # Case: Change at t0 (Split points between u1_tilde and u2_tilde)
            ll1 = np.sum(norm.logpdf(y[:t0], loc=u1_tilde, scale=np.sqrt(sigma0_sq)))
            ll2 = np.sum(norm.logpdf(y[t0:], loc=u2_tilde, scale=np.sqrt(sigma0_sq)))
            log_post[i] = ll1 + ll2
            
    # Normalize to get Probabilities
    probs = np.exp(log_post - np.max(log_post))
    probs /= np.sum(probs)
    
    # Calculate Posterior Mean [cite: 1443]
    post_mean_t0 = np.sum(t0_range * probs)

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Observed y_k with formatted regime colors
    idx1 = np.where(regimes == 1)[0]
    idx2 = np.where(regimes == 2)[0]
    
    # Connect points within regimes with different colors
    if len(idx1) > 0:
        ax1.plot(idx1, y[idx1], color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.6)
        ax1.scatter(idx1, y[idx1], color='#1f77b4', label="Regime 1", zorder=5)
    if len(idx2) > 0:
        ax1.plot(idx2, y[idx2], color='#ff7f0e', linestyle='-', linewidth=1.5, alpha=0.6)
        ax1.scatter(idx2, y[idx2], color='#ff7f0e', label="Regime 2", zorder=5)
    
    ax1.set_title(f"Sequence of Observed Data (k={n})", fontsize=14)
    ax1.set_ylabel("Value $y_k$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Posterior of t0
    ax2.bar(t0_range, probs, color='teal', alpha=0.6, width=1.0, label="Posterior $P(t_0 | y)$")
    ax2.axvline(post_mean_t0, color='red', linestyle='--', linewidth=2, 
                label=f"Posterior Mean (E[t0] ≈ {post_mean_t0:.2f})")
    
    ax2.set_title("Posterior Probability of Change-Point $t_0$", fontsize=14)
    ax2.set_xlabel("Hypothesized Change-Point ($m$)")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    st.pyplot(fig)
else:
    st.info("Please generate at least two data points to see the Bayesian analysis.")
