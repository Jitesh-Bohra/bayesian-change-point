import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- App Configuration ---
st.set_page_config(page_title="Bayesian Change-Point Detection", layout="wide")

# Persistent state for our sequence of observations: stores (value, regime_label)
if 'y_history' not in st.session_state:
    st.session_state.y_history = []

# --- Title and Description ---
st.title("Live Bayesian Change-Point Detection Dashboard")

st.markdown("""
### How to use this App
1. **Configure Model Priors**: Use the sidebar to set the parameters the model *assumes* to be true (prior beliefs).
2. **Generate Data**: Click the buttons below to generate data points one-by-one. You can switch from Regime 1 to Regime 2 at any time to simulate a "change-point."
3. **Observe Detection**: The top plot displays your data sequence, colored by regime. The bottom plot shows the **Posterior Probability of $t_0$**, identifying where the model thinks the shift occurred.
4. **Analyze the Mean**: The red dashed line represents the **Expected Value (Mean)** of the change-point based on the current evidence.
""")

# --- Sidebar: Prior Parameter Sliders ---
st.sidebar.header("1. Prior / Model Parameters")
st.sidebar.markdown("These values are used by the Bayesian model to calculate the posterior.")

# Prior Means
u1_prior = st.sidebar.slider("Prior Mean μ1 (Regime 1)", -10.0, 10.0, 5.0)
u2_prior = st.sidebar.slider("Prior Mean μ2 (Regime 2)", -10.0, 10.0, 10.0)

# Prior Variances
sigma_model = st.sidebar.slider("Model Variance σ² (Assumed Constant)", 0.1, 10.0, 1.0)
st.sidebar.markdown("---")

st.sidebar.header("2. Actual Generation Parameters")
st.sidebar.markdown("Ground truth values used to sample new points.")
gen_sigma = st.sidebar.slider("Generation σ", 0.1, 5.0, 1.0)

# --- Data Generation Buttons ---
st.write("### Data Generation")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button(f"Generate y_k (Regime 1)"):
        val = np.random.normal(u1_prior, gen_sigma)
        st.session_state.y_history.append((val, 1))

with c2:
    if st.button(f"Generate y_k (Regime 2)"):
        val = np.random.normal(u2_prior, gen_sigma)
        st.session_state.y_history.append((val, 2))

with c3:
    if st.button("Reset Experiment"):
        st.session_state.y_history = []

# --- Calculation Logic ---
if len(st.session_state.y_history) > 1:
    y_vals = np.array([item[0] for item in st.session_state.y_history])
    regimes = np.array([item[1] for item in st.session_state.y_history])
    k_total = len(y_vals)
    
    # Possible change points m from 1 to 99
    ms = np.arange(1, 100)
    log_post = np.zeros(len(ms))
    
    for i, m in enumerate(ms):
        if m >= k_total:
            # Model hypothesis: Change hasn't happened yet
            log_post[i] = np.sum(norm.logpdf(y_vals, loc=u1_prior, scale=np.sqrt(sigma_model)))
        else:
            # Model hypothesis: Change occurred at index m
            ll1 = np.sum(norm.logpdf(y_vals[:m], loc=u1_prior, scale=np.sqrt(sigma_model)))
            ll2 = np.sum(norm.logpdf(y_vals[m:], loc=u2_prior, scale=np.sqrt(sigma_model)))
            log_post[i] = ll1 + ll2
    
    # Normalization
    probs = np.exp(log_post - np.max(log_post))
    probs /= np.sum(probs)
    
    # Calculate Posterior Mean of t0
    # Expected value E[t0] = sum(m * P(t0=m))
    post_mean_t0 = np.sum(ms * probs)

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Realized Values of y_k
    # Splitting for different colors
    idx_r1 = np.where(regimes == 1)[0]
    idx_r2 = np.where(regimes == 2)[0]
    
    # Plot the full line connecting all points
    ax1.plot(range(len(y_vals)), y_vals, color='gray', linestyle='--', alpha=0.3, label="Path")
    
    # Plot segments
    if len(idx_r1) > 0:
        ax1.scatter(idx_r1, y_vals[idx_r1], color='#008080', label="Regime 1", zorder=5)
        # Connect Regime 1 points
        ax1.plot(idx_r1, y_vals[idx_r1], color='#008080', alpha=0.6)
        
    if len(idx_r2) > 0:
        ax1.scatter(idx_r2, y_vals[idx_r2], color='#FF7F50', label="Regime 2", zorder=5)
        # Connect Regime 2 points
        ax1.plot(idx_r2, y_vals[idx_r2], color='#FF7F50', alpha=0.6)

    ax1.set_title(f"Realized Observations $y_k$ (k={k_total})", fontsize=14)
    ax1.set_xlabel("Time Index (k)")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Posterior of t0
    ax2.bar(ms, probs, color='teal', alpha=0.5, width=0.8, label="Posterior $P(t_0 | y)$")
    
    # Highlight the Mean
    ax2.axvline(post_mean_t0, color='red', linestyle='--', linewidth=2, 
                label=f"Posterior Mean (E[t0] ≈ {post_mean_t0:.2f})")
    
    ax2.set_title("Posterior Probability Density of Change-Point $t_0$", fontsize=14)
    ax2.set_xlabel("Hypothesized Change-Point ($m$)")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(0, 100)
    ax2.legend()
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    st.pyplot(fig)
    
    # Metrics display
    st.write(f"**Current Statistical Summary:**")
    st.write(f"The model estimates the most likely change-point is near index **{np.argmax(probs)+1}**, "
             f"with a posterior mean of **{post_mean_t0:.2f}**.")

else:
    st.info("Start by clicking 'Generate y_k' buttons to build your dataset.")
