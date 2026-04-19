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
1. **Configure Model Priors**: Use the sidebar to set the parameters the model *assumes* to be true.
2. **Set Generation Parameters**: Adjust the **Actual** values. These are the "ground truth" values used to sample new data points.
3. **Generate Data**: Click the buttons below to add points. Switch from Regime 1 to Regime 2 to create a change-point.
4. **Observe Detection**: The top plot shows your data colored by regime. The bottom plot shows the **Posterior Probability of $t_0$**.
5. **Analyze the Mean**: The red dashed line represents the **Posterior Mean (Expected Value)** of the change-point index.
""")

# --- Sidebar: Parameter Sliders ---
st.sidebar.header("1. Model Priors (Beliefs)")
st.sidebar.markdown("These values are used in the Bayesian likelihood calculation.")
u1_prior = st.sidebar.slider("Prior Mean μ1", -10.0, 10.0, 5.0)
u2_prior = st.sidebar.slider("Prior Mean μ2", -10.0, 10.0, 10.0)
sigma_model = st.sidebar.slider("Model Variance σ²", 0.1, 10.0, 1.0)

st.sidebar.markdown("---")

st.sidebar.header("2. Actual Values (Ground Truth)")
st.sidebar.markdown("Values used only for generating new data points.")
u1_actual = st.sidebar.slider("Actual μ1", -10.0, 10.0, 5.0)
u2_actual = st.sidebar.slider("Actual μ2", -10.0, 10.0, 10.0)
gen_sigma = st.sidebar.slider("Generation σ", 0.1, 5.0, 1.0)

# --- Data Generation Buttons ---
st.write("### Data Generation")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button(f"Generate y_k (Regime 1: μ={u1_actual})"):
        val = np.random.normal(u1_actual, gen_sigma)
        st.session_state.y_history.append((val, 1))

with c2:
    if st.button(f"Generate y_k (Regime 2: μ={u2_actual})"):
        val = np.random.normal(u2_actual, gen_sigma)
        st.session_state.y_history.append((val, 2))

with c3:
    if st.button("Reset Experiment"):
        st.session_state.y_history = []

# --- Calculation Logic ---
if len(st.session_state.y_history) > 1:
    y_vals = np.array([item[0] for item in st.session_state.y_history])
    regimes = np.array([item[1] for item in st.session_state.y_history])
    k_total = len(y_vals)
    
    # Possible change points m from 1 to 99 (Prior: Discrete Uniform)
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
    
    # Normalization using log-sum-exp trick
    probs = np.exp(log_post - np.max(log_post))
    probs /= np.sum(probs)
    
    # Calculate Posterior Mean of t0
    post_mean_t0 = np.sum(ms * probs)

    # --- Plots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Realized Values of y_k
    idx_r1 = np.where(regimes == 1)[0]
    idx_r2 = np.where(regimes == 2)[0]
    
    # Background path
    ax1.plot(range(len(y_vals)), y_vals, color='gray', linestyle='--', alpha=0.3, label="Path")
    
    # Regime 1 points and lines
    if len(idx_r1) > 0:
        ax1.scatter(idx_r1, y_vals[idx_r1], color='#008080', label=f"Regime 1 (μ={u1_actual})", zorder=5)
        ax1.plot(idx_r1, y_vals[idx_r1], color='#008080', alpha=0.6)
        
    # Regime 2 points and lines
    if len(idx_r2) > 0:
        ax1.scatter(idx_r2, y_vals[idx_r2], color='#FF7F50', label=f"Regime 2 (μ={u2_actual})", zorder=5)
        ax1.plot(idx_r2, y_vals[idx_r2], color='#FF7F50', alpha=0.6)

    ax1.set_title(f"Realized Observations $y_k$ (Total Points: {k_total})", fontsize=14)
    ax1.set_xlabel("Time Index (k)")
    ax1.set_ylabel("Value")
    ax1.legend(loc='upper left')
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
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)
    
    # Metrics display
    st.write(f"**Statistical Summary:**")
    st.write(f"The model's current best estimate for the change-point is index **{np.argmax(probs)+1}**. "
             f"The posterior mean is **{post_mean_t0:.2f}**.")

else:
    st.info("Awaiting data. Use the 'Generate y_k' buttons to build your sequence.")
