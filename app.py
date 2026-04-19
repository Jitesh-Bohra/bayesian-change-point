import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- Page Config ---
st.set_page_config(page_title="Bayesian Change-Point Detection", layout="wide")

# Initialize Session State
if 'y_vals' not in st.session_state:
    st.session_state.y_vals = []
if 'regime_labels' not in st.session_state:
    st.session_state.regime_labels = []

# --- Header ---
st.title("📈 Bayesian Change-Point Detection Dashboard")
st.markdown("""
### How to use
1. **Model Priors**: Set what the model *believes* about the two regimes.
2. **Actual Values**: Set the *ground truth* for data generation.
3. **Generate**: Click buttons to add points. The top plot shows the sequence; the bottom shows the likelihood of the change-point $t_0$.
""")

# --- Sidebar: Prior Parameters ---
st.sidebar.header("1. Model Priors (Beliefs)")
u1_p = st.sidebar.slider("Prior Mean μ1", -10.0, 10.0, 5.0)
u2_p = st.sidebar.slider("Prior Mean μ2", -10.0, 10.0, 10.0)
sig0_p = st.sidebar.slider("Data Variance σ0²", 0.1, 10.0, 1.0)
sig1_p = st.sidebar.slider("Prior Variance σ1²", 0.1, 10.0, 1.0)
sig2_p = st.sidebar.slider("Prior Variance σ2²", 0.1, 10.0, 1.0)

st.sidebar.markdown("---")

# --- Sidebar: Actual Generation Values ---
st.sidebar.header("2. Ground Truth (Actuals)")
u1_a = st.sidebar.slider("Actual μ1", -10.0, 10.0, 5.0)
u2_a = st.sidebar.slider("Actual μ2", -10.0, 10.0, 10.0)
# Updated slider for Variance instead of Sigma
gen_sig_sq = st.sidebar.slider("Actual σ²", 0.1, 10.0, 1.0)

# --- Data Generation ---
st.write("### 1. Data Generation")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Add Regime 1 Point"):
        # We draw using sqrt of the variance slider
        st.session_state.y_vals.append(np.random.normal(u1_a, np.sqrt(gen_sig_sq)))
        st.session_state.regime_labels.append(1)
with c2:
    if st.button("Add Regime 2 Point"):
        st.session_state.y_vals.append(np.random.normal(u2_a, np.sqrt(gen_sig_sq)))
        st.session_state.regime_labels.append(2)
with c3:
    if st.button("Reset All Data"):
        st.session_state.y_vals = []
        st.session_state.regime_labels = []

# --- Bayesian Logic ---
y = np.array(st.session_state.y_vals)
regimes = np.array(st.session_state.regime_labels)
n = len(y)

if n > 1:
    # We use Marginal Likelihood: y ~ N(Prior_Mean, Data_Var + Prior_Var)
    # This incorporates your sigma1^2 and sigma2^2 sliders
    ms = np.arange(1, 100)
    log_l = np.zeros(len(ms))
    
    # Combined variances for the predictive distributions
    var1 = sig0_p + sig1_p
    var2 = sig0_p + sig2_p
    
    for i, m in enumerate(ms):
        if m >= n:
            log_l[i] = np.sum(norm.logpdf(y, loc=u1_p, scale=np.sqrt(var1)))
        else:
            ll1 = np.sum(norm.logpdf(y[:m], loc=u1_p, scale=np.sqrt(var1)))
            ll2 = np.sum(norm.logpdf(y[m:], loc=u2_p, scale=np.sqrt(var2)))
            log_l[i] = ll1 + ll2
    
    probs = np.exp(log_l - np.max(log_l))
    probs /= np.sum(probs)
    post_mean = np.sum(ms * probs)

    # --- Plots ---
    st.write("### 2. Bayesian Analysis")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: y_k with separate regime colors
    idx1 = np.where(regimes == 1)[0]
    idx2 = np.where(regimes == 2)[0]
    
    ax1.plot(range(n), y, color='lightgray', linestyle='--', alpha=0.4, label="Sequence Path")
    if len(idx1) > 0:
        ax1.scatter(idx1, y[idx1], color='#00d1b2', s=50, label="Regime 1 Samples", zorder=3)
        ax1.plot(idx1, y[idx1], color='#00d1b2', alpha=0.5)
    if len(idx2) > 0:
        ax1.scatter(idx2, y[idx2], color='#ff3860', s=50, label="Regime 2 Samples", zorder=3)
        ax1.plot(idx2, y[idx2], color='#ff3860', alpha=0.5)
        
    ax1.set_title(f"Realized Observations (Total: {n})", fontsize=14)
    ax1.set_ylabel("Value $y_k$")
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend()
    
    # Plot 2: Posterior of t0
    ax2.bar(ms, probs, color='teal', alpha=0.6, width=1.0, label="Posterior $P(t_0 | y)$")
    ax2.axvline(post_mean, color='red', linestyle='--', linewidth=2, label=f"Posterior Mean: {post_mean:.2f}")
    ax2.set_title("Posterior Density of Change-Point $t_0$", fontsize=14)
    ax2.set_xlabel("Time Step (m)")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    st.pyplot(fig)
else:
    st.info("Awaiting more data. Please generate at least 2 points.")
