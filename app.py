import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Config & Styling ---
st.set_page_config(page_title="Bayesian Change-Point Detection", layout="wide")

# Initialize Session State
if 'y_vals' not in st.session_state:
    st.session_state.y_vals = []
if 'regime_labels' not in st.session_state:
    st.session_state.regime_labels = []

# --- Header & User Guide ---
st.title("📊 Robust Bayesian Change-Point Detection")
with st.expander("📖 How to use this app & Mathematical Verification", expanded=False):
    st.write("""
    This app performs **Sequential Bayesian Inference** to detect a shift in the process mean.
    - **Step 1:** Set your **Model Priors** in the sidebar. These represent your 'beliefs' before seeing data.
    - **Step 2:** Set the **Actual Ground Truth** for data generation.
    - **Step 3:** Add data points. The model calculates the **Marginal Likelihood** for every possible change-point $t_0$.
    - **Math Note:** We integrate out the unknown means $\mu_1, \mu_2$ using Normal priors, resulting in a predictive distribution that accounts for both data noise ($\sigma_0^2$) and prior uncertainty ($\sigma_1^2, \sigma_2^2$).
    """)

# --- Sidebar: Parameters ---
st.sidebar.header("1. Model Priors (Math Input)")
u1_p = st.sidebar.slider("Prior Mean μ1", -10.0, 10.0, 5.0)
u2_p = st.sidebar.slider("Prior Mean μ2", -10.0, 10.0, 10.0)
sig0_p = st.sidebar.slider("Data Variance σ0²", 0.1, 10.0, 1.0)
sig1_p = st.sidebar.slider("Prior Variance σ1²", 0.1, 10.0, 1.0)
sig2_p = st.sidebar.slider("Prior Variance σ2²", 0.1, 10.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Ground Truth (Data Gen)")
u1_a = st.sidebar.slider("Actual μ1", -10.0, 10.0, 5.0)
u2_a = st.sidebar.slider("Actual μ2", -10.0, 10.0, 10.0)
# Updated slider to explicitly use Variance as requested
actual_var = st.sidebar.slider("Actual σ²", 0.1, 10.0, 1.0)

# --- Data Generation Buttons ---
st.write("### 1. Real-Time Data Stream")
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Generate Regime 1 Sample"):
        st.session_state.y_vals.append(np.random.normal(u1_a, np.sqrt(actual_var)))
        st.session_state.regime_labels.append(1)
with c2:
    if st.button("Generate Regime 2 Sample"):
        st.session_state.y_vals.append(np.random.normal(u2_a, np.sqrt(actual_var)))
        st.session_state.regime_labels.append(2)
with c3:
    if st.button("Reset Session"):
        st.session_state.y_vals = []
        st.session_state.regime_labels = []

# --- Mathematical Engine ---
def log_marginal_likelihood(data, mu0, tau2, sigma2):
    m = len(data)
    if m == 0: return 0
    sum_y = np.sum(data)
    sum_y2 = np.sum(data**2)
    # Ratios for simplified calculation
    lambda_ratio = sigma2 / tau2
    
    term1 = - (m / 2) * np.log(2 * np.pi * sigma2)
    term2 = 0.5 * np.log(sigma2 / (m * tau2 + sigma2))
    term3 = - (1 / (2 * sigma2)) * (sum_y2 + lambda_ratio * (mu0**2) - ((sum_y + lambda_ratio * mu0)**2) / (m + lambda_ratio))
    return term1 + term2 + term3

y = np.array(st.session_state.y_vals)
regs = np.array(st.session_state.regime_labels)
n = len(y)

if n > 1:
    ms = np.arange(1, 100) # Hypothesized t0
    log_post = np.zeros(len(ms))
    
    for i, m in enumerate(ms):
        if m >= n:
            # Entire sequence belongs to Regime 1
            log_post[i] = log_marginal_likelihood(y, u1_p, sig1_p, sig0_p)
        else:
            # Sequence splits at m
            l1 = log_marginal_likelihood(y[:m], u1_p, sig1_p, sig0_p)
            l2 = log_marginal_likelihood(y[m:], u2_p, sig2_p, sig0_p)
            log_post[i] = l1 + l2
            
    # Convert log-space to probabilities (Log-Sum-Exp Trick)
    probs = np.exp(log_post - np.max(log_post))
    probs /= np.sum(probs)
    post_mean_t0 = np.sum(ms * probs)

    # --- Visualization ---
    st.write("### 2. Bayesian Posterior Analysis")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1.2]})
    
    # Plot 1: Formatted Data Sequence
    idx1 = np.where(regs == 1)[0]
    idx2 = np.where(regs == 2)[0]
    
    ax1.plot(range(n), y, color='lightgray', linestyle='--', alpha=0.4, label="Sequence Connection")
    if len(idx1) > 0:
        ax1.scatter(idx1, y[idx1], color='#00d1b2', s=60, edgecolors='black', label="Regime 1", zorder=3)
        ax1.plot(idx1, y[idx1], color='#00d1b2', alpha=0.6, linewidth=2)
    if len(idx2) > 0:
        ax1.scatter(idx2, y[idx2], color='#ff3860', s=60, edgecolors='black', label="Regime 2", zorder=3)
        ax1.plot(idx2, y[idx2], color='#ff3860', alpha=0.6, linewidth=2)
        
    ax1.set_title(f"Realized Observations (n={n})", loc='left', fontweight='bold')
    ax1.set_ylabel("Value $y_k$")
    ax1.legend(facecolor='white', framealpha=1)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Posterior of t0
    ax2.bar(ms, probs, color='teal', alpha=0.5, width=0.8, label="Posterior Density $P(t_0 | y)$")
    ax2.axvline(post_mean_t0, color='#e74c3c', linestyle='--', linewidth=2.5, 
                label=f"Posterior Mean: {post_mean_t0:.2f}")
    
    # Visual cues for the user
    ax2.fill_between(ms, 0, probs, color='teal', alpha=0.1)
    ax2.set_title("Probability Distribution of the Change-Point $t_0$", loc='left', fontweight='bold')
    ax2.set_xlabel("Time Step (m)")
    ax2.set_ylabel("Probability")
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, max(probs)*1.2 if max(probs)>0 else 1)
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    
    st.pyplot(fig)
else:
    st.info("💡 **Awaiting Data:** Generate at least 2 points to start the Bayesian engine.")
