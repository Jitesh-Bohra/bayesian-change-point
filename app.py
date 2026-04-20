import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, invgamma

# --- Page Config ---
st.set_page_config(page_title="Bayesian Change-Point Analysis", layout="wide")

# Initialize Session State
if 'y_vals' not in st.session_state:
    st.session_state.y_vals = []
if 'regime_labels' not in st.session_state:
    st.session_state.regime_labels = []

# --- New User Instructions ---
st.title("🔍 Full Bayesian Posterior Analysis")

st.markdown("""
### 🚀 Quick Start Guide
1. **Configure the Sidebar**: 
    - **Model Priors**: Set what the model *expects* (e.g., if you think the change is large, set $\mu_1$ and $\mu_2$ far apart).
    - **Ground Truth**: These values control the *actual* data being generated.
2. **Feed the Model**: Click **'Generate Regime 1 Sample'** to start the stream. After a few points, click **'Generate Regime 2 Sample'** to simulate a structural break.
3. **Analyze**: Monitor the metrics and plots to see how the model updates its belief about the change-point and regime parameters.
---
""")

# --- Sidebar: Parameters (Synchronized Defaults) ---
st.sidebar.header("1. Model Priors")
u1_p = st.sidebar.slider("Prior Mean μ1", -10.0, 10.0, -10.0)
u2_p = st.sidebar.slider("Prior Mean μ2", -10.0, 10.0, 10.0)
sig0_p = st.sidebar.slider("Prior σ0² (Data Var)", 0.1, 10.0, 1.0)
sig1_p = st.sidebar.slider("Prior σ1² (μ1 Var)", 0.1, 10.0, 10.0)
sig2_p = st.sidebar.slider("Prior σ2² (μ2 Var)", 0.1, 10.0, 10.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Ground Truth")
u1_a = st.sidebar.slider("Actual μ1", -10.0, 10.0, -0.06)
u2_a = st.sidebar.slider("Actual μ2", -10.0, 10.0, 6.01)
actual_var = st.sidebar.slider("Actual σ²", 0.1, 10.0, 4.96)

# --- Data Generation ---
st.write("### 1. Data Stream Control")
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
    lambda_ratio = sigma2 / tau2
    sum_y = np.sum(data)
    sum_y2 = np.sum(data**2)
    # Numerical stability terms
    t1 = - (m / 2) * np.log(2 * np.pi * sigma2)
    t2 = 0.5 * np.log(sigma2 / (m * tau2 + sigma2))
    # Core posterior exponent
    t3 = - (1 / (2 * sigma2)) * (sum_y2 + lambda_ratio * (mu0**2) - ((sum_y + lambda_ratio * mu0)**2) / (m + lambda_ratio))
    return t1 + t2 + t3

y = np.array(st.session_state.y_vals)
regs = np.array(st.session_state.regime_labels)
n = len(y)

if n > 1:
    # We evaluate t0 across a fixed horizon for visualization
    ms = np.arange(1, 101) 
    log_post_t0 = np.zeros(len(ms))
    
    mu1_means, mu1_vars = [], []
    mu2_means, mu2_vars = [], []
    
    for i, m in enumerate(ms):
        # 1. log-posterior for t0
        if m >= n:
            log_post_t0[i] = log_marginal_likelihood(y, u1_p, sig1_p, sig0_p)
        else:
            l1 = log_marginal_likelihood(y[:m], u1_p, sig1_p, sig0_p)
            l2 = log_marginal_likelihood(y[m:], u2_p, sig2_p, sig0_p)
            log_post_t0[i] = l1 + l2
        
        # 2. Analytical Mixture Components for means
        # Regime 1 parameters
        d1 = y if m >= n else y[:m]
        prec1 = (1/sig1_p) + (len(d1)/sig0_p)
        mu1_vars.append(1/prec1)
        mu1_means.append((1/prec1) * ((u1_p/sig1_p) + (np.sum(d1)/sig0_p)))
        
        # Regime 2 parameters
        d2 = y[m:] if m < n else np.array([])
        if len(d2) > 0:
            prec2 = (1/sig2_p) + (len(d2)/sig0_p)
            mu2_vars.append(1/prec2)
            mu2_means.append((1/prec2) * ((u2_p/sig2_p) + (np.sum(d2)/sig0_p)))
        else:
            mu2_means.append(u2_p)
            mu2_vars.append(sig2_p)

    # Normalize weights using Log-Sum-Exp for stability
    weights = np.exp(log_post_t0 - np.max(log_post_t0))
    weights /= np.sum(weights)

    # Calculate Posterior Mean and Mode of t0
    post_mean_t0 = np.sum(ms * weights)
    post_mode_t0 = ms[np.argmax(weights)]

    # --- Metrics Display ---
    st.write("#### Posterior Estimates for $t_0$")
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Posterior Mean ($E[t_0|y]$)", f"{post_mean_t0:.2f}")
    m_col2.metric("Posterior Mode (MAP)", f"{int(post_mode_t0)}")
    
    # --- Rearranged Visualizations ---
    st.write("### 2. Analysis Results")
    try:
        fig, axes = plt.subplots(4, 1, figsize=(12, 20))
        
        # Plot 1: Data Sequence
        idx1, idx2 = np.where(regs == 1)[0], np.where(regs == 2)[0]
        axes[0].plot(range(n), y, color='lightgray', linestyle='--', alpha=0.4)
        if len(idx1) > 0:
            axes[0].scatter(idx1, y[idx1], color='#00d1b2', label="Regime 1 Samples", zorder=3)
            axes[0].plot(idx1, y[idx1], color='#00d1b2', alpha=0.3)
        if len(idx2) > 0:
            axes[0].scatter(idx2, y[idx2], color='#ff3860', label="Regime 2 Samples", zorder=3)
            axes[0].plot(idx2, y[idx2], color='#ff3860', alpha=0.3)
        axes[0].set_title("1. Observed Data Sequence", fontweight='bold', loc='left')
        axes[0].set_ylabel("Value $y_k$")
        axes[0].legend(loc='upper left')

        # Plot 2: Posterior Density of t0
        axes[1].bar(ms, weights, color='teal', alpha=0.6, width=0.8)
        axes[1].axvline(post_mean_t0, color='red', linestyle='--', label=f"Mean: {post_mean_t0:.1f}")
        axes[1].set_title("2. Posterior Density of Change-Point $t_0$", fontweight='bold', loc='left')
        axes[1].set_xlabel("Time Step (m)")
        axes[1].set_ylabel("Probability")
        axes[1].set_xlim(0, 105)
        axes[1].legend()

        # Plot 3: Posterior Density of u1 and u2 (Mixture Distribution)
        mu_grid = np.linspace(-15, 15, 500)
        pdf_mu1, pdf_mu2 = np.zeros_like(mu_grid), np.zeros_like(mu_grid)
        for i in range(len(ms)):
            pdf_mu1 += weights[i] * norm.pdf(mu_grid, mu1_means[i], np.sqrt(mu1_vars[i]))
            pdf_mu2 += weights[i] * norm.pdf(mu_grid, mu2_means[i], np.sqrt(mu2_vars[i]))
        axes[2].plot(mu_grid, pdf_mu1, label="$\mu_1$ Posterior", color='#00d1b2', lw=2.5)
        axes[2].plot(mu_grid, pdf_mu2, label="$\mu_2$ Posterior", color='#ff3860', lw=2.5)
        axes[2].set_title("3. Posterior Densities of Regime Means $\mu_1, \mu_2$", fontweight='bold', loc='left')
        axes[2].set_xlabel("Mean Value")
        axes[2].legend()

        # Plot 4: Posterior Density of sigma^2 (Mixture Inverse Gamma)
        sig_grid = np.linspace(0.01, 15, 500)
        pdf_sig = np.zeros_like(sig_grid)
        for i, m in enumerate(ms):
            d1 = y if m >= n else y[:m]
            d2 = y[m:] if m < n else np.array([])
            # Sum of Squares with epsilon to prevent scale=0
            ss = np.sum((d1 - mu1_means[i])**2) + (np.sum((d2 - mu2_means[i])**2) if len(d2)>0 else 0)
            ss = max(ss, 1e-6) 
            pdf_sig += weights[i] * invgamma.pdf(sig_grid, a=n/2, scale=ss/2)
            
        axes[3].plot(sig_grid, pdf_sig, color='purple', lw=2.5)
        axes[3].set_title("4. Posterior Density of Data Variance $\sigma^2$", fontweight='bold', loc='left')
        axes[3].set_xlabel("$\sigma^2$ Value")

        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Visualization Error: {e}. Try adding more data points to stabilize the variances.")
else:
    st.info("💡 **Awaiting Data:** Generate at least 2 points to start the Bayesian analysis.")
