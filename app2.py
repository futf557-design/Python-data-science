import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import requests

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Rainfall Distribution Analyzer",
    layout="wide",
    page_icon="🌧️"
)

# ----------------------------------------------------------
# MATPLOTLIB STYLE HELPER
# ----------------------------------------------------------
def style_chart(ax, fig):
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0a1628")
    ax.tick_params(colors="#a8d8ea", labelsize=9)
    ax.xaxis.label.set_color("#a8d8ea")
    ax.yaxis.label.set_color("#a8d8ea")
    ax.title.set_color("#e0f4ff")
    ax.spines["bottom"].set_color("#1e3a5f")
    ax.spines["left"].set_color("#1e3a5f")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#1a3050", linewidth=0.6, linestyle="--", alpha=0.8)
    ax.set_axisbelow(True)

# ----------------------------------------------------------
# SIDEBAR — Project Info + Chart Settings
# ----------------------------------------------------------
st.sidebar.title("🌧️ Rainfall Analyzer")
st.sidebar.caption("Tamil Nadu · Data Science Project")
st.sidebar.divider()

st.sidebar.subheader("Project Info")
st.sidebar.markdown("**Course:** Python for Data Science")
st.sidebar.markdown("**Team Members:** RISHI ARAVINDH L , SAKTHI VIGNESH S , SRINIKESH P")
st.sidebar.markdown("**Roll No:** A2544 , A2548 , A2554")
st.sidebar.markdown("**Dept:** 1st Year AI & DS")
st.sidebar.markdown("**College:** K. Ramakrishna College of Engineering")
st.sidebar.divider()

# ----------------------------------------------------------
# MAIN TITLE
# ----------------------------------------------------------
st.title("🌧️ Rainfall Distribution Analyzer")
st.caption("Interactive data exploration and AI-powered predictions for Tamil Nadu rainfall patterns")
st.divider()

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
file_path = "tamilnadu_rainfall.csv"

if not os.path.exists(file_path):
    st.error(f"⚠️ File `{file_path}` not found. Please place it in the same folder as `app.py`.")
    st.stop()

df = pd.read_csv(file_path)

# ----------------------------------------------------------
# DATA EDITOR
# ----------------------------------------------------------
st.subheader("📋 Dataset Editor")
st.info("💡 Click any cell to edit  •  Scroll to the bottom to add new rows  •  Select a row + Delete key to remove it")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="editor")
st.divider()

# ----------------------------------------------------------
# AXIS SELECTION (sidebar)
# ----------------------------------------------------------
numeric_cols = edited_df.select_dtypes(include=["float64", "int64"]).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found in the dataset. Please check your CSV file.")
    st.stop()

st.sidebar.subheader("Chart Settings")
x_axis = st.sidebar.selectbox("X-axis (Category / Year / Month):", edited_df.columns)
y_axis = st.sidebar.selectbox("Y-axis (Rainfall Value):", numeric_cols)

# ----------------------------------------------------------
# SUMMARY METRICS
# ----------------------------------------------------------
st.subheader("📊 Summary Statistics")

col_data = edited_df[y_axis].dropna()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Records",  len(edited_df))
m2.metric("Mean",           f"{col_data.mean():.1f} mm")
m3.metric("Maximum",        f"{col_data.max():.1f} mm")
m4.metric("Minimum",        f"{col_data.min():.1f} mm")
m5.metric("Std Deviation",  f"{col_data.std():.1f} mm")

st.divider()

# ----------------------------------------------------------
# VISUALIZATIONS
# ----------------------------------------------------------
st.subheader("📈 Live Visualizations")

tab1, tab2, tab3 = st.tabs(["🔵 Scatter Plot", "📊 Histogram + KDE", "📉 Regression Trend"])

# ── TAB 1: SCATTER PLOT ───────────────────────────────────
with tab1:
    st.write(f"**Scatter Plot — {y_axis} vs {x_axis}**")

    scatter_data = edited_df.dropna(subset=[x_axis, y_axis])

    fig, ax = plt.subplots(figsize=(11, 5))
    style_chart(ax, fig)

    if pd.api.types.is_numeric_dtype(scatter_data[x_axis]):
        sc = ax.scatter(
            scatter_data[x_axis],
            scatter_data[y_axis],
            c=scatter_data[y_axis].values,
            cmap="cool",
            s=75,
            alpha=0.85,
            edgecolors="#0a1628",
            linewidths=0.5
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.ax.tick_params(colors="#a8d8ea", labelsize=8)
        cbar.set_label(f"{y_axis} (mm)", color="#a8d8ea", fontsize=9)
    else:
        cats  = scatter_data[x_axis].astype("category")
        codes = cats.cat.codes
        sc = ax.scatter(
            codes,
            scatter_data[y_axis],
            c=scatter_data[y_axis].values,
            cmap="cool",
            s=75,
            alpha=0.85,
            edgecolors="#0a1628",
            linewidths=0.5
        )
        ax.set_xticks(range(len(cats.cat.categories)))
        ax.set_xticklabels(cats.cat.categories, rotation=40, ha="right", fontsize=8)

    ax.set_xlabel(x_axis, fontsize=10)
    ax.set_ylabel(f"{y_axis} (mm)", fontsize=10)
    ax.set_title(f"Scatter Plot — {y_axis} vs {x_axis}", fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── TAB 2: HISTOGRAM + KDE ────────────────────────────────
with tab2:
    st.write(f"**Histogram & KDE Density Curve — {y_axis}**")

    hist_data = edited_df[y_axis].dropna()
    bins      = min(20, max(5, len(hist_data) // 4))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1b2a")

    # Left: Histogram
    ax1 = axes[0]
    style_chart(ax1, fig)

    n, _, patches = ax1.hist(hist_data, bins=bins, edgecolor="#0a1628", linewidth=0.5, alpha=0.92)
    norm = plt.Normalize(n.min(), n.max())
    for cnt, patch in zip(n, patches):
        patch.set_facecolor(plt.cm.cool(norm(cnt)))

    ax1.axvline(hist_data.mean(),   color="#ffa040", lw=1.8, ls="--", label=f"Mean: {hist_data.mean():.1f} mm")
    ax1.axvline(hist_data.median(), color="#ff5f5f", lw=1.8, ls=":",  label=f"Median: {hist_data.median():.1f} mm")
    ax1.set_xlabel(f"{y_axis} (mm)", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.set_title(f"Histogram — {y_axis}", fontsize=11, fontweight="bold", pad=12)
    ax1.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#a8d8ea", fontsize=8)

    # Right: KDE
    ax2 = axes[1]
    style_chart(ax2, fig)

    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(hist_data)
        xs  = np.linspace(hist_data.min(), hist_data.max(), 250)
        ax2.fill_between(xs, kde(xs), alpha=0.28, color="#3ab5ff")
        ax2.plot(xs, kde(xs), color="#3ab5ff", linewidth=2.5)
        ax2.axvline(hist_data.mean(),   color="#ffa040", lw=1.8, ls="--", label=f"Mean: {hist_data.mean():.1f} mm")
        ax2.axvline(hist_data.median(), color="#ff5f5f", lw=1.8, ls=":",  label=f"Median: {hist_data.median():.1f} mm")
        ax2.legend(facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#a8d8ea", fontsize=8)
    except ImportError:
        ax2.hist(hist_data, bins=bins, color="#3ab5ff", edgecolor="#0d1b2a", alpha=0.7)
        st.caption("Install `scipy` for the KDE curve: `pip install scipy`")

    ax2.set_xlabel(f"{y_axis} (mm)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("KDE — Density Curve", fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── TAB 3: REGRESSION TREND ──────────────────────────────
with tab3:
    if pd.api.types.is_numeric_dtype(edited_df[x_axis]):
        ml_data = edited_df.dropna(subset=[x_axis, y_axis])
        X       = ml_data[[x_axis]].values
        y_vals  = ml_data[y_axis].values

        model   = LinearRegression()
        model.fit(X, y_vals)
        preds   = model.predict(X)
        r2      = model.score(X, y_vals)

        fig, ax = plt.subplots(figsize=(11, 5))
        style_chart(ax, fig)

        residuals = y_vals - preds
        for i in range(len(X)):
            col = "#3aff9f" if residuals[i] >= 0 else "#ff6b6b"
            ax.plot([X[i][0], X[i][0]], [preds[i], y_vals[i]],
                    color=col, alpha=0.35, linewidth=1)

        ax.scatter(ml_data[x_axis], y_vals,
                   color="#3ab5ff", s=65, zorder=5, alpha=0.85,
                   edgecolors="#0a1628", linewidths=0.5)
        ax.plot(ml_data[x_axis], preds,
                color="#ff9f40", linewidth=2.5, zorder=6)

        ax.legend(
            handles=[
                mpatches.Patch(color="#3ab5ff", label="Actual Data"),
                mpatches.Patch(color="#ff9f40", label="Regression Line"),
                mpatches.Patch(color="#3aff9f", label="Above Trend"),
                mpatches.Patch(color="#ff6b6b", label="Below Trend"),
            ],
            facecolor="#0d1b2a", edgecolor="#1e3a5f", labelcolor="#a8d8ea", fontsize=8
        )
        ax.set_xlabel(x_axis, fontsize=10)
        ax.set_ylabel(f"{y_axis} (mm)", fontsize=10)
        ax.set_title(
            f"Linear Regression  |  R² = {r2:.4f}  |  Slope = {model.coef_[0]:.3f}  |  Intercept = {model.intercept_:.2f}",
            fontsize=11, fontweight="bold", pad=14
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.write("**Model Summary**")
        r1, r2_col, r3 = st.columns(3)
        r1.metric("R² Score",   f"{r2:.4f}")
        r2_col.metric("Slope",  f"{model.coef_[0]:.4f}")
        r3.metric("Intercept",  f"{model.intercept_:.4f}")
    else:
        st.warning(f"X-axis `{x_axis}` must be numeric for regression. Please select a numeric column from the sidebar.")

st.divider()

# ----------------------------------------------------------
# AI LIVE PREDICTION — Claude API
# ----------------------------------------------------------
st.subheader("🤖 AI-Powered Live Prediction")
st.caption("The AI reads your live edited data every time you click Analyze — real-time insights powered by Groq.")

# Build live data context from the edited DataFrame
summary_str = edited_df.describe().round(2).to_string()
recent_str  = edited_df.tail(10).to_string(index=False)

default_prompt = (
    "Based on the Tamil Nadu rainfall dataset below, provide:\n"
    "1. Key observations about the rainfall distribution and seasonal patterns\n"
    "2. Any anomalies or extreme values worth noting\n"
    "3. A short trend forecast based on the data\n"
    "4. One actionable recommendation for water resource management\n\n"
    f"--- Dataset Statistics ---\n{summary_str}\n\n"
    f"--- Recent 10 Records ---\n{recent_str}"
)

user_question = st.text_area(
    "Your analysis request (auto-filled with your live data context — edit freely):",
    value=default_prompt,
    height=200,
    key="ai_prompt"
)

col_a, col_b, _ = st.columns([1.5, 1, 5])

with col_a:
    run_ai = st.button("⚡ Analyze Now", use_container_width=True)

with col_b:
    if st.button("🔄 Clear", use_container_width=True):
        st.session_state["ai_response"] = ""
        st.rerun()

if run_ai:
    if not user_question.strip():
        st.warning("Please enter a question or keep the default prompt.")
    else:
        # Get Groq API key from environment variable or Streamlit secrets
        groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("⚠️ GROQ_API_KEY is not set. Please add it as an environment variable or Streamlit secret and restart.")
            st.stop()
        
        with st.spinner("Connecting to Groq AI — analyzing your rainfall data..."):
            try:
                model_name = os.getenv("GROQ_MODEL") or st.secrets.get("GROQ_MODEL") or "llama-3.1-8b-instant"
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {groq_api_key}"
                    },
                    json={
                        "model": model_name,
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a data science expert specializing in rainfall analysis and climate patterns for Tamil Nadu, India. Provide clear, concise, and actionable insights. Use numbered points for readability."
                            },
                            {
                                "role": "user",
                                "content": user_question
                            }
                        ]
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    ai_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    st.session_state["ai_response"] = ai_text
                else:
                    st.session_state["ai_response"] = (
                        f"API Error {response.status_code}:\n{response.text}"
                    )

            except requests.exceptions.Timeout:
                st.session_state["ai_response"] = "Request timed out. Please try again."
            except Exception as e:
                st.session_state["ai_response"] = f"Error: {str(e)}"

if st.session_state.get("ai_response"):
    st.success("AI analysis complete!")
    st.subheader("AI Insight")
    st.info(st.session_state["ai_response"])

st.divider()
st.caption("🌧️ Rainfall Distribution Analyzer · K. Ramakrishna College of Engineering · AI & DS · 2025")
