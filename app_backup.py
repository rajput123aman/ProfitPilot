import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ProfitPilot | Order Intelligence",
    page_icon="ORDERS",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- CUSTOM THEME ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { --ink:#172746; --muted:#667794; --line:#e5ebf5; --blue:#4169f5; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: #f5f7fc; color: var(--ink); }
.block-container { max-width: 1600px; padding-top: 1.5rem; padding-bottom: 2.5rem; }
[data-testid="stHeader"] { background: rgba(245,247,252,.92); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#111b32 0%,#1b2d50 100%); }
[data-testid="stSidebar"] * { color:#eef3ff; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c8d4ec; }
h1 { color:#152545; font-weight:800 !important; letter-spacing:-1.2px; }
h2,h3 { color:#1b2d50; font-weight:700 !important; }
[data-testid="stMetric"] {
  background:linear-gradient(145deg,#fff,#fbfcff); border:1px solid var(--line);
  border-radius:16px; padding:17px 19px; box-shadow:0 5px 18px rgba(26,48,89,.045);
}
[data-testid="stMetricLabel"] { color:#687995; font-weight:600; }
[data-testid="stMetricValue"] { color:#172746; font-weight:800; }
[data-testid="stPlotlyChart"] {
  background:#fff; border:1px solid var(--line); border-radius:16px;
  padding:8px; box-shadow:0 4px 16px rgba(26,48,89,.035);
}
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:650; }
.stButton > button[kind="primary"] { background:#4169f5; border:0; color:white; }
div[data-testid="stExpander"] { background:white; border:1px solid var(--line); border-radius:12px; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color:#34496d; font-weight:600; }
hr { border-color:#e4eaf4; }
.hero {
  background:linear-gradient(115deg,#e8efff 0%,#fff 58%,#e5f8f2 100%);
  border:1px solid #dce6f8; border-radius:20px; padding:23px 26px; margin:8px 0 22px;
}
.hero-kicker { font-size:11px; font-weight:800; color:#4169f5; letter-spacing:1.6px; text-transform:uppercase; }
.hero-title { font-size:clamp(22px,2.5vw,31px); font-weight:800; color:#152545; margin:7px 0; letter-spacing:-.8px; }
.hero-sub { color:#5b6e8d; font-size:14px; line-height:1.7; max-width:850px; }
.pill { display:inline-block; padding:7px 11px; border-radius:999px; background:#e6f8ee; color:#18794e; font-size:11px; font-weight:800; }
.section-kicker { color:#7b89a1; text-transform:uppercase; font-size:11px; font-weight:800; letter-spacing:1.5px; margin:8px 0 9px; }
.insight {
  background:#fff; border:1px solid #e5ebf5; border-left:4px solid #5578f5;
  border-radius:12px; padding:13px 15px; margin:0 0 10px; color:#344967;
  font-size:13px; line-height:1.6;
}
.mini-note { color:#8390a7; font-size:12px; }
.footer { color:#8491a8; font-size:12px; text-align:center; padding:22px 0 3px; }
</style>
""", unsafe_allow_html=True)

# ---------------- SYNTHETIC MODEL ----------------
@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    n = 2400
    data = pd.DataFrame({
        "order_value": rng.integers(300, 5000, n),
        "is_cod": rng.integers(0, 2, n),
        "customer_return_rate": rng.uniform(0, 0.8, n),
        "distance_km": rng.integers(2, 1000, n),
        "previous_orders": rng.integers(0, 30, n),
    })
    score = (
        -2
        + 1.5 * data["is_cod"]
        + 2.5 * data["customer_return_rate"]
        + 0.001 * data["distance_km"]
        - 0.06 * data["previous_orders"]
        + 0.0001 * (2500 - data["order_value"])
    )
    probability = 1 / (1 + np.exp(-score))
    labels = rng.binomial(1, probability)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(
        n_estimators=120, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy

@st.cache_data
def make_orders():
    rng = np.random.default_rng(10)
    n = 100
    return pd.DataFrame({
        "Order ID": [f"ORD-{i:03}" for i in range(1, n + 1)],
        "Order Value (INR )": rng.integers(300, 5000, n),
        "Product Cost (INR )": rng.integers(150, 2500, n),
        "COD": rng.integers(0, 2, n),
        "Customer Return Rate": rng.uniform(0, 0.8, n).round(2),
        "Distance (km)": rng.integers(2, 1000, n),
        "Previous Orders": rng.integers(0, 30, n),
        "Shipping Cost (INR )": rng.integers(40, 150, n),
    })

model, test_accuracy = build_model()
orders = make_orders()

features = pd.DataFrame({
    "order_value": orders["Order Value (INR )"],
    "is_cod": orders["COD"],
    "customer_return_rate": orders["Customer Return Rate"],
    "distance_km": orders["Distance (km)"],
    "previous_orders": orders["Previous Orders"],
})
orders["RTO Risk (%)"] = (model.predict_proba(features)[:, 1] * 100).round(1)
orders["Payment Method"] = np.where(orders["COD"] == 1, "COD", "Prepaid")

# Simplified estimates, not accounting-grade profit
orders["Delivery Profit (INR )"] = (
    orders["Order Value (INR )"] - orders["Product Cost (INR )"]
    - orders["Shipping Cost (INR )"] - 20
)
orders["Estimated RTO Cost (INR )"] = orders["Shipping Cost (INR )"] * 2 + 20
risk = orders["RTO Risk (%)"] / 100
orders["Expected Profit (INR )"] = (
    (1 - risk) * orders["Delivery Profit (INR )"]
    - risk * orders["Estimated RTO Cost (INR )"]
).round(2)

def recommend(row):
    r = row["RTO Risk (%)"]
    if r >= 60 and row["COD"] == 1:
        return "Verify customer + confirm COD"
    if r >= 60:
        return "Verify address before dispatch"
    if r >= 35 and row["COD"] == 1:
        return "Call customer before dispatch"
    if r >= 35:
        return "Confirm delivery details"
    if row["Expected Profit (INR )"] < 0:
        return "Review pricing and shipping"
    return "Standard delivery"

orders["Recommended Action"] = orders.apply(recommend, axis=1)
orders["Risk Category"] = np.select(
    [orders["RTO Risk (%)"] >= 60, orders["RTO Risk (%)"] >= 35],
    ["High", "Medium"], default="Low"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# [BOX] ProfitPilot")
    st.caption("SMARTER DELIVERY DECISIONS")
    st.divider()
    st.markdown("###  Filters")
    min_risk = st.slider("Minimum RTO risk", 0, 100, 0, format="%d%%")
    payment_filter = st.selectbox("Payment method", ["All", "COD", "Prepaid"])
    category_filter = st.selectbox("Risk category", options=["All", "High", "Medium", "Low"], index=0, key="risk_category_filter")
    profit_filter = st.selectbox("Expected profit", ["All", "Positive only", "Negative only"])
    st.divider()
    if st.button("Reset Reset filters", use_container_width=True):
        st.rerun()
    st.markdown("### Info Prototype")
    st.caption("100 simulated orders  Random Forest  Synthetic labels")
    st.caption("Built by Team DivergeX")

# ---------------- APPLY FILTERS ----------------
filtered = orders[orders["RTO Risk (%)"] >= min_risk].copy()
if payment_filter != "All":
    filtered = filtered[filtered["Payment Method"] == payment_filter]
if category_filter != "All":
    filtered = filtered[filtered["Risk Category"] == category_filter]
if profit_filter == "Positive only":
    filtered = filtered[filtered["Expected Profit (INR )"] >= 0]
elif profit_filter == "Negative only":
    filtered = filtered[filtered["Expected Profit (INR )"] < 0]

# ---------------- HEADER ----------------
top_l, top_r = st.columns([4, 1])
with top_l:
    st.title("ProfitPilot")
    st.caption("Profit-Aware Delivery & RTO Intelligence Engine")
with top_r:
    st.markdown("<div style='text-align:right;padding-top:20px'><span class='pill'>* DEMO MODE</span></div>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">Order intelligence workspace</div>
  <div class="hero-title">Smarter Decisions. Higher Profits.</div>
  <div class="hero-sub">
    Identify orders that may return to origin, estimate the profit impact,
    and prioritize customer verification before dispatch.<br>
    <b>Team DivergeX</b> &nbsp;|&nbsp; Hackathon prototype &nbsp;|&nbsp; Synthetic demo data
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-kicker">Business overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("[BOX] Orders in view", f"{len(filtered):,}")
with k2:
    st.metric("[!] High-risk orders", f"{int((filtered['RTO Risk (%)'] >= 60).sum()):,}")
with k3:
    total_profit = filtered["Expected Profit (INR )"].sum()
    st.metric(" Expected profit", f"INR {total_profit:,.0f}")
with k4:
    avg_risk = filtered["RTO Risk (%)"].mean()
    st.metric(" Average RTO risk", f"{avg_risk:.1f}%" if not filtered.empty else "N/A")

st.write("")
st.markdown('<div class="section-kicker">Analytics & performance</div>', unsafe_allow_html=True)
chart1, chart2 = st.columns([1.15, 1])

with chart1:
    st.subheader("RTO Risk Distribution")
    if not filtered.empty:
        fig = px.histogram(
            filtered, x="RTO Risk (%)", nbins=12,
            color_discrete_sequence=["#5578f5"]
        )
        fig.update_layout(
            height=330, margin=dict(l=12, r=12, t=20, b=12),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#52627d"), xaxis_title="RTO risk (%)",
            yaxis_title="Orders", showlegend=False
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#edf1f8")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No orders match these filters.")

with chart2:
    st.subheader("Expected Profit by Payment Method")
    if not filtered.empty:
        summary = filtered.groupby("Payment Method", as_index=False)["Expected Profit (INR )"].sum()
        fig2 = px.bar(
            summary, x="Payment Method", y="Expected Profit (INR )",
            color="Payment Method",
            color_discrete_map={"COD": "#8068ed", "Prepaid": "#2ba889"},
            text_auto=".2s"
        )
        fig2.update_layout(
            height=330, margin=dict(l=12, r=12, t=20, b=12),
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="#52627d"), xaxis_title="",
            yaxis_title="Estimated profit (INR )", showlegend=False
        )
        fig2.update_xaxes(showgrid=False)
        fig2.update_yaxes(gridcolor="#edf1f8")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No orders match these filters.")

# ---------------- ACTION CENTER ----------------
st.markdown('<div class="section-kicker">Actionable insights</div>', unsafe_allow_html=True)
st.subheader("Business Action Center")
insight_col, action_col = st.columns(2)

with insight_col:
    if filtered.empty:
        st.info("Adjust the filters to view insights.")
    else:
        high_n = int((filtered["RTO Risk (%)"] >= 60).sum())
        loss_n = int((filtered["Expected Profit (INR )"] < 0).sum())
        medium_n = int(((filtered["RTO Risk (%)"] >= 35) & (filtered["RTO Risk (%)"] < 60)).sum())
        st.markdown(f'<div class="insight"> <b>{high_n} high-risk orders</b> may need customer or address verification.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight"> <b>{medium_n} medium-risk orders</b> may benefit from a pre-dispatch confirmation.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight"> <b>{loss_n} orders</b> have negative estimated expected profit under the current simplified cost assumptions.</div>', unsafe_allow_html=True)

with action_col:
    st.markdown("####  Suggested workflow")
    st.markdown('<div class="insight">1. <b>Verify first:</b> prioritize high-risk COD orders.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">2. <b>Confirm next:</b> contact medium-risk customers before dispatch.</div>', unsafe_allow_html=True)
    st.markdown('<div class="insight">3. <b>Review economics:</b> inspect orders with negative expected profit.</div>', unsafe_allow_html=True)
    st.caption("Recommendations are rules-based suggestions, not guaranteed outcomes.")

# ---------------- ORDER TABLE ----------------
st.divider()
table_l, table_r = st.columns([3, 1])
with table_l:
    st.subheader("[BOX] Order-level intelligence")
    st.caption("Sorted by highest predicted RTO risk")
with table_r:
    st.write("")
    st.download_button(
        "Download Download CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="profitpilot_orders.csv",
        mime="text/csv",
        use_container_width=True
    )

if not filtered.empty:
    display = filtered.sort_values("RTO Risk (%)", ascending=False).copy()
    display["Customer Return Rate"] = display["Customer Return Rate"] * 100
    st.dataframe(
        display[[
            "Order ID", "Order Value (INR )", "Payment Method",
            "Customer Return Rate", "Distance (km)", "RTO Risk (%)",
            "Delivery Profit (INR )", "Estimated RTO Cost (INR )",
            "Expected Profit (INR )", "Recommended Action"
        ]],
        use_container_width=True,
        hide_index=True,
        height=390,
        column_config={
            "RTO Risk (%)": st.column_config.ProgressColumn(
                "RTO risk", min_value=0, max_value=100, format="%.1f%%"
            ),
            "Customer Return Rate": st.column_config.NumberColumn(
                "Customer return rate", format="%.0f%%"
            ),
            "Order Value (INR )": st.column_config.NumberColumn(format="INR %.0f"),
            "Delivery Profit (INR )": st.column_config.NumberColumn(format="INR %.0f"),
            "Estimated RTO Cost (INR )": st.column_config.NumberColumn(format="INR %.0f"),
            "Expected Profit (INR )": st.column_config.NumberColumn(format="INR %.2f"),
        }
    )
else:
    st.warning("No orders match the selected filters.")

# ---------------- MODEL TRANSPARENCY ----------------
with st.expander(" Model details & limitations"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", "Random Forest")
    m2.metric("Synthetic training records", "2,400")
    m3.metric("Synthetic test accuracy", f"{test_accuracy:.1%}")
    st.warning(
        "This hackathon prototype is trained on simulated data with synthetic labels. "
        "The test score does not establish real-world RTO prediction accuracy. "
        "Profit calculations are simplified estimates and omit some possible business costs. "
        "Validate with real historical orders and outcomes before operational use."
    )

st.markdown("""
<div class="footer">
  <b>ProfitPilot</b> | Built by Team DivergeX<br>
  Profit-aware delivery decisions &nbsp;|&nbsp; Hackathon prototype
</div>
""", unsafe_allow_html=True)
