import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="ProfitPilot | Order Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {--ink:#172746;--muted:#667794;--line:#e5ebf5;--blue:#4169f5;}
html,body,[class*="css"]{font-family:Inter,sans-serif;}
.stApp{background:#f5f7fc;color:var(--ink);}
.block-container{max-width:1600px;padding-top:1.5rem;padding-bottom:2.5rem;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#111b32,#1b2d50);}
[data-testid="stSidebar"] *{color:#eef3ff;}
[data-testid="stMetric"]{background:white;border:1px solid var(--line);
border-radius:14px;padding:16px;box-shadow:0 4px 15px #1a30590b;}
.hero{background:linear-gradient(115deg,#e8efff,#fff 58%,#e5f8f2);
border:1px solid #dce6f8;border-radius:18px;padding:23px;margin:10px 0 22px;}
.insight{background:white;border:1px solid #e5ebf5;border-left:4px solid #5578f5;
border-radius:10px;padding:12px;margin-bottom:9px;}
.footer{color:#8491a8;font-size:12px;text-align:center;padding:22px;}
</style>
""", unsafe_allow_html=True)


# ---------- SYNTHETIC DEMO MODEL WITH CUSTOMER HISTORY ----------
HISTORY_FEATURES = [
    "customer_rto_rate", "delivery_success_rate", "historical_return_rate",
    "cancellation_rate", "cod_success_rate"
]
MODEL_FEATURES = [
    "order_value", "is_cod", "customer_return_rate", "distance_km",
    "previous_orders", *HISTORY_FEATURES
]

@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    n = 3200
    data = pd.DataFrame({
        "order_value": rng.integers(300, 5000, n),
        "is_cod": rng.integers(0, 2, n),
        "customer_return_rate": rng.uniform(0, 0.8, n),
        "distance_km": rng.integers(2, 1000, n),
        "previous_orders": rng.integers(0, 30, n),
        "customer_rto_rate": rng.beta(1.5, 7, n),
        "delivery_success_rate": rng.beta(7, 2, n),
        "historical_return_rate": rng.beta(1.5, 8, n),
        "cancellation_rate": rng.beta(1.2, 10, n),
        "cod_success_rate": rng.beta(6, 2, n),
    })
    score = (
        -2.3 + 1.25 * data["is_cod"]
        + 1.8 * data["customer_return_rate"]
        + 0.001 * data["distance_km"]
        - 0.035 * data["previous_orders"]
        + 3.0 * data["customer_rto_rate"]
        - 1.8 * data["delivery_success_rate"]
        + 1.1 * data["historical_return_rate"]
        + 1.0 * data["cancellation_rate"]
        - 1.1 * data["cod_success_rate"]
        + 0.0001 * (2500 - data["order_value"])
    )
    probability = 1 / (1 + np.exp(-score))
    labels = rng.binomial(1, probability)
    x_train, x_test, y_train, y_test = train_test_split(
        data[MODEL_FEATURES], labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = RandomForestClassifier(
        n_estimators=140, max_depth=11, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy


@st.cache_data
def make_demo_orders():
    rng = np.random.default_rng(10)
    n = 100
    customer_ids = [f"CUST-{i:03}" for i in rng.integers(1, 46, n)]
    return pd.DataFrame({
        "Order ID": [f"ORD-{i:03}" for i in range(1, n + 1)],
        "Customer ID": customer_ids,
        "Order Value": rng.integers(300, 5000, n),
        "Product Cost": rng.integers(150, 2500, n),
        "COD": rng.integers(0, 2, n),
        "Customer Return Rate": rng.uniform(0, 0.8, n).round(2),
        "Distance (km)": rng.integers(2, 1000, n),
        "Previous Orders": rng.integers(0, 30, n),
        "Shipping Cost": rng.integers(40, 150, n),
    })


@st.cache_data
def make_demo_customer_history():
    rng = np.random.default_rng(25)
    rows = []
    for i in range(1, 46):
        total = int(rng.integers(1, 26))
        rto = int(rng.binomial(total, rng.uniform(0.02, 0.48)))
        returned = int(rng.binomial(max(total - rto, 0), rng.uniform(0.01, 0.30)))
        cancelled = int(rng.binomial(total, rng.uniform(0.01, 0.22)))
        cod_orders = int(rng.integers(0, total + 1))
        cod_delivered = int(rng.binomial(cod_orders, rng.uniform(0.35, 0.98))) if cod_orders else 0
        delivered = max(0, total - rto - cancelled)
        rows.append({
            "customer_id": f"CUST-{i:03}", "total_orders": total,
            "delivered_orders": min(delivered, total), "rto_orders": rto,
            "returned_orders": returned, "cancelled_orders": cancelled,
            "cod_orders": cod_orders, "cod_delivered_orders": cod_delivered,
        })
    return pd.DataFrame(rows)


def apply_customer_history(orders, history_df=None):
    """Join optional historical counts and derive rates; missing history uses neutral priors."""
    df = orders.copy()
    if "Customer ID" not in df.columns:
        df["Customer ID"] = "UNKNOWN"
    df["Customer ID"] = df["Customer ID"].astype(str).str.strip()
    df["History Available"] = False

    if history_df is not None and not history_df.empty:
        hist = history_df.copy()
        hist.columns = hist.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        aliases = {"customer_id": ["customer id", "customerid", "customer"],
                   "total_orders": ["total_order", "previous_orders"],
                   "delivered_orders": ["delivered", "successful_deliveries"],
                   "rto_orders": ["rto", "rto_count"],
                   "returned_orders": ["returns", "return_orders"],
                   "cancelled_orders": ["cancelled", "canceled_orders", "cancellations"],
                   "cod_orders": ["total_cod_orders"],
                   "cod_delivered_orders": ["cod_delivered", "successful_cod_orders"]}
        for standard, alternatives in aliases.items():
            for alt in alternatives:
                alt_key = alt.replace(" ", "_")
                if standard not in hist.columns and alt_key in hist.columns:
                    hist = hist.rename(columns={alt_key: standard})
        required_hist = ["customer_id", "total_orders", "delivered_orders", "rto_orders",
                         "returned_orders", "cancelled_orders", "cod_orders", "cod_delivered_orders"]
        missing_hist = [c for c in required_hist if c not in hist.columns]
        if missing_hist:
            raise ValueError("Customer history CSV mein columns missing hain: " + ", ".join(missing_hist))
        for col in required_hist[1:]:
            hist[col] = pd.to_numeric(hist[col], errors="coerce").fillna(0).clip(lower=0)
        hist["customer_id"] = hist["customer_id"].astype(str).str.strip()
        denom = hist["total_orders"].replace(0, np.nan)
        hist["customer_rto_rate"] = (hist["rto_orders"] / denom).clip(0, 1)
        hist["delivery_success_rate"] = (hist["delivered_orders"] / denom).clip(0, 1)
        hist["historical_return_rate"] = (hist["returned_orders"] / denom).clip(0, 1)
        hist["cancellation_rate"] = (hist["cancelled_orders"] / denom).clip(0, 1)
        cod_denom = hist["cod_orders"].replace(0, np.nan)
        hist["cod_success_rate"] = (hist["cod_delivered_orders"] / cod_denom).clip(0, 1)
        hist = hist.drop_duplicates("customer_id", keep="last")
        rate_cols = HISTORY_FEATURES
        df = df.merge(hist[["customer_id", *rate_cols]], left_on="Customer ID", right_on="customer_id", how="left")
        df.drop(columns=["customer_id"], inplace=True, errors="ignore")
        df["History Available"] = df["customer_rto_rate"].notna()
    else:
        for col in HISTORY_FEATURES:
            df[col] = np.nan

    # Neutral priors keep first-time/unknown customers from being automatically labelled risky.
    defaults = {
        "customer_rto_rate": 0.15, "delivery_success_rate": 0.75,
        "historical_return_rate": 0.10, "cancellation_rate": 0.08,
        "cod_success_rate": 0.70,
    }
    for col, default in defaults.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default).clip(0, 1)
    return df


def prepare_orders(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    aliases = {
        "Order ID": ["order_id"],
        "Customer ID": ["customer_id", "customerid", "customer"],
        "Order Value": ["order_value", "Order Value (INR )", "Order Value (INR)"],
        "Product Cost": ["product_cost", "Product Cost (INR )", "Product Cost (INR)"],
        "COD": ["is_cod", "Payment Method", "payment_method"],
        "Customer Return Rate": ["customer_return_rate"],
        "Distance (km)": ["distance_km", "Distance"],
        "Previous Orders": ["previous_orders"],
        "Shipping Cost": ["shipping_cost", "Shipping Cost (INR )", "Shipping Cost (INR)"],
    }

    rename = {}
    for standard, alternatives in aliases.items():
        for col in [standard] + alternatives:
            if col in df.columns:
                rename[col] = standard
                break
    df = df.rename(columns=rename)

    required = [
        "Order Value", "Product Cost", "COD", "Customer Return Rate",
        "Distance (km)", "Previous Orders", "Shipping Cost"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("CSV mein required columns missing hain: " + ", ".join(missing))

    if "Order ID" not in df.columns:
        df["Order ID"] = [f"CSV-{i:03}" for i in range(1, len(df) + 1)]

    payment = df["COD"].astype(str).str.strip().str.lower()
    payment_map = {
        "cod": 1, "cash on delivery": 1, "1": 1, "true": 1,
        "prepaid": 0, "online": 0, "0": 0, "false": 0
    }
    if not payment.isin(payment_map).all():
        raise ValueError("COD column mein COD/Prepaid ya 1/0 values use karo.")
    df["COD"] = payment.map(payment_map)

    numeric_cols = [
        "Order Value", "Product Cost", "Customer Return Rate",
        "Distance (km)", "Previous Orders", "Shipping Cost"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        raise ValueError("CSV mein koi order row nahi hai.")
    if df[numeric_cols].isna().any().any():
        raise ValueError("Numeric columns mein blank ya invalid values hain.")
    if (df[["Order Value", "Product Cost", "Distance (km)",
            "Previous Orders", "Shipping Cost"]] < 0).any().any():
        raise ValueError("Value, cost, distance aur order counts negative nahi ho sakte.")

    if df["Customer Return Rate"].max() > 1:
        if df["Customer Return Rate"].max() > 100:
            raise ValueError("Return rate 0–1 ya 0–100 scale par hona chahiye.")
        df["Customer Return Rate"] /= 100
    if not df["Customer Return Rate"].between(0, 1).all():
        raise ValueError("Return rate 0–1 ya 0–100 scale par hona chahiye.")
    if not df["COD"].isin([0, 1]).all():
        raise ValueError("COD value 1 ya 0 honi chahiye.")
    return df


def analyze_orders(df, model):
    df = df.copy()
    features = pd.DataFrame({
        "order_value": df["Order Value"],
        "is_cod": df["COD"],
        "customer_return_rate": df["Customer Return Rate"],
        "distance_km": df["Distance (km)"],
        "previous_orders": df["Previous Orders"],
        **{col: df[col] for col in HISTORY_FEATURES},
    })[MODEL_FEATURES]

    df["RTO Risk (%)"] = (model.predict_proba(features)[:, 1] * 100).round(1)
    df["Payment Method"] = np.where(df["COD"] == 1, "COD", "Prepaid")
    df["Delivery Profit"] = (
        df["Order Value"] - df["Product Cost"] - df["Shipping Cost"] - 20
    )
    df["Estimated RTO Cost"] = df["Shipping Cost"] * 2 + 20
    risk = df["RTO Risk (%)"] / 100
    df["Expected Profit"] = (
        (1 - risk) * df["Delivery Profit"] - risk * df["Estimated RTO Cost"]
    ).round(2)

    def recommend(row):
        r, profit, cod = row["RTO Risk (%)"], row["Expected Profit"], row["COD"]
        if profit < 0 and r >= 60:
            return "HOLD & REVIEW: High risk + expected loss"
        if profit < 0:
            return "REVIEW PRICING: Expected loss"
        if r >= 60 and cod == 1:
            return "VERIFY CUSTOMER: High-risk COD"
        if r >= 60:
            return "VERIFY ADDRESS: High RTO risk"
        if r >= 35 and cod == 1:
            return "CALL CUSTOMER: Medium-risk COD"
        if r >= 35:
            return "CONFIRM DELIVERY DETAILS"
        return "STANDARD DELIVERY"

    df["Recommended Action"] = df.apply(recommend, axis=1)
    df["Risk Category"] = np.select(
        [df["RTO Risk (%)"] >= 60, df["RTO Risk (%)"] >= 35],
        ["High", "Medium"], default="Low"
    )
    df["Decision Priority"] = np.select(
        [
            (df["Expected Profit"] < 0) & (df["RTO Risk (%)"] >= 60),
            df["RTO Risk (%)"] >= 60,
            (df["Expected Profit"] < 0) | (df["RTO Risk (%)"] >= 35)
        ],
        ["Critical", "High", "Medium"], default="Normal"
    )

    # Simple, transparent indicators rather than causal explanations.
    def risk_reasons(row):
        reasons = []
        if row["COD"] == 1:
            reasons.append("COD payment")
        if row["Customer Return Rate"] >= 0.4:
            reasons.append("high current-order return signal")
        if row.get("History Available", False):
            if row["customer_rto_rate"] >= 0.3:
                reasons.append("high historical RTO rate")
            if row["delivery_success_rate"] < 0.6:
                reasons.append("low historical delivery success")
            if row["historical_return_rate"] >= 0.3:
                reasons.append("frequent historical returns")
            if row["cancellation_rate"] >= 0.25:
                reasons.append("frequent historical cancellations")
            if row["COD"] == 1 and row["cod_success_rate"] < 0.6:
                reasons.append("low historical COD success")
        else:
            reasons.append("customer history unavailable; neutral baseline used")
        if row["Distance (km)"] >= 500:
            reasons.append("long delivery distance")
        if row["Previous Orders"] <= 2:
            reasons.append("few previous orders")
        if row["Order Value"] <= 800:
            reasons.append("low order value")
        return ", ".join(reasons) if reasons else "No listed risk flag from simple rules"

    df["Risk Indicators"] = df.apply(risk_reasons, axis=1)
    return df


# ---------- MODEL + INPUT ----------
model, test_accuracy = build_model()
st.sidebar.title("📦 ProfitPilot")
st.sidebar.caption("SMARTER DELIVERY DECISIONS")
st.sidebar.divider()
st.sidebar.subheader("Order data")
data_mode = st.sidebar.radio("Choose data source", ["Demo data", "Upload CSV"])

if data_mode == "Upload CSV":
    st.sidebar.caption("Upload a CSV with the required order columns.")
    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
        color: #ffffff !important; opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
        color: #334155 !important; opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #2563eb !important; color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("Upload orders CSV", type=["csv"])
    template = pd.DataFrame({
        "Order ID": ["ORD-101", "ORD-102"], "Customer ID": ["CUST-001", "CUST-002"],
        "Order Value": [1500, 2200], "Product Cost": [700, 1000],
        "COD": [1, 0], "Customer Return Rate": [0.25, 0.10],
        "Distance (km)": [120, 60], "Previous Orders": [5, 12],
        "Shipping Cost": [80, 65],
    })
    st.sidebar.download_button(
        "Download CSV template", template.to_csv(index=False).encode("utf-8"),
        file_name="profitpilot_template.csv", mime="text/csv",
        use_container_width=True
    )
    history_template = pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-002"], "total_orders": [10, 6],
        "delivered_orders": [6, 5], "rto_orders": [3, 0], "returned_orders": [1, 1],
        "cancelled_orders": [1, 1], "cod_orders": [8, 3], "cod_delivered_orders": [4, 3],
    })
    st.sidebar.download_button(
        "Download customer history template", history_template.to_csv(index=False).encode("utf-8"),
        file_name="customer_history_template.csv", mime="text/csv", use_container_width=True
    )
    history_file = st.sidebar.file_uploader("Optional customer history CSV", type=["csv"], key="history_upload")
    if uploaded_file is None:
        st.title("ProfitPilot")
        st.info("CSV upload karo ya sidebar se Demo data select karo.")
        st.stop()
    try:
        clean_orders = prepare_orders(pd.read_csv(uploaded_file))
        history_data = pd.read_csv(history_file) if history_file is not None else None
        clean_orders = apply_customer_history(clean_orders, history_data)
        orders = analyze_orders(clean_orders, model)
        st.sidebar.success(f"{len(orders)} orders loaded")
    except Exception as e:
        st.error(f"CSV process nahi ho saki: {e}")
        st.stop()
    st.caption(f"Data source: Uploaded CSV · {uploaded_file.name}")
else:
    demo_orders = prepare_orders(make_demo_orders())
    demo_orders = apply_customer_history(demo_orders, make_demo_customer_history())
    orders = analyze_orders(demo_orders, model)
    st.caption("Data source: 100 simulated demo orders + simulated customer history")

# ---------- FILTERS ----------
st.sidebar.divider()
st.sidebar.subheader("Filters")
min_risk = st.sidebar.slider("Minimum RTO risk", 0, 100, 0, format="%d%%")
payment_filter = st.sidebar.selectbox("Payment method", ["All", "COD", "Prepaid"])
category_filter = st.sidebar.selectbox("Risk category", ["All", "High", "Medium", "Low"])
profit_filter = st.sidebar.selectbox("Expected profit", ["All", "Positive only", "Negative only"])

filtered = orders[orders["RTO Risk (%)"] >= min_risk].copy()
if payment_filter != "All":
    filtered = filtered[filtered["Payment Method"] == payment_filter]
if category_filter != "All":
    filtered = filtered[filtered["Risk Category"] == category_filter]
if profit_filter == "Positive only":
    filtered = filtered[filtered["Expected Profit"] >= 0]
elif profit_filter == "Negative only":
    filtered = filtered[filtered["Expected Profit"] < 0]

# ---------- HEADER ----------
st.title("ProfitPilot")
st.caption("Profit-Aware Delivery & RTO Intelligence Engine")
st.markdown("""
<div class="hero">
  <b>ORDER INTELLIGENCE WORKSPACE</b>
  <h2>Smarter Decisions. Higher Profits.</h2>
  Identify orders that may return to origin, estimate the profit impact,
  and prioritize customer verification before dispatch.<br><br>
  <b>Team DivergeX</b> | Hackathon prototype
</div>
""", unsafe_allow_html=True)

# ---------- BUSINESS OVERVIEW ----------
st.subheader("Business overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Orders in view", f"{len(filtered):,}")
k2.metric("High-risk orders", f"{int((filtered['RTO Risk (%)'] >= 60).sum()):,}")
k3.metric("Expected profit", f"INR {filtered['Expected Profit'].sum():,.0f}")
avg_risk = filtered["RTO Risk (%)"].mean()
k4.metric("Average RTO risk", f"{avg_risk:.1f}%" if not filtered.empty else "N/A")

# ---------- CUSTOMER HISTORY INSIGHTS ----------
st.subheader("Customer history intelligence")
history_count = int(filtered["History Available"].sum()) if "History Available" in filtered else 0
h1, h2, h3, h4 = st.columns(4)
h1.metric("Orders matched to history", f"{history_count:,}")
h2.metric("High historical RTO rate", f"{int((filtered['customer_rto_rate'] >= 0.30).sum()):,}")
h3.metric("Low delivery success", f"{int((filtered['delivery_success_rate'] < 0.60).sum()):,}")
h4.metric("History coverage", f"{(history_count / len(filtered) * 100):.1f}%" if len(filtered) else "N/A")
st.caption("History rates are calculated from uploaded counts. Customers without a match use neutral baseline rates and are not automatically classified as high-risk.")

# ---------- CHARTS ----------
st.subheader("Analytics & performance")
chart1, chart2 = st.columns(2)
with chart1:
    st.markdown("**RTO Risk Distribution**")
    if not filtered.empty:
        fig = px.histogram(filtered, x="RTO Risk (%)", nbins=12,
                           color_discrete_sequence=["#5578f5"])
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No orders match these filters.")
with chart2:
    st.markdown("**Expected Profit by Payment Method**")
    if not filtered.empty:
        summary = filtered.groupby("Payment Method", as_index=False)["Expected Profit"].sum()
        fig2 = px.bar(
            summary, x="Payment Method", y="Expected Profit",
            color="Payment Method",
            color_discrete_map={"COD": "#8068ed", "Prepaid": "#2ba889"},
            text_auto=".2s"
        )
        fig2.update_layout(height=320)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No orders match these filters.")

# ---------- PROFIT WHAT-IF SIMULATOR ----------
st.divider()
st.subheader("Profit What-if Simulator")
st.caption("Compare current estimates with hypothetical strategies. This is not a guaranteed outcome.")
with st.container(border=True):
    sim1, sim2 = st.columns(2)
    with sim1:
        prepaid_pct = st.slider(
            "High-risk COD orders hypothetically converted to prepaid (%)",
            0, 100, 25, 5, key="sim_prepaid"
        )
    with sim2:
        verification_reduction = st.slider(
            "Assumed RTO-risk reduction after verification (%)",
            0, 50, 10, 5, key="sim_verify"
        )
    verification_cost = st.number_input(
        "Assumed verification cost per order (INR)", 0, 500, 10, 5,
        key="sim_cost"
    )

    if filtered.empty:
        st.info("Filters se orders select karo to simulator use kar sako.")
    else:
        sim = filtered.copy()
        sim["Scenario Risk"] = sim["RTO Risk (%)"] / 100
        sim["Scenario COD"] = sim["COD"].astype(int)
        candidates = sim[(sim["COD"] == 1) & (sim["RTO Risk (%)"] >= 35)].sort_values(
            "RTO Risk (%)", ascending=False
        )
        convert_count = int(len(candidates) * prepaid_pct / 100)
        converted_indices = candidates.index[:convert_count]
        sim.loc[converted_indices, "Scenario COD"] = 0

        # Illustrative assumption: conversion reduces original risk by 15% relatively.
        sim.loc[converted_indices, "Scenario Risk"] *= 0.85
        verify_mask = (sim["RTO Risk (%)"] >= 60) & (~sim.index.isin(converted_indices))
        sim.loc[verify_mask, "Scenario Risk"] *= (1 - verification_reduction / 100)

        sim["Scenario Delivery Profit"] = (
            sim["Order Value"] - sim["Product Cost"] - sim["Shipping Cost"] - 20
        )
        sim["Scenario RTO Cost"] = sim["Shipping Cost"] * 2 + 20
        sim["Scenario Expected Profit"] = (
            (1 - sim["Scenario Risk"]) * sim["Scenario Delivery Profit"]
            - sim["Scenario Risk"] * sim["Scenario RTO Cost"]
            - verify_mask.astype(int) * verification_cost
        ).round(2)

        current_profit = sim["Expected Profit"].sum()
        scenario_profit = sim["Scenario Expected Profit"].sum()
        profit_change = scenario_profit - current_profit
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current estimated profit", f"₹{current_profit:,.0f}")
        m2.metric("Scenario estimated profit", f"₹{scenario_profit:,.0f}")
        m3.metric("Estimated difference", f"₹{profit_change:,.0f}",
                  delta=f"{profit_change:,.0f}")
        m4.metric("Orders converted in scenario", f"{convert_count:,}")

        comparison = pd.DataFrame({
            "Scenario": ["Current estimate", "What-if scenario"],
            "Estimated Profit (INR)": [current_profit, scenario_profit]
        })
        fig_sim = px.bar(
            comparison, x="Scenario", y="Estimated Profit (INR)",
            color="Scenario", text_auto=".2s", title="Current vs Hypothetical Profit"
        )
        fig_sim.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_sim, use_container_width=True)

        st.caption(
            "Assumptions: selected COD orders have 15% relative risk reduction after "
            "prepaid conversion; verification reduces remaining high-risk orders' risk "
            "by the chosen percentage. These are illustrative assumptions, not learned "
            "from historical outcomes. The simulator does not model acceptance rates "
            "or all operational costs."
        )
        scenario_export = sim.copy()
        scenario_export["Scenario RTO Risk (%)"] = (scenario_export["Scenario Risk"] * 100).round(1)
        scenario_export["Scenario Payment Method"] = np.where(
            scenario_export["Scenario COD"] == 1, "COD", "Prepaid"
        )
        st.download_button(
            "Download What-if Scenario CSV",
            data=scenario_export.to_csv(index=False).encode("utf-8"),
            file_name="profitpilot_what_if_scenario.csv",
            mime="text/csv", use_container_width=True
        )

# ---------- ACTION CENTER ----------
st.subheader("Business Action Center")
if filtered.empty:
    st.info("Adjust filters to see order insights.")
else:
    high_n = int((filtered["RTO Risk (%)"] >= 60).sum())
    medium_n = int(((filtered["RTO Risk (%)"] >= 35) & (filtered["RTO Risk (%)"] < 60)).sum())
    loss_n = int((filtered["Expected Profit"] < 0).sum())
    critical_n = int((filtered["Decision Priority"] == "Critical").sum())
    a, b, c, d = st.columns(4)
    a.markdown(f'<div class="insight"><b>{high_n} high-risk orders</b><br>May need customer/address verification.</div>', unsafe_allow_html=True)
    b.markdown(f'<div class="insight"><b>{medium_n} medium-risk orders</b><br>May benefit from confirmation.</div>', unsafe_allow_html=True)
    c.markdown(f'<div class="insight"><b>{loss_n} orders</b><br>Have negative estimated expected profit.</div>', unsafe_allow_html=True)
    d.markdown(f'<div class="insight"><b>{critical_n} critical-priority orders</b><br>Review risk and expected loss first.</div>', unsafe_allow_html=True)
st.caption("Suggested actions are rules-based, not guaranteed outcomes.")

# ---------- ORDER TABLE + DOWNLOAD ----------
st.divider()
st.subheader("Order-level intelligence")
st.caption("Sorted by highest predicted RTO risk. Risk indicators are simple rule-based flags, not causal explanations.")
if not filtered.empty:
    display = filtered.sort_values("RTO Risk (%)", ascending=False).copy()
    display["Customer Return Rate"] = (display["Customer Return Rate"] * 100).round(1)
    for rate_col in HISTORY_FEATURES:
        display[rate_col] = (display[rate_col] * 100).round(1)
    st.dataframe(
        display[[
            "Order ID", "Customer ID", "History Available", "Order Value", "Product Cost", "Payment Method",
            "Customer Return Rate", "Previous Orders", "customer_rto_rate", "delivery_success_rate",
            "historical_return_rate", "cancellation_rate", "cod_success_rate", "Distance (km)",
            "Shipping Cost", "RTO Risk (%)", "Risk Category", "Decision Priority",
            "Risk Indicators", "Delivery Profit", "Estimated RTO Cost",
            "Expected Profit", "Recommended Action"
        ]],
        use_container_width=True, hide_index=True, height=420
    )
    st.download_button(
        "Download analyzed orders CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="profitpilot_analyzed_orders.csv",
        mime="text/csv", use_container_width=True
    )

    with st.expander("Inspect one order's decision"):
        selected_id = st.selectbox("Choose an order", display["Order ID"].tolist())
        selected = filtered[filtered["Order ID"] == selected_id].iloc[0]
        q1, q2, q3 = st.columns(3)
        q1.metric("RTO risk", f"{selected['RTO Risk (%)']:.1f}%")
        q2.metric("Expected profit", f"₹{selected['Expected Profit']:,.2f}")
        q3.metric("Priority", selected["Decision Priority"])
        st.write("**Recommended action:**", selected["Recommended Action"])
        st.write("**Risk indicators:**", selected["Risk Indicators"])
        st.write("**Customer history matched:**", "Yes" if selected["History Available"] else "No — neutral baseline used")
        if selected["History Available"]:
            st.write({
                "Historical RTO rate": f"{selected['customer_rto_rate']:.1%}",
                "Delivery success rate": f"{selected['delivery_success_rate']:.1%}",
                "Historical return rate": f"{selected['historical_return_rate']:.1%}",
                "Cancellation rate": f"{selected['cancellation_rate']:.1%}",
                "COD success rate": f"{selected['cod_success_rate']:.1%}",
            })
else:
    st.warning("No orders match the selected filters.")

# ---------- TRANSPARENCY ----------
with st.expander("Model details & limitations"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", "Random Forest")
    m2.metric("Synthetic training records", "3,200")
    m3.metric("Synthetic test accuracy", f"{test_accuracy:.1%}")
    st.warning(
        "This prototype uses synthetic training data and synthetic labels, including simulated customer-history patterns. Test accuracy "
        "does not establish real-world RTO prediction accuracy. Uploaded orders are scored "
        "by this model and do not retrain it. Profit values are simplified estimates, not "
        "accounting-grade results. Scenario improvements rely on editable assumptions. "
        "Validate against historical orders and actual outcomes before operational use. Customer history must be authorized and handled with consent; no Flipkart/Amazon private account history is accessed."
    )

st.markdown("""
<div class="footer">
<b>ProfitPilot</b> | Built by Team DivergeX<br>
Profit-aware delivery decisions | Hackathon prototype
</div>
""", unsafe_allow_html=True)
