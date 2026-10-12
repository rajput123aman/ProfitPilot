
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


# ---------- SYNTHETIC DEMO MODEL ----------
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
        data, labels, test_size=0.2, random_state=42,
        stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))
    return model, accuracy


@st.cache_data
def make_demo_orders():
    rng = np.random.default_rng(10)
    n = 100

    return pd.DataFrame({
        "Order ID": [f"ORD-{i:03}" for i in range(1, n + 1)],
        "Order Value": rng.integers(300, 5000, n),
        "Product Cost": rng.integers(150, 2500, n),
        "COD": rng.integers(0, 2, n),
        "Customer Return Rate": rng.uniform(0, 0.8, n).round(2),
        "Distance (km)": rng.integers(2, 1000, n),
        "Previous Orders": rng.integers(0, 30, n),
        "Shipping Cost": rng.integers(40, 150, n),
    })


def prepare_orders(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Accept common alternative column names
    aliases = {
        "Order ID": ["order_id", "Order ID"],
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
        "Order Value", "Product Cost", "COD",
        "Customer Return Rate", "Distance (km)",
        "Previous Orders", "Shipping Cost"
    ]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            "CSV mein ye required columns missing hain: "
            + ", ".join(missing)
        )

    if "Order ID" not in df.columns:
        df["Order ID"] = [f"CSV-{i:03}" for i in range(1, len(df) + 1)]

    # Convert payment labels into 1 = COD, 0 = Prepaid
    payment = df["COD"].astype(str).str.strip().str.lower()
    payment_map = {
        "cod": 1, "cash on delivery": 1, "1": 1, "true": 1,
        "prepaid": 0, "online": 0, "0": 0, "false": 0
    }

    if payment.isin(payment_map).all():
        df["COD"] = payment.map(payment_map)
    else:
        raise ValueError(
            "COD column mein sirf COD/Prepaid ya 1/0 values use karo."
        )

    numeric_cols = [
        "Order Value", "Product Cost", "Customer Return Rate",
        "Distance (km)", "Previous Orders", "Shipping Cost"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_cols].isna().any().any():
        raise ValueError(
            "Numeric columns mein blank ya invalid values hain. "
            "CSV ko correct karke dobara upload karo."
        )

    if df.empty:
        raise ValueError("CSV mein koi order row nahi hai.")

    if (df[["Order Value", "Product Cost", "Distance (km)",
            "Previous Orders", "Shipping Cost"]] < 0).any().any():
        raise ValueError("Cost, value, distance aur order counts negative nahi ho sakte.")

    # Accept return rate as either 0–1 or percentage 0–100
    if df["Customer Return Rate"].max() > 1:
        if df["Customer Return Rate"].max() > 100:
            raise ValueError("Customer Return Rate 0–1 ya 0–100 scale par hona chahiye.")
        df["Customer Return Rate"] = df["Customer Return Rate"] / 100

    if not df["Customer Return Rate"].between(0, 1).all():
        raise ValueError("Customer Return Rate 0–1 ya 0–100 scale par hona chahiye.")

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
    })

    df["RTO Risk (%)"] = (
        model.predict_proba(features)[:, 1] * 100
    ).round(1)

    df["Payment Method"] = np.where(
        df["COD"] == 1, "COD", "Prepaid"
    )

    df["Delivery Profit"] = (
        df["Order Value"] - df["Product Cost"]
        - df["Shipping Cost"] - 20
    )

    df["Estimated RTO Cost"] = df["Shipping Cost"] * 2 + 20

    risk = df["RTO Risk (%)"] / 100

    df["Expected Profit"] = (
        (1 - risk) * df["Delivery Profit"]
        - risk * df["Estimated RTO Cost"]
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
        if row["Expected Profit"] < 0:
            return "Review pricing and shipping"
        return "Standard delivery"

    df["Recommended Action"] = df.apply(recommend, axis=1)
    df["Risk Category"] = np.select(
        [df["RTO Risk (%)"] >= 60, df["RTO Risk (%)"] >= 35],
        ["High", "Medium"],
        default="Low"
    )

    return df


# ---------- MODEL + INPUT ----------
model, test_accuracy = build_model()

st.sidebar.title("📦 ProfitPilot")
st.sidebar.caption("SMARTER DELIVERY DECISIONS")
st.sidebar.divider()

st.sidebar.subheader("Order data")
data_mode = st.sidebar.radio(
    "Choose data source",
    ["Demo data", "Upload CSV"]
)

if data_mode == "Upload CSV":
    st.sidebar.caption("Upload a CSV with the required order columns.")

    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] label,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] label p {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] section * {
        color: #334155 !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background-color: #2563eb !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader(
        "Upload orders CSV", type=["csv"]
    )


    template = pd.DataFrame({
        "Order ID": ["ORD-101", "ORD-102"],
        "Order Value": [1500, 2200],
        "Product Cost": [700, 1000],
        "COD": [1, 0],
        "Customer Return Rate": [0.25, 0.10],
        "Distance (km)": [120, 60],
        "Previous Orders": [5, 12],
        "Shipping Cost": [80, 65],
    })

    st.sidebar.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="profitpilot_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    if uploaded_file is None:
        st.title("ProfitPilot")
        st.info("CSV upload karo ya sidebar se Demo data select karo.")
        st.stop()

    try:
        raw_orders = pd.read_csv(uploaded_file)
        clean_orders = prepare_orders(raw_orders)
        orders = analyze_orders(clean_orders, model)
        st.sidebar.success(f"{len(orders)} orders loaded")
    except Exception as e:
        st.error(f"CSV process nahi ho saki: {e}")
        st.stop()

    st.caption(f"Data source: Uploaded CSV · {uploaded_file.name}")

else:
    orders = analyze_orders(make_demo_orders(), model)
    st.caption("Data source: 100 simulated demo orders")

# ---------- FILTERS ----------
st.sidebar.divider()
st.sidebar.subheader("Filters")

min_risk = st.sidebar.slider(
    "Minimum RTO risk", 0, 100, 0, format="%d%%"
)
payment_filter = st.sidebar.selectbox(
    "Payment method", ["All", "COD", "Prepaid"]
)
category_filter = st.sidebar.selectbox(
    "Risk category", ["All", "High", "Medium", "Low"]
)
profit_filter = st.sidebar.selectbox(
    "Expected profit", ["All", "Positive only", "Negative only"]
)

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
k2.metric(
    "High-risk orders",
    f"{int((filtered['RTO Risk (%)'] >= 60).sum()):,}"
)
k3.metric(
    "Expected profit",
    f"INR {filtered['Expected Profit'].sum():,.0f}"
)
avg_risk = filtered["RTO Risk (%)"].mean()
k4.metric(
    "Average RTO risk",
    f"{avg_risk:.1f}%" if not filtered.empty else "N/A"
)

# ---------- CHARTS ----------
st.subheader("Analytics & performance")
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("**RTO Risk Distribution**")
    if not filtered.empty:
        fig = px.histogram(
            filtered, x="RTO Risk (%)", nbins=12,
            color_discrete_sequence=["#5578f5"]
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No orders match these filters.")

with chart2:
    st.markdown("**Expected Profit by Payment Method**")
    if not filtered.empty:
        summary = filtered.groupby(
            "Payment Method", as_index=False
        )["Expected Profit"].sum()

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

# ---------- ACTION CENTER ----------
st.subheader("Business Action Center")

if filtered.empty:
    st.info("Adjust filters to see order insights.")
else:
    high_n = int((filtered["RTO Risk (%)"] >= 60).sum())
    medium_n = int(
        ((filtered["RTO Risk (%)"] >= 35) &
         (filtered["RTO Risk (%)"] < 60)).sum()
    )
    loss_n = int((filtered["Expected Profit"] < 0).sum())

    a, b, c = st.columns(3)
    a.markdown(
        f'<div class="insight"><b>{high_n} high-risk orders</b>'
        ' may need customer/address verification.</div>',
        unsafe_allow_html=True
    )
    b.markdown(
        f'<div class="insight"><b>{medium_n} medium-risk orders</b>'
        ' may benefit from confirmation.</div>',
        unsafe_allow_html=True
    )
    c.markdown(
        f'<div class="insight"><b>{loss_n} orders</b>'
        ' have negative estimated expected profit.</div>',
        unsafe_allow_html=True
    )

st.caption(
    "Suggested actions are rules-based, not guaranteed outcomes."
)

# ---------- ORDER TABLE + DOWNLOAD ----------
st.divider()
st.subheader("Order-level intelligence")
st.caption("Sorted by highest predicted RTO risk")

if not filtered.empty:
    display = filtered.sort_values(
        "RTO Risk (%)", ascending=False
    ).copy()

    display["Customer Return Rate"] = (
        display["Customer Return Rate"] * 100
    ).round(1)

    st.dataframe(
        display[[
            "Order ID", "Order Value", "Product Cost",
            "Payment Method", "Customer Return Rate",
            "Distance (km)", "Previous Orders", "Shipping Cost",
            "RTO Risk (%)", "Risk Category", "Delivery Profit",
            "Estimated RTO Cost", "Expected Profit",
            "Recommended Action"
        ]],
        use_container_width=True,
        hide_index=True,
        height=400
    )

    st.download_button(
        "Download analyzed orders CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="profitpilot_analyzed_orders.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.warning("No orders match the selected filters.")

# ---------- TRANSPARENCY ----------
with st.expander("Model details & limitations"):
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", "Random Forest")
    m2.metric("Synthetic training records", "2,400")
    m3.metric("Synthetic test accuracy", f"{test_accuracy:.1%}")

    st.warning(
        "This prototype uses synthetic training data and synthetic labels. "
        "Its test accuracy does not establish real-world RTO prediction "
        "accuracy. Uploaded orders are scored by this existing model; "
        "the model is not retrained on the uploaded CSV. Profit calculations "
        "are simplified estimates, not accounting-grade results. "
        "Validate against historical orders and actual outcomes before "
        "operational use."
    )

st.markdown("""
<div class="footer">
<b>ProfitPilot</b> | Built by Team DivergeX<br>
Profit-aware delivery decisions | Hackathon prototype
</div>
""", unsafe_allow_html=True)
