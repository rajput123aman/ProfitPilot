
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="ProfitPilot AI",
    page_icon="📦",
    layout="wide"
)

st.title("🚀 ProfitPilot AI")
st.subheader("Profit-Aware Delivery & RTO Intelligence Engine")
st.caption("Built by Team DivergeX | Hackathon Prototype")

@st.cache_resource
def build_model():
    rng = np.random.default_rng(42)
    n = 2000

    data = pd.DataFrame({
        "order_value": rng.integers(300, 5000, n),
        "is_cod": rng.integers(0, 2, n),
        "customer_return_rate": rng.uniform(0, 0.8, n),
        "distance_km": rng.integers(2, 1000, n),
        "previous_orders": rng.integers(0, 30, n)
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

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)

    score = accuracy_score(y_test, model.predict(X_test))
    return model, score

model, test_accuracy = build_model()

# Generate demo orders
@st.cache_data
def make_orders():
    rng = np.random.default_rng(10)
    n = 100

    df = pd.DataFrame({
        "Order ID": [f"ORD-{i:03}" for i in range(1, n + 1)],
        "Order Value (₹)": rng.integers(300, 5000, n),
        "Product Cost (₹)": rng.integers(150, 2500, n),
        "COD": rng.integers(0, 2, n),
        "Customer Return Rate": rng.uniform(0, 0.8, n).round(2),
        "Distance (km)": rng.integers(2, 1000, n),
        "Previous Orders": rng.integers(0, 30, n),
        "Shipping Cost (₹)": rng.integers(40, 150, n)
    })
    return df

orders = make_orders()

features = pd.DataFrame({
    "order_value": orders["Order Value (₹)"],
    "is_cod": orders["COD"],
    "customer_return_rate": orders["Customer Return Rate"],
    "distance_km": orders["Distance (km)"],
    "previous_orders": orders["Previous Orders"]
})

orders["RTO Risk (%)"] = (
    model.predict_proba(features)[:, 1] * 100
).round(1)

# Simplified profit estimates
orders["Delivery Profit (₹)"] = (
    orders["Order Value (₹)"]
    - orders["Product Cost (₹)"]
    - orders["Shipping Cost (₹)"] - 20
)

orders["RTO Loss (₹)"] = (
    orders["Shipping Cost (₹)"] * 2 + 20
)

risk = orders["RTO Risk (%)"] / 100

orders["Expected Profit (₹)"] = (
    (1 - risk) * orders["Delivery Profit (₹)"]
    - risk * orders["RTO Loss (₹)"]
).round(2)

def recommend(row):
    if row["RTO Risk (%)"] >= 60:
        return "Verify customer/address"
    elif row["RTO Risk (%)"] >= 35:
        return "Confirm before dispatch"
    return "Standard delivery"

orders["Recommended Action"] = orders.apply(recommend, axis=1)

# Sidebar filters
st.sidebar.header("Order Filters")
min_risk = st.sidebar.slider("Minimum RTO risk (%)", 0, 100, 0)
payment = st.sidebar.selectbox(
    "Payment method", ["All", "COD", "Prepaid"]
)

filtered = orders[orders["RTO Risk (%)"] >= min_risk].copy()

if payment == "COD":
    filtered = filtered[filtered["COD"] == 1]
elif payment == "Prepaid":
    filtered = filtered[filtered["COD"] == 0]

# KPI cards
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Orders", len(filtered))
c2.metric(
    "High-Risk Orders",
    int((filtered["RTO Risk (%)"] >= 60).sum())
)
c3.metric(
    "Expected Profit",
    f"₹{filtered['Expected Profit (₹)'].sum():,.0f}"
)
c4.metric(
    "Average RTO Risk",
    f"{filtered['RTO Risk (%)'].mean():.1f}%"
    if not filtered.empty else "N/A"
)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("RTO Risk Distribution")
    fig = px.histogram(
        filtered, x="RTO Risk (%)", nbins=15
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Expected Profit by Payment")
    if not filtered.empty:
        chart_data = filtered.copy()
        chart_data["Payment"] = np.where(
            chart_data["COD"] == 1, "COD", "Prepaid"
        )
        summary = chart_data.groupby(
            "Payment", as_index=False
        )["Expected Profit (₹)"].sum()

        fig2 = px.bar(
            summary, x="Payment", y="Expected Profit (₹)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No orders match these filters.")

st.subheader("📦 Order-Level Intelligence")
st.dataframe(
    filtered.sort_values("RTO Risk (%)", ascending=False),
    use_container_width=True,
    hide_index=True
)

with st.expander("Model details and limitations"):
    st.write("Algorithm: Random Forest Classifier")
    st.write("Training records: 2,000 simulated orders")
    st.write(f"Test accuracy on simulated labels: {test_accuracy:.1%}")
    st.warning(
        "Synthetic-data prototype only. This test score is not "
        "evidence of real-world RTO prediction accuracy. "
        "Profit uses simplified cost assumptions."
    )

st.download_button(
    "Download Order Report (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="profitpilot_orders.csv",
    mime="text/csv"
)
