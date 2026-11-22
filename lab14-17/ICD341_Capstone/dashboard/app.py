import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Zambia Maize Production Dashboard", layout="wide")
st.title("Zambia Maize Production Analysis Dashboard")
st.write("ICD341 Capstone Project: Statistical Analysis, Modeling, and Visualization.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "maize_production.csv")

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Run `python src/analysis.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# Sidebar Filters
st.sidebar.header("Filter Data")
districts = st.sidebar.multiselect(
    "Select Districts:", options=df["District"].unique(), default=list(df["District"].unique())
)
year_range = st.sidebar.slider(
    "Select Year Range:",
    int(df["Year"].min()),
    int(df["Year"].max()),
    (int(df["Year"].min()), int(df["Year"].max())),
)

filtered = df[(df["District"].isin(districts)) & (df["Year"].between(*year_range))]

# KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Production", f"{filtered['Production_Tonnes'].sum():,.1f} tonnes")
col2.metric("Average Production", f"{filtered['Production_Tonnes'].mean():,.1f} tonnes")
col3.metric("Average Rainfall", f"{filtered['Rainfall_mm'].mean():,.1f} mm")

# Charts Row
c1, c2 = st.columns(2)

with c1:
    st.subheader("Mean Production by District")
    fig1, ax1 = plt.subplots()
    filtered.groupby("District")["Production_Tonnes"].mean().sort_values().plot(
        kind="barh", ax=ax1, color="seagreen"
    )
    ax1.set_xlabel("Production (tonnes)")
    st.pyplot(fig1)

with c2:
    st.subheader("Annual Total Production Trend")
    fig2, ax2 = plt.subplots()
    filtered.groupby("Year")["Production_Tonnes"].sum().plot(
        kind="line", marker="o", ax=ax2, color="darkorange"
    )
    ax2.set_ylabel("Total Production (tonnes)")
    ax2.set_xticks(sorted(filtered["Year"].unique()))
    st.pyplot(fig2)

# Interactive Yield Predictor
st.subheader("Interactive Maize Production Predictor")
model = LinearRegression().fit(df[["Area_Hectares", "Rainfall_mm"]], df["Production_Tonnes"])

p1, p2 = st.columns(2)
input_area = p1.number_input("Planted Area (Hectares):", min_value=10.0, max_value=500.0, value=150.0)
input_rain = p2.number_input("Annual Rainfall (mm):", min_value=300.0, max_value=1500.0, value=850.0)

pred_tonnes = model.predict(pd.DataFrame([[input_area, input_rain]], columns=["Area_Hectares", "Rainfall_mm"]))[0]
st.success(f"Predicted Maize Production: **{pred_tonnes:,.2f} tonnes**")

# Raw Data Table
st.subheader("Filtered Dataset")
st.dataframe(filtered, use_container_width=True)