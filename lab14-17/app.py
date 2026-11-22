import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 1. Dashboard Title & Introduction
st.title("Zambia Maize Production Dashboard")
st.write("This dashboard analyses maize production in selected districts.")

# 2. Create Dataset (Exercise 2)
data = {
    "District": ["Lusaka", "Chongwe", "Kabwe", "Chipata", "Mumbwa"],
    "Production": [500, 350, 450, 600, 400]
}
df = pd.DataFrame(data)

# 3. Display Dataset & Statistics (Exercise 3)
st.subheader("Maize Production Data")
st.dataframe(df)

st.subheader("Production Statistics")
col1, col2 = st.columns(2)
col1.metric("Total Production", f"{df['Production'].sum():,} tonnes")
col2.metric("Average Production", f"{df['Production'].mean():.1f} tonnes")

# 4. Bar Chart Visualization (Exercise 3)
st.subheader("Production by District")
fig, ax = plt.subplots()
ax.bar(df["District"], df["Production"], color="steelblue")
ax.set_xlabel("District")
ax.set_ylabel("Production (tonnes)")
ax.set_title("Maize Production Comparison")
plt.xticks(rotation=30)
st.pyplot(fig)

# 5. Interactive District Selection (Exercise 4)
st.subheader("Select a District")
selected_district = st.selectbox("Choose a district:", df["District"])
result = df[df["District"] == selected_district]

st.write("**Selected District:**", selected_district)
st.write("**Maize Production:**", f"{result['Production'].iloc[0]} tonnes")