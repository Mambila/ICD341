import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "maize_production.csv")
OUT_PATH = os.path.join(BASE_DIR, "outputs", "graphs.png")


def generate_dataset(seed=42):
    """Generate a reproducible Zambian district maize production dataset."""
    np.random.seed(seed)
    districts = ["Lusaka", "Chongwe", "Kabwe", "Chipata", "Mumbwa", "Choma", "Mkushi", "Petauke"]
    years = [2021, 2022, 2023, 2024, 2025]
    rows = []

    for district in districts:
        base_area = np.random.uniform(90, 220)  # hectares
        for year in years:
            rainfall = np.random.normal(850, 110)  # mm
            area = base_area + np.random.normal(0, 12)
            production = 2.75 * area + 0.32 * rainfall + np.random.normal(0, 22)
            rows.append({
                "District": district,
                "Year": year,
                "Area_Hectares": round(max(area, 20), 2),
                "Rainfall_mm": round(max(rainfall, 300), 1),
                "Production_Tonnes": round(max(production, 50), 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset to: {DATA_PATH} ({len(df)} records)")
    return df


def run_analysis():
    # 1. Load or generate data
    df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else generate_dataset()

    # 2. Data cleaning check
    df = df.dropna().drop_duplicates()

    # 3. Descriptive statistics
    print("\n=== Descriptive Statistics ===")
    print(df[["Area_Hectares", "Rainfall_mm", "Production_Tonnes"]].describe().round(2))

    print("\n=== Average Production by District (Tonnes) ===")
    district_summary = df.groupby("District")["Production_Tonnes"].agg(["mean", "std", "min", "max"]).round(2)
    print(district_summary)

    # 4. Multiple Linear Regression Model
    X = df[["Area_Hectares", "Rainfall_mm"]]
    y = df["Production_Tonnes"]

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))

    print("\n=== Linear Regression Model ===")
    print(f"Intercept:          {model.intercept_:.4f}")
    print(f"Area Coefficient:   {model.coef_[0]:.4f} tonnes/hectare")
    print(f"Rainfall Coeff.:    {model.coef_[1]:.4f} tonnes/mm")
    print(f"R-squared:          {r2:.4f}")
    print(f"RMSE:               {rmse:.2f} tonnes")

    # 5. Visualizations (saved to outputs/graphs.png)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1: Mean Production by District
    df.groupby("District")["Production_Tonnes"].mean().sort_values().plot(
        kind="barh", ax=axes[0], color="seagreen"
    )
    axes[0].set_title("Mean Maize Production by District")
    axes[0].set_xlabel("Production (tonnes)")

    # Chart 2: Annual Total Production Trend
    df.groupby("Year")["Production_Tonnes"].sum().plot(
        kind="line", marker="o", ax=axes[1], color="darkorange", linewidth=2
    )
    axes[1].set_title("Total Annual Maize Production")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Total Production (tonnes)")
    axes[1].set_xticks(sorted(df["Year"].unique()))

    # Chart 3: Actual vs Predicted
    axes[2].scatter(y, predictions, color="steelblue", alpha=0.75)
    axes[2].plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2)
    axes[2].set_title(f"Actual vs. Predicted (R² = {r2:.3f})")
    axes[2].set_xlabel("Actual Production (tonnes)")
    axes[2].set_ylabel("Predicted Production (tonnes)")

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300)
    print(f"\nSaved graphs to: {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    run_analysis()