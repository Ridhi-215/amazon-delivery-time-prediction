# src/02_data_cleaning.py
from pathlib import Path
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "amazon_delivery.csv"
OUTPUT_PATH = ROOT / "data" / "processed_data.csv"

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print("Initial Shape:", df.shape)

    # --- 1) Remove duplicates ---
    df.drop_duplicates(inplace=True)
    print("After removing duplicates:", df.shape)

    # --- 2) Handle missing values ---
    # Example: fill missing Agent_Rating with mean
    if "Agent_Rating" in df.columns:
        df["Agent_Rating"].fillna(df["Agent_Rating"].mean(), inplace=True)

    # Drop rows where critical info (store/drop location or delivery time) is missing
    critical_cols = ["Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude", "Delivery_Time"]
    df.dropna(subset=critical_cols, inplace=True)

    # Fill remaining categorical NaN with "Unknown"
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col].fillna("Unknown", inplace=True)

    print("After handling missing values:", df.shape)

    # --- 3) Convert date/time columns ---
    time_cols = ["Order_Date", "Order_Time", "Pickup_Time"]
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    print("Converted time columns to datetime.")

    # --- 4) Save cleaned dataset ---
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Cleaned data saved at: {OUTPUT_PATH}")

    # Quick check
    print("\nPreview of cleaned data:")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
