# src/01_load_data.py
from pathlib import Path
import pandas as pd

# 1) compute project root and data path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "amazon_delivery.csv"

def main():
    print("Trying to load:", DATA_PATH)
    if not DATA_PATH.exists():
        print("ERROR: file not found. Make sure amazon_delivery.csv is in data/ folder.")
        return

    df = pd.read_csv(DATA_PATH)
    print("Loaded DataFrame with shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

if __name__ == '__main__':
    main()
