"""Convert the UCI Online Retail II workbook to raw.parquet used by the scripts.
Download: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0)."""
import sys, pandas as pd
x = pd.read_excel(sys.argv[1] if len(sys.argv) > 1 else "online_retail_II.xlsx", sheet_name=None)
df = pd.concat(x.values(), ignore_index=True)
for c in ["Invoice", "StockCode", "Description", "Country"]: df[c] = df[c].astype(str)
df.to_parquet("raw.parquet"); print(len(df), "rows written to raw.parquet")
