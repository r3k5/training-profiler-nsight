import os, re, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"results\nsys_csv"; CSV=os.path.join(OUT,"gpukernsum.csv"); PNG=os.path.join(OUT,"top10_kernels.png")
if not os.path.exists(CSV): raise SystemExit(f"Missing {CSV}")

df=pd.read_csv(CSV)
def coerce_num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False)
                           .str.replace(r"[^0-9eE+\-\.]", "", regex=True), errors="coerce")

name_col = next((c for c in df.columns if re.search(r"(kernel|name|symbol|function|k)$", c, re.I)), None)
time_col = next((c for c in df.columns if re.search(r"(total.*time|duration|elapsed|t$)", c, re.I)), None)
if not name_col or not time_col:
    print("Columns:", list(df.columns)); raise SystemExit("Could not detect kernel name/time columns.")

df["_t"]=coerce_num(df[time_col])
top=df[[name_col,"_t"]].dropna().sort_values("_t", ascending=False).head(10)
if top.empty: raise SystemExit("No rows to plot after numeric conversion.")

plt.figure(); plt.barh(top[name_col], top["_t"]); plt.gca().invert_yaxis()
plt.xlabel(time_col); plt.title("Top 10 GPU Kernels by Time"); plt.tight_layout()
os.makedirs(OUT, exist_ok=True); plt.savefig(PNG); print("Wrote", PNG)
