# -*- coding: utf-8 -*-
import os, re, pandas as pd

OUT = r"results\nsys_csv"
CSV = os.path.join(OUT, "gpukernsum.csv")
PNG = os.path.join(OUT, "top10_kernels.png")
SNIP = os.path.join(OUT, "README_snippet.md")

def coerce_num(s):
    return pd.to_numeric(
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(r"[^0-9eE+\-\.]", "", regex=True),
        errors="coerce",
    )

df = pd.read_csv(CSV)
name_col = next((c for c in df.columns if re.search(r"(kernel|name|symbol|function|k)$", c, re.I)), None)
time_col = next((c for c in df.columns if re.search(r"(total.*time|duration|elapsed|t$)", c, re.I)), None)

df["_t"] = coerce_num(df[time_col])
top = df[[name_col, "_t"]].dropna().sort_values("_t", ascending=False).head(10)

lines = []
lines.append("## Nsight Systems profile - top kernels")
lines.append("")
png_rel = PNG.replace("\\", "/")  # forward slashes for Markdown
lines.append(f"![top kernels]({png_rel})")
lines.append("")
lines.append("| # | Kernel | Total Time (arb) |")
lines.append("|---:|---|---:|")
for i, (k, t) in enumerate(top.to_records(index=False), 1):
    lines.append(f"| {i} | `{k}` | {t:.0f} |")

with open(SNIP, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Wrote", SNIP)
