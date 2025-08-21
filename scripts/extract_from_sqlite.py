# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Mike Davis

import sys, os, sqlite3, csv
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: extract_from_sqlite.py <nsys_report.sqlite> <outdir>")
    sys.exit(1)

db_path = Path(sys.argv[1])
out_dir  = Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(str(db_path))
cur = con.cursor()

def table_cols(table):
    cur.execute(f"PRAGMA table_info('{table}')")
    return [r[1] for r in cur.fetchall()]

def pick(cols, needles):
    cl = [c.lower() for c in cols]
    for n in needles:
        for i,c in enumerate(cl):
            if n in c:
                return cols[i]
    return None

def write_csv(path, rows, header):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header); w.writerows(rows)

# --- discover all tables ---
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
tl = [t.lower() for t in tables]

# =========================
# 1) GPU kernel summary
# =========================
kernel_tbl = None
for t in tables:
    l = t.lower()
    if "kernel" in l and ("cupti" in l or "cuda" in l or "gpu" in l):
        kernel_tbl = t; break
if kernel_tbl is None:
    for t in tables:
        if "kernel" in t.lower():
            kernel_tbl = t; break

kern_rows, kern_hdr = [], []
if kernel_tbl:
    c = table_cols(kernel_tbl)
    name = pick(c, ["demangled","shortname","name","symbol","function"])
    dur  = pick(c, ["duration","elapsed","time","msec","usec","nsec","ns"])
    start= pick(c, ["start"])
    end  = pick(c, ["end","stop","finish"])
    if not dur and start and end:
        q = f'SELECT {name} AS k, SUM({end}-{start}) AS t FROM "{kernel_tbl}" GROUP BY {name} ORDER BY t DESC LIMIT 200'
    else:
        q = f'SELECT {name} AS k, SUM({dur}) AS t FROM "{kernel_tbl}" GROUP BY {name} ORDER BY t DESC LIMIT 200'
    try:
        cur.execute(q); kern_rows = cur.fetchall(); kern_hdr = ["Kernel","TotalTime"]
    except Exception as e:
        print("Kernel query failed:", e)

if kern_rows:
    write_csv(out_dir/"gpukernsum.csv", kern_rows, kern_hdr)
    print("Wrote", out_dir/"gpukernsum.csv")
else:
    print("Could not extract kernel summary; tables:", tables)

# =========================
# 2) CUDA API summary
# =========================
# Prefer a CUPTI runtime table; fall back to any "*RUNTIME*" table.
api_tbl = None
for t in tables:
    if "cupti_activity_kind_runtime" in t.lower():
        api_tbl = t; break
if api_tbl is None:
    for t in tables:
        if "runtime" in t.lower():
            api_tbl = t; break

api_rows, api_hdr = [], []
if api_tbl:
    c = table_cols(api_tbl)
    # Try to find a text name column directly
    name_col = pick(c, ["apiname","name","api","function"])
    # Duration or start/end
    dur  = pick(c, ["duration","elapsed","time","msec","usec","nsec","ns"])
    start= pick(c, ["start"])
    end  = pick(c, ["end","stop","finish"])

    if not dur and start and end:
        dur_expr = f"({end}-{start})"
    elif dur:
        dur_expr = dur
    else:
        dur_expr = None

    if name_col and dur_expr:
        q = f'SELECT {name_col} AS api, SUM({dur_expr}) AS t, COUNT(*) AS calls FROM "{api_tbl}" GROUP BY {name_col} ORDER BY t DESC LIMIT 200'
        try:
            cur.execute(q); api_rows = cur.fetchall(); api_hdr = ["CUDA_API","TotalTime","Calls"]
        except Exception as e:
            print("Direct API query failed:", e)

    if not api_rows:
        # Many builds store the API name as a string id; try to join StringIds.
        if "StringIds" in tables:
            sc = table_cols("StringIds")
            sid_col = pick(sc, ["sid","id","stringid"])
            text_col= pick(sc, ["text","string","name","value"])
        else:
            sid_col = text_col = None

        # Find a SID-like column in the runtime table (e.g., NameSid, ApiSid, etc.)
        sid_in_runtime = pick(c, ["sid","stringid","namesid","apisid"])
        if sid_in_runtime and sid_col and text_col and dur_expr:
            q = f'''
                SELECT S.{text_col} AS api, SUM({dur_expr}) AS t, COUNT(*) AS calls
                FROM "{api_tbl}" R
                JOIN "StringIds" S ON R.{sid_in_runtime} = S.{sid_col}
                GROUP BY S.{text_col}
                ORDER BY t DESC
                LIMIT 200
            '''
            try:
                cur.execute(q); api_rows = cur.fetchall(); api_hdr = ["CUDA_API","TotalTime","Calls"]
            except Exception as e:
                print("StringIds-join API query failed:", e)

if api_rows:
    write_csv(out_dir/"cudaapisum.csv", api_rows, api_hdr)
    print("Wrote", out_dir/"cudaapisum.csv")
else:
    print("Could not extract CUDA API summary; tables:", tables)

con.close()
