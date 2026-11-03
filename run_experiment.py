"""Frugal AI feasibility experiment — original frozen pipeline (Oct 2026), reconstructed verbatim.
Only change: load() also accepts a .parquet copy of the same xlsx for speed."""
import sys, os, time, json, pickle, tracemalloc, warnings
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
LOG = {}
def load(path):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        sheets = pd.read_excel(path, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    df = df.rename(columns={"InvoiceNo": "Invoice", "UnitPrice": "Price", "CustomerID": "CustomerID", "Customer ID": "CustomerID"})
    return df
raw = load(SRC)
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
raw["Invoice"] = raw["Invoice"].astype(str)
raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
clean_log = {"raw_rows": int(len(raw))}
df = raw.copy()
m = df["Invoice"].str.upper().str.startswith("C"); clean_log["cancelled"] = int(m.sum())
returns = df[m].copy()
df = df[~m]
m = (df["Quantity"] <= 0) | (df["Price"] <= 0); clean_log["nonpositive_qty_or_price"] = int(m.sum()); df = df[~m]
NONPRODUCT = {"POST", "DOT", "M", "D", "C2", "S", "B", "CRUK", "PADS", "BANK CHARGES", "AMAZONFEE",
              "ADJUST", "ADJUST2", "TEST001", "TEST002", "GIFT"}
m = df["StockCode"].isin(NONPRODUCT) | df["StockCode"].str.match(r"^(GIFT|TEST|ADJUST)")
clean_log["non_product_codes"] = int(m.sum()); df = df[~m]
clean_log["clean_rows"] = int(len(df))
LOG["cleaning"] = clean_log
def resource(fn):
    tracemalloc.start(); t = time.perf_counter(); r = fn(); dt = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
    return r, dt, peak
def model_size_mb(model): return len(pickle.dumps(model)) / 2**20
def wape(y, p): return np.sum(np.abs(y - p)) / np.sum(np.abs(y))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mape_nz(y, p):
    nz = y != 0
    return float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])))
df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
last_full = df["Week"].max() - pd.Timedelta(weeks=1)
dfw = df[df["Week"] <= last_full]
weeks = pd.date_range(dfw["Week"].min(), last_full, freq="7D")
wk = dfw.groupby(["StockCode", "Week"])["Quantity"].sum().unstack(fill_value=0).reindex(columns=weeks, fill_value=0)
cover = (wk > 0).mean(axis=1)
eligible = wk[cover >= 0.80]
top = eligible.sum(axis=1).sort_values(ascending=False).head(50).index
Y = wk.loc[top]
LOG["h1_setup"] = {"n_weeks": int(len(weeks)), "eligible_products": int(len(eligible)), "products_used": int(len(top))}
long = Y.stack().rename("y").reset_index().rename(columns={"level_1": "Week"})
long.columns = ["StockCode", "Week", "y"]
long = long.sort_values(["StockCode", "Week"])
g = long.groupby("StockCode")["y"]
for L in (1, 2, 3, 4):
    long[f"lag{L}"] = g.shift(L)
FIX = os.environ.get("FIX_MA", "1") == "1"   # corrected by default; FIX_MA=0 reproduces the original indexing error
if FIX:
    long["ma4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    long["ma8"] = g.transform(lambda s: s.shift(1).rolling(8).mean())
else:
    long["ma4"] = g.shift(1).rolling(4).mean().reset_index(level=0, drop=True)
    long["ma8"] = g.shift(1).rolling(8).mean().reset_index(level=0, drop=True)
long["lag52"] = g.shift(52)
long["weekofyear"] = long["Week"].dt.isocalendar().week.astype(int)
long["month"] = long["Week"].dt.month
long["peak"] = long["month"].isin([11, 12]).astype(int)
long["item"] = long["StockCode"].astype("category").cat.codes
FEATS = ["lag1", "lag2", "lag3", "lag4", "ma4", "ma8", "weekofyear", "month", "peak", "item"]
TEST_WEEKS = weeks[-26:]
RETRAIN_EVERY = 4
def make_models():
    return {"RF": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1),
            "XGB": xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=SEED, n_jobs=1, verbosity=0)}
def run_h1(history_weeks=None):
    res_t = {"RF":0.0,"XGB":0.0}; res_m = {"RF":0.0,"XGB":0.0}; size = {}
    rows = []; models = None
    for i, w in enumerate(TEST_WEEKS):
        if i % RETRAIN_EVERY == 0:
            tr = long[(long["Week"] < w)].dropna(subset=FEATS)
            if history_weeks is not None:
                tr = tr[tr["Week"] >= w - pd.Timedelta(weeks=history_weeks)]
            models = make_models()
            for k, mdl in models.items():
                _, dt, pk = resource(lambda: mdl.fit(tr[FEATS], tr["y"]))
                res_t[k] += dt; res_m[k] = max(res_m[k], pk); size[k] = model_size_mb(mdl)
        te = long[long["Week"] == w]
        out = te[["StockCode", "Week", "y", "lag1", "ma4", "lag52"]].copy()
        for k, mdl in models.items():
            out[k] = np.clip(mdl.predict(te[FEATS]), 0, None)
        rows.append(out)
    P = pd.concat(rows)
    P["Naive"] = P["lag1"]; P["MovingAvg"] = P["ma4"]; P["SeasonalNaive"] = P["lag52"]
    return P, res_t, res_m, size
def mase_scale(code):
    s = Y.loc[code]; s = s[s.index < TEST_WEEKS[0]]
    return np.mean(np.abs(np.diff(s.values))) or 1.0
def score_h1(P, methods):
    res = {}
    for k in methods:
        sub = P.dropna(subset=[k]); y, p = sub["y"].values, sub[k].values
        mase = np.mean([np.mean(np.abs(g_["y"] - g_[k])) / mase_scale(c) for c, g_ in sub.groupby("StockCode")])
        res[k] = {"WAPE": round(float(wape(y, p)),4), "RMSE": round(rmse(y, p),1), "MASE": round(float(mase),3), "MAPE_nonzero": round(mape_nz(y, p),3)}
    return res
def per_item_wape(P, k):
    return P.groupby("StockCode").apply(lambda g_: wape(g_["y"].values, g_[k].values))
P_full, t_full, m_full, s_full = run_h1(None)
P_full.to_pickle(os.path.join(OUT,"P_full.pkl"))
METH = ["Naive", "MovingAvg", "SeasonalNaive", "RF", "XGB"]
h1 = score_h1(P_full, METH)
best_base = min(["Naive", "MovingAvg", "SeasonalNaive"], key=lambda k: h1[k]["WAPE"])
best_ml = min(["RF", "XGB"], key=lambda k: h1[k]["WAPE"])
impr = 1 - h1[best_ml]["WAPE"] / h1[best_base]["WAPE"]
a, b = per_item_wape(P_full, best_ml), per_item_wape(P_full, best_base)
stat, pval = wilcoxon(a, b)
LOG["h1_full"] = {"scores": h1, "best_baseline": best_base, "best_model": best_ml, "rel_impr": round(float(impr),4),
                  "wilcoxon_p": round(float(pval),4), "items_improved": int((a < b).sum()), "train_time": t_full, "size": s_full}
print(json.dumps(LOG, indent=1, default=float)); sys.stdout.flush()
if os.environ.get("H1_ONLY")=="1": sys.exit()
cust = df.dropna(subset=["CustomerID"]).copy()
cust["Revenue"] = cust["Quantity"] * cust["Price"]
ret = returns.dropna(subset=["CustomerID"]).copy()
def snapshot(cutoff, hist_months=None):
    h = cust[cust["InvoiceDate"] < cutoff]; r = ret[ret["InvoiceDate"] < cutoff]
    if hist_months:
        lo = cutoff - pd.DateOffset(months=hist_months); h = h[h["InvoiceDate"] >= lo]; r = r[r["InvoiceDate"] >= lo]
    if h.empty: return None
    gb = h.groupby("CustomerID")
    f = pd.DataFrame({"recency": (cutoff - gb["InvoiceDate"].max()).dt.days, "frequency": gb["Invoice"].nunique(),
        "monetary": gb["Revenue"].sum(), "tenure": (cutoff - gb["InvoiceDate"].min()).dt.days, "distinct_items": gb["StockCode"].nunique()})
    f["avg_basket"] = f["monetary"] / f["frequency"]
    f["recent90"] = h[h["InvoiceDate"] >= cutoff - pd.Timedelta(days=90)].groupby("CustomerID")["Invoice"].nunique().reindex(f.index, fill_value=0)
    f["return_ratio"] = (r.groupby("CustomerID")["Invoice"].nunique().reindex(f.index, fill_value=0) / f["frequency"])
    fut = cust[(cust["InvoiceDate"] >= cutoff) & (cust["InvoiceDate"] < cutoff + pd.Timedelta(days=90))]["CustomerID"].unique()
    f["churn"] = (~f.index.isin(fut)).astype(int); f["cutoff"] = cutoff
    return f
CF = ["recency", "frequency", "monetary", "tenure", "distinct_items", "avg_basket", "recent90", "return_ratio"]
TRAIN_CUTS = pd.to_datetime(["2010-03-01", "2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01"])
TEST_CUTS = pd.to_datetime(["2011-06-01", "2011-09-01"])
tr = pd.concat([snapshot(c) for c in TRAIN_CUTS]); te = pd.concat([snapshot(c) for c in TEST_CUTS])
tr.to_pickle(os.path.join(OUT,"churn_tr.pkl")); te.to_pickle(os.path.join(OUT,"churn_te.pkl"))
y = te["churn"].values
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)).fit(tr[CF], tr["churn"])
print("n_test",len(te),"churn",y.mean().round(3),"LR AUC",round(roc_auc_score(y, lr.predict_proba(te[CF])[:,1]),3),"rule AUC",round(roc_auc_score(y, te["recency"]),3))
