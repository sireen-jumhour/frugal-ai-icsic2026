import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
raw = pd.read_parquet("raw.parquet")
raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
raw["Invoice"] = raw["Invoice"].astype(str)
raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
df = raw.copy()
m = df["Invoice"].str.upper().str.startswith("C"); returns = df[m].copy(); df = df[~m]
df = df[~((df["Quantity"] <= 0) | (df["Price"] <= 0))]
NONPRODUCT = {"POST","DOT","M","D","C2","S","B","CRUK","PADS","BANK CHARGES","AMAZONFEE","ADJUST","ADJUST2","TEST001","TEST002","GIFT"}
df = df[~(df["StockCode"].isin(NONPRODUCT) | df["StockCode"].str.match(r"^(GIFT|TEST|ADJUST)"))]
df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
last_full = df["Week"].max() - pd.Timedelta(weeks=1)
dfw = df[df["Week"] <= last_full]
weeks = pd.date_range(dfw["Week"].min(), last_full, freq="7D")
wk = dfw.groupby(["StockCode","Week"])["Quantity"].sum().unstack(fill_value=0).reindex(columns=weeks, fill_value=0)
eligible = wk[(wk > 0).mean(axis=1) >= 0.80]
top = eligible.sum(axis=1).sort_values(ascending=False).head(50).index
Y = wk.loc[top]
long = Y.stack().rename("y").reset_index(); long.columns = ["StockCode","Week","y"]
long = long.sort_values(["StockCode","Week"]).reset_index(drop=True)
g = long.groupby("StockCode")["y"]
for L in (1,2,3,4): long[f"lag{L}"] = g.shift(L)
long["ma4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())   # corrected
long["ma8"] = g.transform(lambda s: s.shift(1).rolling(8).mean())   # corrected
long["lag52"] = g.shift(52)
long["weekofyear"] = long["Week"].dt.isocalendar().week.astype(int)
long["month"] = long["Week"].dt.month
long["peak"] = long["month"].isin([11,12]).astype(int)
long["item"] = long["StockCode"].astype("category").cat.codes
FEATS = ["lag1","lag2","lag3","lag4","ma4","ma8","weekofyear","month","peak","item"]
TEST_WEEKS = weeks[-26:]
def wape(y,p): return np.sum(np.abs(y-p))/np.sum(np.abs(y))
def mase_scale(code):
    s = Y.loc[code]; s = s[s.index < TEST_WEEKS[0]]
    return np.mean(np.abs(np.diff(s.values))) or 1.0
def scores(P,k):
    sub=P.dropna(subset=[k]); y,p=sub.y.values,sub[k].values; nz=y!=0
    return dict(WAPE=wape(y,p),RMSE=float(np.sqrt(np.mean((y-p)**2))),
      MASE=float(np.mean([np.mean(np.abs(g_.y-g_[k]))/mase_scale(c) for c,g_ in sub.groupby("StockCode")])),
      MAPE=float(np.mean(np.abs((y[nz]-p[nz])/y[nz]))))
def per_item(P,k): return P.groupby("StockCode").apply(lambda g_: wape(g_.y.values,g_[k].values))
def walk(make, history_weeks=None, feats=FEATS):
    rows=[]; models=None; t=0.0
    import time
    for i,w in enumerate(TEST_WEEKS):
        if i%4==0:
            tr=long[long.Week<w].dropna(subset=feats)
            if history_weeks: tr=tr[tr.Week>=w-pd.Timedelta(weeks=history_weeks)]
            models=make(); t0=time.perf_counter()
            for mdl in models.values(): mdl.fit(tr[feats],tr.y)
            t+=time.perf_counter()-t0
        te=long[long.Week==w]; out=te[["StockCode","Week","y","lag1","ma4","lag52"]].copy()
        for k,mdl in models.items(): out[k]=np.clip(mdl.predict(te[feats]),0,None)
        rows.append(out)
    P=pd.concat(rows); P["Naive"]=P.lag1; P["MovingAvg"]=P.ma4; P["SeasonalNaive"]=P.lag52
    return P,t
# Croston family, one-step-ahead, fitted on all history before each test week
def intermittent(alpha=0.1, beta=0.1):
    out=[]
    for code in Y.index:
        y=Y.loc[code].values.astype(float); n=len(y); T0=n-26
        # Croston/SBA
        z=None;p=None;q=1; cro=np.full(n,np.nan); tsb=np.full(n,np.nan)
        first=np.argmax(y>0); z=y[first]; p=1.0; q=1
        # TSB init
        zt=y[first]; pt=float((y[:T0]>0).mean())
        for t in range(first+1,n):
            cro[t]=z/p; tsb[t]=pt*zt
            if y[t]>0:
                z=z+alpha*(y[t]-z); p=p+alpha*(q-p); q=1
                zt=zt+alpha*(y[t]-zt); pt=pt+beta*(1-pt)
            else:
                q+=1; pt=pt+beta*(0-pt)
        for t in range(T0,n):
            out.append(dict(StockCode=code,Week=Y.columns[t],y=y[t],Croston=cro[t],SBA=cro[t]*(1-alpha/2),TSB=tsb[t]))
    return pd.DataFrame(out)
