from common import *
import json, xgboost as xgb, time
from sklearn.ensemble import RandomForestRegressor
rng=np.random.default_rng(2026)
VAL_START=TEST_WEEKS[0]-pd.Timedelta(weeks=8)
trv=long[long.Week<VAL_START].dropna(subset=FEATS); va=long[(long.Week>=VAL_START)&(long.Week<TEST_WEEKS[0])]
def vw(p): return wape(va.y.values,np.clip(p,0,None))
BUDGET=20
# RF / XGB random search (equal budget)
rf_space=[dict(n_estimators=int(rng.choice([100,200,300])),max_depth=[None,5,10,20][int(rng.integers(4))],min_samples_leaf=int(rng.choice([1,2,5,10,20])),max_features=float(rng.choice([1.0,0.5,0.33]))) for _ in range(BUDGET)]
xg_space=[dict(n_estimators=int(rng.choice([100,200,400])),max_depth=int(rng.choice([2,3,4,6])),learning_rate=float(rng.choice([0.03,0.05,0.1,0.3])),subsample=float(rng.choice([0.7,0.85,1.0])),min_child_weight=int(rng.choice([1,5,10]))) for _ in range(BUDGET)]
res={"RF":[], "XGB":[]}
t0=time.time()
for c in rf_space:
    m=RandomForestRegressor(random_state=42,n_jobs=1,**c).fit(trv[FEATS],trv.y); res["RF"].append(vw(m.predict(va[FEATS])))
for c in xg_space:
    m=xgb.XGBRegressor(random_state=42,n_jobs=1,verbosity=0,**c).fit(trv[FEATS],trv.y); res["XGB"].append(vw(m.predict(va[FEATS])))
best_rf=rf_space[int(np.argmin(res["RF"]))]; best_xg=xg_space[int(np.argmin(res["XGB"]))]
# baselines tuned on same validation window: MA window, SBA alpha
mav={}
for w_ in range(2,13):
    preds=[]; ys=[]
    for code in Y.index:
        y=Y.loc[code].values.astype(float); idx=[list(Y.columns).index(w) for w in sorted(va.Week.unique())]
        for t in idx: preds.append(y[t-w_:t].mean()); ys.append(y[t])
    mav[w_]=wape(np.array(ys),np.array(preds))
best_ma=min(mav,key=mav.get)
print("tuning secs",round(time.time()-t0),"bestRF",best_rf,"bestXGB",best_xg,"bestMA",best_ma,{k:round(v,3) for k,v in mav.items()},flush=True)
# MA with tuned window on test
long["maT"]=long.groupby("StockCode")["y"].transform(lambda s: s.shift(1).rolling(best_ma).mean())
out={"tuning":{"budget_per_model":BUDGET,"val_weeks":8,"best_rf":best_rf,"best_xgb":best_xg,"best_ma_window":int(best_ma),
     "val_wape":{"RF_best":float(min(res["RF"])),"XGB_best":float(min(res["XGB"])),"MA_best":float(mav[best_ma])}}}
# seeds
SEEDS=list(range(10))
seed_res={"RF_default":[],"XGB_default":[],"RF_tuned":[],"XGB_tuned":[]}
for s in SEEDS:
    P,_=walk(lambda:{"RF_default":RandomForestRegressor(n_estimators=100,random_state=s,n_jobs=1),
                    "XGB_default":xgb.XGBRegressor(n_estimators=200,max_depth=6,random_state=s,n_jobs=1,verbosity=0),
                    "RF_tuned":RandomForestRegressor(random_state=s,n_jobs=1,**best_rf),
                    "XGB_tuned":xgb.XGBRegressor(random_state=s,n_jobs=1,verbosity=0,**best_xg)})
    for k in seed_res: seed_res[k].append(wape(P.y.values,P[k].values))
    if s==0:
        tl=long[long.Week.isin(TEST_WEEKS)]; out["MA_tuned_test_WAPE"]=float(wape(tl.y.values,tl.maT.values))
    print(s,{k:round(v[-1],4) for k,v in seed_res.items()},flush=True)
out["seeds"]={k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)),min=float(min(v)),max=float(max(v))) for k,v in seed_res.items()}
json.dump(out,open("h1_ext.json","w"),indent=1,default=float); print(json.dumps(out,indent=1,default=float))
