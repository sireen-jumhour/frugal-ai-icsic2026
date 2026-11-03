import sys,os
sys.argv=["x","raw.parquet","out_orig"]
src=open("run_experiment.py").read()
src=src.split("P_full, t_full, m_full, s_full = run_h1(None)")[0]   # data prep only
exec(src)
cust = df.dropna(subset=["CustomerID"]).copy(); cust["Revenue"]=cust["Quantity"]*cust["Price"]
ret = returns.dropna(subset=["CustomerID"]).copy()
exec(open("run_experiment.py").read().split("ret = returns.dropna(subset=[\"CustomerID\"]).copy()")[1].split("tr = pd.concat")[0])
from sklearn.metrics import roc_auc_score
for hm in [3,6,12,None]:
    tr=pd.concat([s for s in (snapshot(c,hm) for c in TRAIN_CUTS) if s is not None]); te=pd.concat([snapshot(c,hm) for c in TEST_CUTS])
    lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)).fit(tr[CF],tr.churn)
    y=te.churn.values
    print(hm,len(te),round(roc_auc_score(y,lr.predict_proba(te[CF])[:,1]),3),round(roc_auc_score(y,te.recency),3),flush=True)
