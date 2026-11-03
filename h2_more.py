import pandas as pd,numpy as np,json,time,pickle,xgboost as xgb,warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
tr=pd.read_pickle("out_orig/churn_tr.pkl"); te=pd.read_pickle("out_orig/churn_te.pkl")
CF=["recency","frequency","monetary","tenure","distinct_items","avg_basket","recent90","return_ratio"]
y=te.churn.values; uc,inv=np.unique(te.index.values,return_inverse=True)
groups=[np.where(inv==i)[0] for i in range(len(uc))]
def ci(s1,s2,n=1000):
    rng=np.random.default_rng(42); d=[]
    for _ in range(n):
        idx=np.concatenate([groups[i] for i in rng.integers(0,len(uc),len(uc))]); d.append(roc_auc_score(y[idx],s1[idx])-roc_auc_score(y[idx],s2[idx]))
    return [round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4),round(float(np.mean(np.array(d)>0)),3)]
qs=lambda c: np.quantile(tr[c],[.2,.4,.6,.8])
def q(c): return np.searchsorted(qs(c),te[c].values,side="right")+1
rfm=(q("recency")+(6-q("frequency"))+(6-q("monetary"))).astype(float)
def timed(m):
    t=time.perf_counter(); m.fit(tr[CF],tr.churn); return m,time.perf_counter()-t
lr,tl=timed(make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)))
rf,trf=timed(RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_leaf=5,max_features=0.33,random_state=42,n_jobs=1))
xg,txg=timed(xgb.XGBClassifier(n_estimators=200,max_depth=2,learning_rate=0.03,subsample=0.85,min_child_weight=1,random_state=42,n_jobs=1,verbosity=0))
s={k:m.predict_proba(te[CF])[:,1] for k,m in [("lr",lr),("rf",rf),("xg",xg)]}
out={"AUC":{k:round(roc_auc_score(y,v),4) for k,v in s.items()},
 "size_KB":{k:round(len(pickle.dumps(m))/1024,1) for k,m in [("lr",lr),("rf",rf),("xg",xg)]},
 "train_s":{"lr":round(tl,3),"rf":round(trf,2),"xg":round(txg,2)},
 "CI":{"LR8_vs_RFMscore":ci(s["lr"],rfm),"RFtuned_vs_LR8":ci(s["rf"],s["lr"]),"XGBtuned_vs_LR8":ci(s["xg"],s["lr"]),"RFtuned_vs_rule":ci(s["rf"],te.recency.values.astype(float))}}
json.dump(out,open("h2_more.json","w"),indent=1); print(json.dumps(out,indent=1))
