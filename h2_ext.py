import pandas as pd, numpy as np, json, time, xgboost as xgb, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
tr=pd.read_pickle("out_orig/churn_tr.pkl"); te=pd.read_pickle("out_orig/churn_te.pkl")
CF=["recency","frequency","monetary","tenure","distinct_items","avg_basket","recent90","return_ratio"]
RFM=["recency","frequency","monetary"]
last=tr.cutoff.max(); fit=tr[tr.cutoff<last]; val=tr[tr.cutoff==last]
y=te.churn.values; cust=te.index.values
uc,inv=np.unique(cust,return_inverse=True)
def clus_ci(s1,s2,n=1000,seed=42):
    rng=np.random.default_rng(seed); d=[]
    groups=[np.where(inv==i)[0] for i in range(len(uc))]
    for _ in range(n):
        pick=rng.integers(0,len(uc),len(uc)); idx=np.concatenate([groups[i] for i in pick])
        d.append(roc_auc_score(y[idx],s1[idx])-roc_auc_score(y[idx],s2[idx]))
    return [round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)]
out={}
rule=te.recency.values.astype(float)
LR8=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(tr[CF],tr.churn); s_lr8=LR8.predict_proba(te[CF])[:,1]
LR3=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)).fit(tr[RFM],tr.churn); s_lr3=LR3.predict_proba(te[RFM])[:,1]
# classic RFM quintile scoring rule (cutpoints from training data)
def q(col,asc):
    qs=np.quantile(tr[col],[.2,.4,.6,.8]); s=np.searchsorted(qs,te[col].values,side="right")+1
    return s if asc else 6-s
rfm_rule=(q("recency",True)+(6-q("frequency",True))+(6-q("monetary",True))).astype(float)  # higher = more likely churn
out["AUC"]={"recency_rule":roc_auc_score(y,rule),"RFM_score_rule":roc_auc_score(y,rfm_rule),"LR_RFM_only":roc_auc_score(y,s_lr3),"LR_8_features":roc_auc_score(y,s_lr8)}
out["CI_clustered"]={"LR8_vs_rule":clus_ci(s_lr8,rule),"LR8_vs_RFMscore":clus_ci(s_lr8,rfm_rule),"LR8_vs_LR_RFM":clus_ci(s_lr8,s_lr3)}
print(json.dumps(out,indent=1,default=float),flush=True)
# equal tuning budget (20 configs each), select on validation AUC (last training cutoff)
rng=np.random.default_rng(2026); B=20
lr_space=[dict(C=float(c)) for c in np.logspace(-3,2,B)]
rf_space=[dict(n_estimators=int(rng.choice([100,200,300])),max_depth=[None,4,8,12][int(rng.integers(4))],min_samples_leaf=int(rng.choice([1,5,20,50])),max_features=float(rng.choice([1.0,0.5,0.33]))) for _ in range(B)]
xg_space=[dict(n_estimators=int(rng.choice([100,200,400])),max_depth=int(rng.choice([2,3,4,6])),learning_rate=float(rng.choice([0.03,0.05,0.1,0.3])),subsample=float(rng.choice([0.7,0.85,1.0])),min_child_weight=int(rng.choice([1,5,10]))) for _ in range(B)]
mk={"LR":lambda c,s:make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,**c)),
    "RF":lambda c,s:RandomForestClassifier(random_state=s,n_jobs=1,**c),
    "XGB":lambda c,s:xgb.XGBClassifier(random_state=s,n_jobs=1,verbosity=0,**c)}
sp={"LR":lr_space,"RF":rf_space,"XGB":xg_space}; best={}
for k in sp:
    v=[roc_auc_score(val.churn,mk[k](c,42).fit(fit[CF],fit.churn).predict_proba(val[CF])[:,1]) for c in sp[k]]
    best[k]=sp[k][int(np.argmax(v))]; print(k,"best val",round(max(v),4),best[k],flush=True)
out["tuned_config"]=best
default={"LR":{},"RF":dict(n_estimators=100),"XGB":dict(n_estimators=200,max_depth=6)}
seeds={}
for k in ["LR","RF","XGB"]:
    for lab,cfg in [("default",default[k]),("tuned",best[k])]:
        a=[]
        for s in range(10):
            sc=mk[k](cfg,s).fit(tr[CF],tr.churn).predict_proba(te[CF])[:,1]; a.append(roc_auc_score(y,sc))
            if k=="LR": a=a*10; break
        seeds[f"{k}_{lab}"]=dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)),min=float(min(a)),max=float(max(a)))
        print(k,lab,seeds[f"{k}_{lab}"],flush=True)
out["seeds_test_AUC"]=seeds
json.dump(out,open("h2_ext.json","w"),indent=1,default=float)
