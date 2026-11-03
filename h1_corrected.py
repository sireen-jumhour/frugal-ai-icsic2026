from common import *
import json, xgboost as xgb, pickle
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import wilcoxon
R={}
mk=lambda s=42:{"RF":RandomForestRegressor(n_estimators=100,random_state=s,n_jobs=1),"XGB":xgb.XGBRegressor(n_estimators=200,max_depth=6,random_state=s,n_jobs=1,verbosity=0)}
P,t=walk(mk); I=intermittent()
P=P.merge(I[["StockCode","Week","Croston","SBA","TSB"]],on=["StockCode","Week"])
P.to_pickle("P_corr.pkl")
M=["Naive","MovingAvg","SeasonalNaive","Croston","SBA","TSB","RF","XGB"]
R["full"]={k:{a:round(b,4) for a,b in scores(P,k).items()} for k in M}
for ml in ["RF","XGB"]:
    a,b=per_item(P,ml),per_item(P,"MovingAvg")
    R["full"][ml+"_vs_MA"]=dict(rel=round(1-R["full"][ml]["WAPE"]/R["full"]["MovingAvg"]["WAPE"],4),p=float(wilcoxon(a,b).pvalue),better_items=int((a<b).sum()))
    a2=per_item(P,"Naive")
    R["full"][ml+"_vs_Naive"]=dict(rel=round(1-R["full"][ml]["WAPE"]/R["full"]["Naive"]["WAPE"],4),p=float(wilcoxon(a,a2).pvalue),better_items=int((a<a2).sum()))
R["train_time"]=t
sc={}
for mo,wks in [(3,13),(6,26),(12,52)]:
    Pm,_=walk(lambda:mk(),wks)
    sc[mo]={k:round(scores(Pm,k)["WAPE"],4) for k in ["Naive","MovingAvg","RF","XGB"]}
R["scarcity"]=sc
json.dump(R,open("h1_corrected.json","w"),indent=1); print(json.dumps(R,indent=1))
