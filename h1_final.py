from common import *
import json,pickle,time,xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
cfg=json.load(open("h1_ext.json"))["tuning"]
br,bx=cfg["best_rf"],cfg["best_xgb"]
res={}
for name,mk in [("RF",lambda:RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=1)),
                ("XGB",lambda:xgb.XGBRegressor(n_estimators=200,max_depth=6,random_state=42,n_jobs=1,verbosity=0)),
                ("RF_tuned",lambda:RandomForestRegressor(random_state=42,n_jobs=1,**br)),
                ("XGB_tuned",lambda:xgb.XGBRegressor(random_state=42,n_jobs=1,verbosity=0,**bx))]:
    holder={}
    def make():
        m=mk(); holder['m']=m; return {name:m}
    P,t=walk(make)
    s=scores(P,name); s={k:round(v,4) for k,v in s.items()}
    s["size_MB"]=round(len(pickle.dumps(holder['m']))/2**20,2); s["train_s_7fits"]=round(t,1)
    res[name]=s; print(name,s,flush=True)
json.dump(res,open("h1_final.json","w"),indent=1)
