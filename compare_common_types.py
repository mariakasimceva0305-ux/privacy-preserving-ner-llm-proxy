"""Recompute micro scores on the same entity types for each baseline."""
from pathlib import Path
import argparse,json

TYPES=("Email","Номер телефона")

def compare(metrics):
    result={}
    for model,stats in metrics.items():
        c={k:sum(stats["per_type"].get(t,{}).get(k,0) for t in TYPES) for k in ("tp","fp","fn")}
        p=c["tp"]/max(1,c["tp"]+c["fp"]);r=c["tp"]/max(1,c["tp"]+c["fn"])
        result[model]={**c,"precision":p,"recall":r,"f1":2*p*r/max(1e-15,p+r)}
    return {"entity_types":list(TYPES),"models":result,
            "scope":"Same two entity types, same existing test; no retraining"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("metrics",type=Path);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();result=compare(json.loads(a.metrics.read_text(encoding="utf-8")))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
