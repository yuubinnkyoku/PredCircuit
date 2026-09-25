from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from diagnose_pcalm_width64_update_lattice_alignment import run_alignment
from predcircuit.pcalm import ResidualMLP, Schedule, bp_loss, method_grad

STATE_LR=15.0/64.0
ALPHA=59.0/64.0
DUAL_LEAK=5.0/256.0
BUDGETS=(64,80,96)

def clone(base):
    m=ResidualMLP(depth=32,width=64,input_dim=8,output_dim=4,activation="relu",seed=0)
    m.load_state_dict(base.state_dict())
    return m

def apply(model,grads,lr):
    with torch.no_grad():
        for w,g in zip(model.weights,grads,strict=True): w.add_(g,alpha=-lr)

def loss(model,x,y):
    with torch.no_grad(): return float(bp_loss(model,x,y))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--updates",type=int,default=24)
    p.add_argument("--weight-lr",type=float,default=0.01)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    gen=torch.Generator().manual_seed(a.seed+90000)
    teacher=torch.randn(8,4,generator=gen)/8**0.5
    train_x=torch.randn(64,8,generator=gen); eval_x=torch.randn(256,8,generator=gen)
    train_y=torch.tanh(train_x@teacher); eval_y=torch.tanh(eval_x@teacher)
    base=ResidualMLP(depth=32,width=64,input_dim=8,output_dim=4,activation="relu",seed=a.seed+64032)
    names=["bp","spc_t80",*[f"pcalm_fixed_t{t}" for t in BUDGETS]]
    models={name:clone(base) for name in names}
    rows=[]
    for update in range(a.updates+1):
        for name,m in models.items():
            rows.append({"seed":a.seed,"method":name,"update":update,"eval_loss":loss(m,eval_x,eval_y)})
        if update==a.updates: break
        idx=torch.arange(update*4,update*4+4)%64; x,y=train_x[idx],train_y[idx]
        apply(models["bp"],method_grad(models["bp"],x,y,Schedule("bp",budget=0),state_lr=STATE_LR,rho=1.0),a.weight_lr)
        sg=method_grad(models["spc_t80"],x,y,Schedule("pc",budget=80),state_lr=STATE_LR,rho=1.0)
        if not all(torch.isfinite(g).all() for g in sg): raise RuntimeError("sPC non-finite")
        apply(models["spc_t80"],sg,a.weight_lr)
        for t in BUDGETS:
            name=f"pcalm_fixed_t{t}"
            gs,stats=run_alignment(models[name],x,y,update_precision="fixed14_i1",state_precision="fixed16_i3",dual_precision="fixed12_i1",budget=t,state_lr=STATE_LR,dual_leak=DUAL_LEAK,alpha=ALPHA)
            if not bool(stats["finite"]) or not all(torch.isfinite(g).all() for g in gs): raise RuntimeError(f"{name} non-finite")
            apply(models[name],gs,a.weight_lr)
    frame=pd.DataFrame(rows); initial=frame[frame.update==0].set_index("method").eval_loss
    frame["loss_ratio_to_initial"]=[r.eval_loss/initial[r.method] for r in frame.itertuples()]
    a.out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.out,index=False)
    print(frame[frame.update==a.updates].to_string(index=False))
if __name__=="__main__": main()
