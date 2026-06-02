#!/usr/bin/env python3
import torch, sys, argparse
from collections import defaultdict
sys.path.insert(0, ".")
from rhgnn_end_to_end import set_seed, load_temporal_kg, build_snapshot_graphs, evaluate
from rhgnn_v2 import adversarial_loss
from rhgnn_v3 import build_history_vocab
from rhgnn_v4 import RHGNNv4, soft_label_loss
from geoopt.optim import RiemannianAdam

EPOCHS=200; PATIENCE=40; EVAL_EVERY=20; DATA_DIR="data/ICEWS14"

VARIANTS = [
    ("Full RHGNN-C",      {}, {}),
    ("w/o Contrastive",   {}, {"no_contrast": True}),
    ("w/o Soft Labels",   {}, {"no_soft": True}),
    ("w/o Temp Smooth",   {"no_smooth": True}, {"no_smooth": True}),
    ("w/o History Vocab", {"no_history": True}, {}),
    ("w/o Subgraph",      {"no_sgcn": True}, {}),
    ("w/o H-GRU",         {"no_hgru": True}, {}),
    ("w/o ODE",           {"no_ode": True}, {}),
    ("w/o Hyperbolic",    {"no_hyperbolic": True}, {}),
]

def make_model(data, device, no_history=False, no_hyperbolic=False,
               no_ode=False, no_hgru=False, no_sgcn=False, no_smooth=False):
    model = RHGNNv4(
        num_entities=data.num_entities, num_relations=data.num_relations,
        dim=200, dropout=0.1, ode_steps=5 if not no_ode else 0,
        num_sgcn_layers=0 if no_sgcn else 2,
        smooth_label=0.1, lambda_smooth=0.0 if no_smooth else 0.01,
    ).to(device)
    model.set_history_vocab(build_history_vocab(data.train) if not no_history else {})
    if no_hyperbolic:
        def _to_hyp_eucl(ids): return model.entity_emb(ids)
        def _score_all_eucl(h, r, tau, snapshots, delta_t=1.0):
            msg=model._aggregate_messages(h,tau,snapshots)
            v=model.entity_emb(h)
            rz=torch.sigmoid(model.Wr(v)+model.Ur(msg))
            z=torch.sigmoid(model.Wz(v)+model.Uz(msg))
            hc=torch.tanh(model.Wh(v)+model.Uh(rz*msg))
            h_j=z*hc+(1-z)*v
            pred=h_j+model.relation_emb(r)
            dist=(pred.unsqueeze(1)-model.entity_emb.weight.unsqueeze(0)).norm(dim=-1)
            return -dist+model.bias.unsqueeze(0)
        model._to_hyp=_to_hyp_eucl
        model.score_all_tails=_score_all_eucl
    if no_hgru:
        model._hgru_jump=lambda h,m,delta_t=1.0:h
    if no_ode:
        model._ode_flow=lambda h,dt:h
    return model

def run(args):
    device=torch.device("cuda:0")
    set_seed(42)
    name,model_kw,train_kw=VARIANTS[args.variant]
    print(f"Variant {args.variant}: {name}",flush=True)
    data=load_temporal_kg(DATA_DIR)
    snapshots=build_snapshot_graphs(data.train)
    ts_to_real={}
    for raw_ts in data.sorted_timestamps:
        tid=data.time2id.get(str(raw_ts),raw_ts)
        ts_to_real[tid]=raw_ts
    tgd=defaultdict(list)
    for q in data.train: tgd[q[3]].append(q)
    train_groups=sorted(tgd.items())
    model=make_model(data,device,**model_kw)
    model.build_nbr_index(snapshots)
    model.refresh_msg_table(device)
    opt=RiemannianAdam(model.parameters(),lr=1e-3)
    sch=torch.optim.lr_scheduler.StepLR(opt,step_size=50,gamma=0.8)
    best_mrr,best_state,no_imp=0.0,None,0
    no_contrast=train_kw.get("no_contrast",False)
    no_soft=train_kw.get("no_soft",False)
    no_smooth=model_kw.get("no_smooth",False)
    for epoch in range(1,EPOCHS+1):
        model.train()
        tl,tn=0.0,0
        prev=None
        for ts_id,quads in train_groups:
            rt=ts_to_real.get(ts_id,ts_id)
            dt=max(float(rt-prev) if prev else 1.0,1.0)
            prev=rt
            for s in range(0,len(quads),1024):
                mini=quads[s:s+1024]
                batch=torch.tensor(mini,dtype=torch.long,device=device)
                h,r,t,tau=batch[:,0],batch[:,1],batch[:,2],batch[:,3]
                if no_soft:
                    pos=model.score(h,r,t,tau,snapshots,dt)
                    negs=torch.stack([model.score(h,r,torch.randint(0,data.num_entities,(h.size(0),),device=device),tau,snapshots,dt) for _ in range(10)],dim=-1)
                    loss=adversarial_loss(pos,negs)
                else:
                    scores=model.score_all_tails(h,r,tau,snapshots,dt)
                    loss=soft_label_loss(scores,t,smooth=0.1)
                if not no_contrast:
                    loss=loss+0.1*model.compute_contrastive(h,r,t,tau,snapshots,dt)
                if not no_smooth:
                    loss=loss+model.temporal_smooth_loss()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                tl+=loss.item()*len(mini); tn+=len(mini)
        sch.step(); model.update_prev_emb()
        if epoch%EVAL_EVERY==0:
            model.refresh_msg_table(device)
            vm=evaluate(model,data.valid,data.all_true,snapshots,ts_to_real,device,batch_size=512)
            mrr=vm["MRR"]
            print(f"  ep{epoch:04d} loss={tl/tn:.4f} valid={mrr:.4f}",flush=True)
            if mrr>best_mrr:
                best_mrr=mrr
                best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
                no_imp=0
            else:
                no_imp+=1
                if no_imp>=PATIENCE//EVAL_EVERY:
                    print(f"  Early stop ep{epoch}",flush=True); break
    model.load_state_dict(best_state); model.to(device)
    model.build_nbr_index(snapshots); model.refresh_msg_table(device)
    tm=evaluate(model,data.test,data.all_true,snapshots,ts_to_real,device,batch_size=512)
    print(f"\nRESULT|{name}|MRR={tm['MRR']:.4f}|H@1={tm['Hits@1']:.4f}|H@3={tm['Hits@3']:.4f}|H@10={tm['Hits@10']:.4f}|MAR={tm['MAR']:.1f}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--variant",type=int,required=True)
    run(p.parse_args())
