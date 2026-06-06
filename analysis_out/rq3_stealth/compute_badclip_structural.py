"""
Targeted add-on: compute the STRUCTURAL stealth metrics (|Delta mean-deg|, KS) for the
BadCLIP baseline, which run_rq3.py's group2_structural omitted (it only covered
{STAG, STAG-w/o-L_sem, STAG-w/o-L_str, CrossBA}).

BadCLIP (appendix adaptation) attaches a SMALL GRAPH TRIGGER to target nodes and has NO
structural-concealment objective (no L_str). Its trigger topology is therefore the naive /
dense (host-mismatched) topology -- the same family the framework already uses for CrossBA
and STAG-w/o-L_str (struct_off). The degree deviation and KS depend ONLY on this topology
(not on node features, the victim, the LLM, or SBERT), so we recompute them through the
IDENTICAL code path (R.train_struct_gsn + run_rq3.build_deployed_subgraph).

Sanity reproduction (validates the path matches the published numbers):
  STAG (struct_on, degree-matched) -> |Delta d| ~ 0.66, KS ~ 0.27
  CrossBA (struct_off, naive)      -> |Delta d| ~ 4.39, KS ~ 0.94
BadCLIP (struct_off, naive) is computed by the same loop; no number is hand-entered.

Run:  CUDA_VISIBLE_DEVICES=0 python compute_badclip_structural.py
"""
import os, sys, json
import numpy as np
import torch
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import rq3_stealth as R
import run_rq3 as RR
import common as C
from rq3_stealth import (DEVICE, TARGET_CLASS, NUM_TRIGGER_NODES, POISON_RATE,
                         SEEDS, N_POISON_TEST, K_HOPS)
from torch_geometric.utils import degree

REPO = R.REPO
DATASET = "cora"


def structural_for_cond(items, test_poison_ids, tf, struct_gsn, internal_t):
    """Exact copy of run_rq3.group2_structural's inner computation for ONE condition.
    Degrees are topology-only; tf is passed solely to size the node-feature matrix."""
    d_mean, d_var = [], []
    host_deg_pool, trig_deg_pool = [], []
    for i in test_poison_ids:
        g = items[i].to(DEVICE)
        host_deg = degree(g.edge_index[0], num_nodes=g.x.size(0)).float().cpu().numpy()
        x_all, ei_all, n_host = RR.build_deployed_subgraph(g, tf, struct_gsn, internal_t)
        deg_full = degree(ei_all[0], num_nodes=x_all.size(0)).float().cpu().numpy()
        host_after = deg_full[:n_host]
        trig_after = deg_full[n_host:]
        if trig_after.size == 0 or host_after.size == 0:
            continue
        d_mean.append(abs(float(trig_after.mean()) - float(host_after.mean())))
        d_var.append(abs(float(trig_after.var()) - float(host_after.var())))
        host_deg_pool.extend(host_deg.tolist())
        trig_deg_pool.extend(trig_after.tolist())
    ks = sps.ks_2samp(host_deg_pool, trig_deg_pool)
    return float(np.mean(d_mean)), float(np.mean(d_var)), float(ks.statistic), float(ks.pvalue)


def main():
    print(f"[badclip-struct] loading {DATASET}")
    data = torch.load(os.path.join(REPO, "processed_data", f"{DATASET}.pt"),
                      map_location="cpu", weights_only=False)
    N = data.y.shape[0]
    items, _ = C.build_node_subgraphs(data, num_hops=K_HOPS, device=DEVICE)
    feat_dim = data.x.size(1)
    tf_dummy = torch.zeros(NUM_TRIGGER_NODES, feat_dim, device=DEVICE)  # features irrelevant to degree

    # cond -> which gate family ("on"=degree-matched L_str, "off"=naive dense)
    COND_GATE = {"STAG": "on", "CrossBA": "off", "BadCLIP": "off"}
    acc = {c: dict(dmean=[], dvar=[], ks=[], ksp=[]) for c in COND_GATE}

    for seed in SEEDS:
        print(f"[badclip-struct] ---- seed {seed} ----")
        torch.manual_seed(seed)
        tr, va, te = C.make_split(N, seed=seed)
        rng = np.random.RandomState(seed)
        tr_shuf = tr.copy(); rng.shuffle(tr_shuf)
        n_pois = max(1, int(len(tr) * POISON_RATE))
        poison_ids = set(tr_shuf[:n_pois].tolist())

        te_pool = [i for i in te if int(data.y[i]) != TARGET_CLASS]
        rng.shuffle(te_pool)
        test_poison_ids = te_pool[:N_POISON_TEST]

        # identical to build_attack_artifacts: degree-matched + naive edge-gates
        struct_on, internal_t = R.train_struct_gsn(items, poison_ids, rng, use_str=True, epochs=12)
        struct_off, _ = R.train_struct_gsn(items, poison_ids, rng, use_str=False, epochs=12)
        gates = {"on": struct_on, "off": struct_off}

        for c, fam in COND_GATE.items():
            dm, dv, ks, ksp = structural_for_cond(items, test_poison_ids, tf_dummy,
                                                  gates[fam], internal_t)
            acc[c]["dmean"].append(dm); acc[c]["dvar"].append(dv)
            acc[c]["ks"].append(ks); acc[c]["ksp"].append(ksp)
            print(f"    {c:9s} |Delta d|={dm:.4f}  |Delta var|={dv:.4f}  KS={ks:.4f} (p={ksp:.2e})")

    print("\n[badclip-struct] ===== mean over seeds =====")
    summary = {}
    for c in COND_GATE:
        dm = float(np.mean(acc[c]["dmean"])); dms = float(np.std(acc[c]["dmean"]))
        ks = float(np.mean(acc[c]["ks"])); kss = float(np.std(acc[c]["ks"]))
        summary[c] = dict(delta_mean_deg=dm, delta_mean_deg_std=dms,
                          delta_var_deg=float(np.mean(acc[c]["dvar"])),
                          ks=ks, ks_std=kss, ks_p=float(np.mean(acc[c]["ksp"])),
                          gate=COND_GATE[c])
        print(f"  {c:9s} |Delta d| = {dm:.3f} +/- {dms:.3f}   KS = {ks:.3f} +/- {kss:.3f}   (gate={COND_GATE[c]})")

    print("\n[badclip-struct] sanity vs published: STAG~(0.66,0.27), CrossBA~(4.39,0.94)")
    out = os.path.join(HERE, "badclip_structural.json")
    with open(out, "w") as f:
        json.dump(dict(dataset=DATASET, victim="GraphCLIP", seeds=SEEDS,
                       n_poison_test=N_POISON_TEST, trigger_size=NUM_TRIGGER_NODES,
                       note=("BadCLIP uses a non-degree-matched small graph trigger (no L_str), "
                             "structurally the naive topology family (== CrossBA / STAG-w/o-L_str). "
                             "Degree deviation is topology-only; computed via the identical "
                             "train_struct_gsn + build_deployed_subgraph path."),
                       summary=summary), f, indent=2)
    print(f"[badclip-struct] wrote {out}")


if __name__ == "__main__":
    main()
