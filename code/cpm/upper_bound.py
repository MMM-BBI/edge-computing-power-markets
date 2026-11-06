"""
E1 效率上界：同一张中介化流网络上的最小费用流（设计文档 11.1）。

每个测试时段解一个线性规划（已知当期全部需求；放宽整卡与整单约束；中介整合成本取各中介中最低的 o）：
  变量  s_i ≥ 0      产消者 i 经中介出让的算力
        w_i ≥ 0      产消者 i 卖给 ECC 的闲置算力（仅当 P^EB ≥ c^run）
        y_j ∈ [0,1]  产消者 j 的需求由自有卡满足的比例（仅当 g_j ∈ G_{j,t} 且卡数足够）
        x_jg ∈ [0,1] 需求方 j 的需求经中介由卡型 g 满足的比例（g ∈ G_{j,t}）
        z_j ∈ [0,1]  需求方 j 向 ECC 整单购买的比例
  约束  s_i + w_i + d_i·y_i ≤ cap_i ；Σ_g x_jg + z_j + y_j = 1 ；Σ_j d_j·x_jg ≤ Σ_{i: g_i = g} s_i
  目标  min C = Σ c_i s_i + Σ (c_i − P^EB) w_i + Σ c_j d_j y_j + Σ o_g d_j x_jg + Σ B_j z_j
由 W = Σ(U − U^base) + ΣΠ = −C − ΣU^base（内部转移相互抵消），上界 W* = −C* − ΣU^base。
该上界是效率参照而非紧界（放宽了整卡、整单、单一来源与中介份额约束）。
产消者保留价（公平份额与损失厌恶）只约束出售意愿、不计入成本，因此上界按首优（first-best）处理：规划者不受保留价约束，
回购 w_i 只要求 P^EB ≥ c^run；仿真中实际可行的配置都满足这些约束，故 W* ≥ W 仍成立，效率缺口包含了保留价带来的惜售损失。
U^base 与仿真使用同一口径（env.U_base，按"无中介化市场"的反事实保留价计算，与定价方法无关），福利定义一致。
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

from . import specs
from .env import MarketEnv


def hour_lp(env):
    d, act = env.d, env.active
    P = np.where(env.is_pros)[0]
    J = np.where(act)[0]
    nP, nJ = len(P), len(J)
    Gm = env.Gm[J]
    xj, xg = np.nonzero(Gm)
    nx = len(xj)
    # 变量顺序：s[nP] w[nP] y[nJ] x[nx] z[nJ]
    o_s, o_w, o_y, o_x, o_z = 0, nP, 2 * nP, 2 * nP + nJ, 2 * nP + nJ + nx
    nv = o_z + nJ
    c = np.zeros(nv)
    c[o_s:o_s + nP] = env.c[P]
    peb_own = env.peb[env.own0[P]]
    c[o_w:o_w + nP] = env.c[P] - peb_own
    c[o_y:o_y + nJ] = env.c[J] * d[J]
    c[o_x:o_x + nx] = env.o_min[xg] * d[J][xj]
    c[o_z:o_z + nJ] = env.B[J]
    ub = np.full(nv, np.inf)
    ub[o_w:o_w + nP] = np.where(peb_own >= env.c[P], np.inf, 0.0)
    ub[o_y:o_y + nJ] = np.where(env.can_self[J], 1.0, 0.0)
    ub[o_x:o_x + nx] = 1.0; ub[o_z:o_z + nJ] = 1.0
    # 容量约束（产消者）
    pos_in_J = -np.ones(env.N, dtype=np.int64); pos_in_J[J] = np.arange(nJ)
    rows, cols, vals = [], [], []
    for k, i in enumerate(P):
        rows += [k, k]; cols += [o_s + k, o_w + k]; vals += [1.0, 1.0]
        if pos_in_J[i] >= 0:
            rows.append(k); cols.append(o_y + pos_in_J[i]); vals.append(d[i])
    # 卡型平衡
    own_P = env.own0[P]
    for g in range(specs.G):
        r = nP + g
        sel = np.where(xg == g)[0]
        rows += [r] * len(sel); cols += list(o_x + sel); vals += list(d[J][xj[sel]])
        ps = np.where(own_P == g)[0]
        rows += [r] * len(ps); cols += list(o_s + ps); vals += [-1.0] * len(ps)
    A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(nP + specs.G, nv))
    b_ub = np.concatenate([env.cap[P], np.zeros(specs.G)])
    # 需求覆盖
    rows, cols, vals = [], [], []
    rows += list(xj); cols += list(o_x + np.arange(nx)); vals += [1.0] * nx
    rows += list(range(nJ)); cols += list(o_z + np.arange(nJ)); vals += [1.0] * nJ
    rows += list(range(nJ)); cols += list(o_y + np.arange(nJ)); vals += [1.0] * nJ
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(nJ, nv))
    b_eq = np.ones(nJ)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(np.zeros(nv), ub)), method='highs')
    if res.status != 0:
        raise RuntimeError(f'LP 求解失败: {res.message}')
    C = res.fun
    W_star = -C - env.U_base.sum()
    x = res.x
    return dict(W_star=W_star, C=C, internal=float((d[J][xj] * x[o_x:o_x + nx]).sum()),
                ecc=float((d[J] * x[o_z:o_z + nJ]).sum()), self=float((d[J] * x[o_y:o_y + nJ]).sum()))


def daily_upper_bound(cfg, ds, seed=0, n_days=None):
    env = MarketEnv(cfg, ds, seed=seed)
    n_days = n_days or cfg.test_days
    rows = []
    for k in range(n_days):
        dem, tie, did = ds['test_demands'][k], ds['test_tiers'][k], cfg.train_days + cfg.val_days + k
        env.reset(dem, tie, did)
        acc = dict(day=did, W_star=0.0, demand=0.0, internal=0.0, ecc=0.0, self=0.0)
        for h in range(cfg.steps_per_day):
            env.hour = h; env._prepare_hour()
            r = hour_lp(env)
            acc['W_star'] += r['W_star']; acc['demand'] += float(env.d.sum())
            acc['internal'] += r['internal']; acc['ecc'] += r['ecc']; acc['self'] += r['self']
        rows.append(acc)
    df = pd.DataFrame(rows)
    df['unit_W_star'] = 1e4 * df['W_star'] / df['demand']
    return df
