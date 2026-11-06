"""
网络实验：以正文实验（main，种子 0）的已训练策略为基准，改变交易网络本身，观察网络结构与稳定性。

两类实验
  网络冲击（run_netshock）  不重训，全部定价方法同台：中介退出 / 新进入、产消者退出、纯买方退出、
                            切断某一卡型层、整个市场变小或变大（N = 300…5000，正文策略直接迁移）。
                            冲击即时生效，度量"短期"响应：一级/二级成交量、内部流量、福利、集中度、关系持续性。
  冲击后恢复（run_recovery）只针对 MARL：在冲击后的网络上按再学习协议（config.ADAPT，与最优响应重训相同）继续学习
                            300 天，每 25 天在验证集上度量，得到"冲击 → 下跌 → 再适应"的轨迹（网络韧性）。
                            control 为不加冲击的同样再学习；再适应收益按双重差分计算：
                            (情景末期 − 情景第 0 天) − (control 末期 − control 第 0 天)。
                            冲击移除的产能（中介份额相对原 Σθ、退出的散户）不会被补回，所以能恢复的只是"定价再适应"那一部分。
结果目录：results/main/seed_0/netshock/<情景>/eval_<方法>/、results/main/seed_0/recovery/<情景>/curve.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .experiments import get_cfg, job_dir, results_root, _dataset, _guarded, _run_parallel, get_seeds


def _scenario(name):
    for n, spec in C.NETSHOCK['scenarios']:
        if n == name:
            return dict(spec)
    if name == 'control':
        return {}
    raise KeyError(name)


def shock_setup(base, name, smoke):
    """返回 (cfg, ds, env_kwargs, clone)：clone = {新中介名: 复制其策略的已有中介名}。"""
    spec = _scenario(name)
    if 'n_consumers' in spec:
        cfg = get_cfg(f"struct_n_consumers_{spec['n_consumers']}", smoke)
        return cfg, _dataset(cfg), {}, {}
    cfg = get_cfg(base, smoke)
    ds = _dataset(cfg)
    kw, clone = {}, {}
    if 'exclude_inters' in spec:
        kw['exclude_inters'] = spec['exclude_inters']
    if 'entrants' in spec:
        kw['entrants'] = spec['entrants']
        n0 = len(cfg.thetas)
        clone = {f'inter_{n0 + k}': 'inter_1' for k in range(spec['entrants'])}   # 新进入者 θ = 1，沿用中等规模中介的策略
    if 'blocked_layers' in spec:
        kw['blocked_layers'] = spec['blocked_layers']
    if 'drop_role' in spec:
        prof = ds['profiles']
        idx = np.array([p['id'] for p in prof if p['agent_type'] == spec['drop_role']])
        order = np.random.default_rng(2026).permutation(idx)          # 固定排列：10% ⊂ 20% ⊂ 30% ⊂ 50%
        kw['drop_agents'] = np.sort(order[:int(round(spec['drop_frac'] * len(idx)))]).tolist()
    return cfg, ds, kw, clone


def _shock_learner(cfg, ds, seed, ckpt, kw, clone):
    import torch
    from .env import MarketEnv
    from .runner import load_policy
    env = MarketEnv(cfg, ds, seed=seed, **kw)
    L = load_policy(cfg, ds, seed, ckpt, env=env)
    if clone:
        sd = torch.load(ckpt, map_location='cpu')
        for new, src in clone.items():
            L.nets[new].load_state_dict(sd[src])
    if env.dropped.any():                      # 退出的散户不再行动，也不进入恢复训练的样本（否则零收益样本会稀释共享网络的梯度）
        amap = dict(Prosumer=env.idx_pros, PureBuyer=env.idx_pb)
        for g in L.groups:
            if g.role in amap:
                g.idx = g.idx[~env.dropped[np.asarray(amap[g.role])[g.idx]]]
    L.eval()
    return L


def _netshock_job(base, seed, name, smoke, verbose=False):
    from .runner import evaluate
    cfg, ds, kw, clone = shock_setup(base, name, smoke)
    src = job_dir(base, seed, smoke)
    out = src / 'netshock' / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'scenario.json').write_text(json.dumps(dict(name=name, spec=_scenario(name), env_kwargs={
        k: (v if k != 'drop_agents' else len(v)) for k, v in kw.items()}, clone=clone), default=str))
    for m in C.NETSHOCK['methods']:
        if (out / f'eval_{m}' / 'DONE').exists():
            continue
        L = _shock_learner(cfg, ds, seed, src / 'final.pt', kw, clone) if m == 'marl' else None
        evaluate(cfg, ds, seed, m, out / f'eval_{m}', learner=L, env_kwargs=kw, verbose=verbose)
    return base, seed, 'ok', 0.0


def _netshock_worker(args):
    base, seed, name, smoke, threads = args
    return _guarded(_netshock_job, (base, seed, name, smoke), job_dir(base, seed, smoke) / 'netshock' / f'{name}.log', threads)


def run_netshock(workers=1, smoke=False, threads=1):
    base = C.NETSHOCK['base']
    seed = get_seeds(base, smoke)[0]
    if not (job_dir(base, seed, smoke) / 'final.pt').exists():
        raise RuntimeError('请先完成正文实验 main')
    jobs = [(base, seed, n, smoke, threads) for n, _ in C.NETSHOCK['scenarios']]
    if workers <= 1:
        for j in jobs:
            print(f'▶ 网络冲击 {j[2]}', flush=True)
            _netshock_job(j[0], j[1], j[2], smoke)
    else:
        _run_parallel(_netshock_worker, jobs, workers, lambda j: f'{j[2]}')
    print('✅ 网络冲击评估完成')
    from .analysis import table_netshock
    return table_netshock(smoke)


# ───────────────────────────── 冲击后的恢复（MARL） ─────────────────────────────
def _measure(venv, cfg, ds, L, days):
    from .runner import run_day
    import torch
    acc = dict(W=0.0, D=0.0, p1=0.0, p2=0.0, ecc=0.0, inter=0.0, pros=0.0, pb=0.0)
    with torch.no_grad():
        for k in days:
            r = run_day(venv, cfg, ds, 'val', int(k), learner=L)
            for key in acc:
                acc[key] += r[key]
    n = max(len(days), 1)
    return dict(welfare=acc['W'] / n, unit_welfare=1e4 * acc['W'] / max(acc['D'], 1e-9), p1_vol=acc['p1'] / n,
                p2_vol=acc['p2'] / n, ecc_vol=acc['ecc'] / n, inter_profit=acc['inter'] / n,
                prosumer_surplus=acc['pros'] / n, buyer_surplus=acc['pb'] / n)


def _recovery_job(base, seed, name, smoke, verbose=False):
    from .env import MarketEnv
    from .runner import run_day, warm_env
    cfg, ds, kw, clone = shock_setup(base, name, smoke)
    src = job_dir(base, seed, smoke)
    out = src / 'recovery' / name
    if (out / 'DONE').exists():
        return base, seed, 'skip', 0.0
    out.mkdir(parents=True, exist_ok=True)
    R = C.RECOVERY
    days = R['days'] if not smoke else 4
    check = R['check'] if not smoke else 2
    vdays = np.linspace(0, cfg.val_days - 1, min(R['val_days'], cfg.val_days)).astype(int)
    L = _shock_learner(cfg, ds, seed, src / 'final.pt', kw, clone)
    env = MarketEnv(cfg, ds, seed=seed, **kw)
    venv = MarketEnv(cfg, ds, seed=seed, **kw)
    warm_env(env, cfg, ds, 'train', learner=L); warm_env(venv, cfg, ds, 'train', learner=L)
    # 再学习协议（与可利用度的最优响应重训相同）：重新打开适度探索，学习率取调参值的一半
    L.exploration_scale, L.entropy_coef, L.lr_scale, L.inter_floor = (C.ADAPT['exploration_scale'], cfg.entropy_floor,
                                                                      C.ADAPT['lr_scale'], cfg.inter_explore_end)
    rows = [dict(day=0, **_measure(venv, cfg, ds, L, vdays))]
    L.train()
    for d in range(days):
        run_day(env, cfg, ds, 'train', d % cfg.train_days, learner=L, train=True)
        L.update()
        if (d + 1) % check == 0 or d == days - 1:
            L.eval()
            venv.copy_state_from(env)
            rows.append(dict(day=d + 1, **_measure(venv, cfg, ds, L, vdays)))
            L.train()
            if verbose:
                print(name, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / 'curve.csv', index=False)
    (out / 'DONE').write_text('ok')
    return base, seed, 'ok', 0.0


def _recovery_worker(args):
    base, seed, name, smoke, threads = args
    return _guarded(_recovery_job, (base, seed, name, smoke), job_dir(base, seed, smoke) / 'recovery' / f'{name}.log', threads)


def run_recovery(workers=1, smoke=False, threads=1):
    base = C.RECOVERY['base']
    seed = get_seeds(base, smoke)[0]
    jobs = [(base, seed, n, smoke, threads) for n in C.RECOVERY['scenarios']]
    if workers <= 1:
        for j in jobs:
            print(f'▶ 冲击后恢复 {j[2]}', flush=True)
            _recovery_job(j[0], j[1], j[2], smoke)
    else:
        _run_parallel(_recovery_worker, jobs, workers, lambda j: f'{j[2]}')
    print('✅ 冲击后恢复完成')
    from .analysis import table_recovery
    return table_recovery(smoke)
