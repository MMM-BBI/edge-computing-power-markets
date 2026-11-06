"""
实验编排：每个实验 = 若干"作业"（实验 × 种子）。作业目录 results/<实验>/seed_<种子>/：
  train_log.csv / val_log.csv / final.pt / train_meta.json      MARL 训练产物
  eval_<方法>/...（见 collector.py）                             统一评估产物
  DONE                                                           作业完成标记（已完成的作业自动跳过）
中断后重新运行同一个 cell 即可续跑。并行时每个作业的输出写入 run.log。
"""
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def results_root(smoke=False):
    return ROOT / ('results_smoke' if smoke else 'results')


def figures_root(smoke=False):
    return ROOT / ('figures_smoke' if smoke else 'figures')


TUNED_FILE = 'tuning/best_params.json'


def load_tuned(smoke=False):
    p = results_root(smoke) / TUNED_FILE
    return json.load(open(p)) if p.exists() else {}


def get_cfg(name, smoke=False, strict=True):
    """实验的完整参数 = 统一标准 + 调参选中的算法超参数 + 本实验唯一改变的因素。
    调参试验自带一整组超参数；其余实验要求调参全部完成（strict=False 时允许缺省，只用于不涉及训练的数据与标定 cell）。"""
    if name not in C.EXPERIMENTS and name.startswith('tune_confirm'):
        register_confirm(smoke)
    e = C.EXPERIMENTS[name]
    tuned = load_tuned(smoke)
    need = [] if 'tune_trial' in e else C.TUNE_KEYS        # 调参试验自带全部超参数
    missing = [k for k in need if k not in tuned]
    if missing and strict:
        raise RuntimeError(f'实验 {name} 需要先完成调参：{missing}（按顺序运行阶段一的调参 cell）')
    over = {k: tuned[k] for k in need if k in tuned}
    over.update(e['over'])
    if smoke:
        if 'n_consumers' in e['over']:                  # 冒烟：保留"规模递增"的结构，但都很小
            over['n_consumers'] = 40 + e['over']['n_consumers'] // 100
        return C.make_config('SMOKE', **over)
    return C.make_config(e['preset'], **over)


def get_seeds(name, smoke=False, seeds=None):
    s = list(seeds) if seeds is not None else C.EXPERIMENTS[name]['seeds']
    return s[:2] if smoke else s


def job_dir(name, seed, smoke=False):
    return results_root(smoke) / name / f'seed_{seed}'


def methods_for(name, seed, smoke=False):
    """SDR/MMR 为确定性规则、与训练种子无关，只在第一个种子下评估；CDA/McAfee 的激进度初值随种子变化。"""
    ms = C.EXPERIMENTS[name]['methods']
    first = get_seeds(name, smoke)[0]
    return [m for m in ms if not (m in ('sdr', 'mmr') and seed != first)]


def _dataset(cfg):
    from .data import load_dataset
    return load_dataset(cfg, DATA)


IGNORE_KEYS = {'device', 'torch_threads'}


def _check_stale(out, cfg):
    """已有结果若是用另一组参数跑出来的（例如改了 γ 之后），拒绝跳过，避免新旧结果混在一起。"""
    jm = out / 'job.json'
    if not jm.exists():
        return
    old = json.load(open(jm)).get('cfg', {})
    diff = {k: (old.get(k), v) for k, v in cfg.to_dict().items() if k not in IGNORE_KEYS and old.get(k, v) != v}
    if diff:
        raise RuntimeError(f'{out} 中的结果与当前参数不一致 {diff}。请把旧的 results 文件夹改名保存（如 results_old），'
                           f'或用 fresh=True 重跑该实验。')


def cfg_hash(cfg, seed):
    import hashlib
    # 后加入的参数取默认值时不计入哈希：与加入前的作业哈希逐位相同，已有结果的复用登记仍然有效
    d = {k: v for k, v in cfg.to_dict().items()
         if k not in IGNORE_KEYS and not (k in C.ADDED_KEYS and v == C.DEFAULTS[k])}
    return hashlib.md5(json.dumps([d, seed], sort_keys=True, default=str).encode()).hexdigest()[:16]


def _reuse_identical(cfg, seed, out, smoke):
    """若别的实验已经用完全相同的参数与种子跑过（例如调参选中的那一档与正文实验相同），直接复用其训练与评估结果。"""
    reg = results_root(smoke) / '_registry' / f'{cfg_hash(cfg, seed)}.txt'
    if not reg.exists() or (out / 'final.pt').exists():
        return False
    src = results_root(smoke) / reg.read_text().strip()
    if src == out or not (src / 'final.pt').exists():
        return False
    for f in ('final.pt', 'train_log.csv', 'val_log.csv', 'train_meta.json'):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    for d in src.glob('eval_*'):
        if (d / 'DONE').exists() and not (out / d.name).exists():
            shutil.copytree(d, out / d.name)
    (out / 'reused_from.txt').write_text(str(src.relative_to(results_root(smoke))))
    print(f'   ↺ 参数与种子完全相同，复用 {src.relative_to(results_root(smoke))} 的结果', flush=True)
    return True


def refresh_methods(methods, smoke=False, experiments=None):
    """规则或评估口径改动后，只重做这些方法的评估、不重训 MARL：删除指定实验（默认全部）中 eval_<方法> 目录与作业的
    DONE 标记，再运行相应实验的 cell 时只会补做缺失的评估（MARL 的 final.pt 保留）。
    experiments 可写实验名或前缀通配，例如 ['main', 'robust', 'struct_*']。"""
    import fnmatch
    root = results_root(smoke); n = 0
    for done in root.glob('*/seed_*/DONE'):
        job = done.parent; exp = job.parent.name
        if experiments and not any(fnmatch.fnmatch(exp, pat) for pat in experiments):
            continue
        hit = False
        for m in methods:
            d = job / f'eval_{m}'
            if d.exists():
                shutil.rmtree(d); hit = True
        if hit:
            done.unlink(); n += 1
    print(f'已清除 {n} 个作业中 {methods} 的评估结果；重新运行相应实验的 cell 即可补做评估（不重训）')
    return n


def reset_experiments(names, smoke=False):
    """需要重训的实验：把结果目录整体移到 results/_old/（不删除），并从复用登记中去掉，之后重新运行即可。"""
    import time as _t
    root = results_root(smoke); bak = root / '_old' / _t.strftime('%Y%m%d_%H%M%S'); n = 0
    for name in names:
        d = root / name
        if d.exists():
            bak.mkdir(parents=True, exist_ok=True); shutil.move(str(d), str(bak / name)); n += 1
    reg = root / '_registry'
    if reg.exists():
        for f in reg.glob('*.txt'):
            if f.read_text().split('/')[0].split('\\')[0] in names:
                f.unlink()
    print(f'已移走 {n} 个实验目录到 {bak}，重新运行即可重训')
    return n


def reset_subdirs(rel_paths, smoke=False):
    """删除某个作业下的子结果（例如可利用度中的 prosumer、buyer），重新运行对应 cell 时只重做这些部分。"""
    root = results_root(smoke); n = 0
    for rp in rel_paths:
        for d in root.glob(rp):
            shutil.rmtree(d); n += 1
    print(f'已删除 {n} 个子结果目录'); return n


def run_job(name, seed, smoke=False, verbose=True):
    from .runner import Trainer, evaluate, load_policy
    cfg = get_cfg(name, smoke)
    out = job_dir(name, seed, smoke)
    _check_stale(out, cfg)
    if (out / 'DONE').exists():
        return name, seed, 'skip', 0.0
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ds = _dataset(cfg)
    ms = methods_for(name, seed, smoke)
    _reuse_identical(cfg, seed, out, smoke)
    if 'marl' in ms and not (out / 'final.pt').exists():
        Trainer(cfg, ds, seed, out, verbose=verbose).train()
    for m in ms:
        if (out / f'eval_{m}' / 'DONE').exists():
            continue
        L = load_policy(cfg, ds, seed, out / 'final.pt') if m == 'marl' else None
        evaluate(cfg, ds, seed, m, out / f'eval_{m}', learner=L, verbose=verbose)
    with open(out / 'job.json', 'w') as f:
        json.dump(dict(experiment=name, seed=seed, methods=ms, minutes=(time.time() - t0) / 60,
                       cfg=cfg.to_dict()), f, indent=1, default=str)
    (out / 'DONE').write_text('ok')
    reg = results_root(smoke) / '_registry'; reg.mkdir(parents=True, exist_ok=True)
    (reg / f'{cfg_hash(cfg, seed)}.txt').write_text(str(out.relative_to(results_root(smoke))))
    return name, seed, 'ok', (time.time() - t0) / 60


def _guarded(fn, args, log_path, threads=1):
    os.environ['OMP_NUM_THREADS'] = str(threads); os.environ['MKL_NUM_THREADS'] = str(threads)
    import torch
    torch.set_num_threads(threads)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        try:
            return fn(*args, verbose=True)
        except Exception:
            tb = traceback.format_exc()
            (log_path.parent / 'ERROR.txt').write_text(tb, encoding='utf-8')
            print(tb)
            return (args[0], args[1], 'error', 0.0)


def _job_worker(args):
    name, seed, smoke, threads = args
    return _guarded(run_job, (name, seed, smoke), job_dir(name, seed, smoke) / 'run.log', threads)


def _run_parallel(worker, jobs, workers, label):
    t0 = time.time(); res = []
    if workers <= 1:
        for j in jobs:
            print(f'▶ {label(j)}', flush=True)
            r = worker(j); res.append(r)
            print(f'   → {r[2]}（{r[3]:.1f} min）', flush=True)
        return res
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result(); res.append(r)
            print(f'[{k}/{len(jobs)}] {label(futs[f])} → {r[2]}（{r[3]:.1f} min）| 已用 {(time.time() - t0) / 60:.1f} min',
                  flush=True)
    return res


def run_experiment(name, workers=1, smoke=False, seeds=None, fresh=False, threads=1):
    """运行一个实验的全部作业（训练 + 统一评估）。workers>1 时多进程并行，每个进程用 threads 个 CPU 线程。"""
    seeds = get_seeds(name, smoke, seeds)
    cfg = get_cfg(name, smoke)
    print(f"📋 实验 {name}：{C.EXPERIMENTS[name]['label']} | 种子 {seeds} | 方法 {C.EXPERIMENTS[name]['methods']} | "
          f"N={cfg.n_consumers}, 训练 {cfg.train_days} 天", flush=True)
    if fresh:
        for s in seeds:
            d = job_dir(name, s, smoke)
            if d.exists():
                shutil.rmtree(d)
    _dataset(cfg)                                            # 主进程先生成数据，避免并行进程重复生成
    if workers <= 1:
        res = []
        for s in seeds:
            print(f'▶ {name} / seed {s}', flush=True)
            r = run_job(name, s, smoke, verbose=True); res.append(r)
            print(f'   → {r[2]}（{r[3]:.1f} min）', flush=True)
    else:
        jobs = [(name, s, smoke, threads) for s in seeds]
        res = _run_parallel(_job_worker, jobs, workers, lambda j: f'{j[0]} / seed {j[1]}')
    bad = [r for r in res if r[2] == 'error']
    for r in bad:
        print(f'   ❌ {r[0]} / seed {r[1]}：见 {job_dir(r[0], r[1], smoke) / "ERROR.txt"}')
    if bad:
        raise RuntimeError(f'{len(bad)} 个作业失败')
    print(f'✅ {name} 完成', flush=True)
    return res


def run_group(names, workers=1, smoke=False, threads=1):
    """把多个实验的作业放进同一个进程池（例如同一个 cell 内的几档敏感性），充分利用并行。"""
    jobs = []
    for n in names:
        cfg = get_cfg(n, smoke); _dataset(cfg)
        jobs += [(n, s, smoke, threads) for s in get_seeds(n, smoke)]
    print(f'📋 {len(jobs)} 个作业：{names}', flush=True)
    if workers <= 1:
        res = []
        for j in jobs:
            print(f'▶ {j[0]} / seed {j[1]}', flush=True)
            r = run_job(j[0], j[1], smoke, verbose=True); res.append(r)
            print(f'   → {r[2]}（{r[3]:.1f} min）', flush=True)
    else:
        res = _run_parallel(_job_worker, jobs, workers, lambda j: f'{j[0]} / seed {j[1]}')
    bad = [r for r in res if r[2] == 'error']
    for r in bad:
        print(f'   ❌ {r[0]} / seed {r[1]}：见 {job_dir(r[0], r[1], smoke) / "ERROR.txt"}')
    if bad:
        raise RuntimeError(f'{len(bad)} 个作业失败')
    print('✅ 完成', flush=True)
    return res


# ───────────────────────────── 阶段一：调参 ─────────────────────────────
def _trial_row(name, seed, smoke):
    d = job_dir(name, seed, smoke)
    meta = json.load(open(d / 'train_meta.json'))
    test = json.load(open(d / 'eval_marl' / 'summary.json'))
    e = C.EXPERIMENTS[name]
    return dict(trial=e['tune_trial'], seed=seed, experiment=name, **{k: e['over'][k] for k in C.TUNE_KEYS},
                val_unit_welfare=meta['val_final']['val_unit_welfare'], test_unit_welfare=test['unit_welfare'],
                train_minutes=meta['train_minutes'])


def run_tuning_search(workers=1, smoke=False, threads=1):
    """调参第 1 步：全部随机组合（含默认值）各训练一次（种子 MAIN_SEED），按验证集排名，记录前 TOP_K 名。"""
    names = [n for n, e in C.EXPERIMENTS.items() if e['group'] == 'tune']
    run_group(names, workers=workers, smoke=smoke, threads=threads)
    seed0 = get_seeds(names[0], smoke)[0]
    df = pd.DataFrame([_trial_row(n, seed0, smoke) for n in names]).sort_values('val_unit_welfare', ascending=False)
    df['rank'] = np.arange(1, len(df) + 1)
    out = results_root(smoke) / 'tuning'; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'search.csv', index=False)
    top = [int(t) for t in df['trial'].iloc[:C.TOP_K]]
    with open(out / 'topk.json', 'w') as f:
        json.dump(top, f)
    print(f'✅ 随机搜索完成：验证集前 {C.TOP_K} 名为试验 {top}')
    return df


def register_confirm(smoke=False):
    p = results_root(smoke) / 'tuning' / 'topk.json'
    if not p.exists():
        raise RuntimeError('请先运行调参第 1 步（run_tuning_search）')
    trials = dict(C.tuning_trials())
    names = []
    for t in json.load(open(p)):
        n = f'tune_confirm_t{t:02d}'
        C.EXPERIMENTS[n] = dict(C._exp('tune_confirm', f'confirm trial {t:02d}', ['marl'], seeds=C.CONFIRM_SEEDS,
                                       **trials[t]), tune_trial=t)
        names.append(n)
    return names


def run_tuning_confirm(workers=1, smoke=False, threads=1):
    """调参第 2 步：前 TOP_K 名组合补跑 CONFIRM_SEEDS，按（种子 0 + 补跑种子）的验证集均值选定最终参数。"""
    names = register_confirm(smoke)
    run_group(names, workers=workers, smoke=smoke, threads=threads)
    rows = []
    for n in names:
        t = C.EXPERIMENTS[n]['tune_trial']
        rows.append(_trial_row(f'tune_t{t:02d}', get_seeds(f'tune_t{t:02d}', smoke)[0], smoke))
        rows += [_trial_row(n, s, smoke) for s in get_seeds(n, smoke)]
    df = pd.DataFrame(rows)
    summ = df.groupby('trial').agg(val_mean=('val_unit_welfare', 'mean'), val_sd=('val_unit_welfare', 'std'),
                                   test_mean=('test_unit_welfare', 'mean'), n_seeds=('seed', 'count')).reset_index()
    summ = summ.sort_values('val_mean', ascending=False)
    best = int(summ['trial'].iloc[0])
    params = dict(C.tuning_trials())[best]
    out = results_root(smoke) / 'tuning'
    df.to_csv(out / 'confirm_by_seed.csv', index=False); summ.to_csv(out / 'confirm.csv', index=False)
    with open(out / 'best_params.json', 'w') as f:
        json.dump(dict(params), f, indent=1)
    print(f'✅ 选定试验 {best:02d}：{params}（{len(get_seeds(names[0], smoke)) + 1} 个种子验证均值 {summ["val_mean"].iloc[0]:.2f}）')
    return summ


# ───────────────────────────── E1 效率上界 ─────────────────────────────
def run_upper_bound(base='main', smoke=False):
    from .upper_bound import daily_upper_bound
    cfg = get_cfg(base, smoke)
    out = results_root(smoke) / base / 'upper_bound.csv'
    if out.exists():
        print(f'（已存在）{out}'); return pd.read_csv(out)
    ds = _dataset(cfg)
    t0 = time.time()
    df = daily_upper_bound(cfg, ds)
    df.to_csv(out, index=False)
    print(f'✅ 上界完成：单位福利上界 {1e4 * df.W_star.sum() / df.demand.sum():.2f}（{(time.time() - t0) / 60:.1f} min）')
    return df


# ───────────────────────────── 可利用度（单边最优响应） ─────────────────────────────
def _exploit_job(base, seed, dev, smoke, verbose=True):
    import torch
    from .env import MarketEnv
    from .learner import Group, default_groups
    from .runner import load_policy, evaluate, run_day, split_day, set_seed
    cfg = get_cfg(base, smoke)
    ds = _dataset(cfg)
    src = job_dir(base, seed, smoke)
    out = src / 'exploit' / dev
    if (out / 'DONE').exists():
        return base, seed, 'skip', 0.0
    out.mkdir(parents=True, exist_ok=True)
    set_seed(1000 + seed)
    env = MarketEnv(cfg, ds, seed=seed)
    groups = default_groups(env, cfg.share_intermediaries)
    sd = torch.load(src / 'final.pt', map_location='cpu')
    if dev.startswith('inter_'):
        dev_name, agent_desc = dev, dev
    else:
        role = 'Prosumer' if dev == 'prosumer' else 'PureBuyer'
        idx_role = env.idx_pros if role == 'Prosumer' else env.idx_pb
        # 群体偏离：随机抽取本角色 5% 的散户共用一个偏离策略一起重训（单个散户在上千人中的学习信号太弱，
        # 单人重训测不出东西）；其余主体策略冻结。收益为偏离群体的人均剩余
        grng = np.random.default_rng(777 + seed)
        n_dev = max(1, int(round(C.EXPLOIT.get('retail_group_frac', 0.05) * len(idx_role))))
        k = np.sort(grng.choice(len(idx_role), n_dev, replace=False))
        g0 = [g for g in groups if g.name == role][0]
        g0.idx = np.setdiff1d(g0.idx, k)
        dev_name = f'dev_{dev}'
        groups.append(Group(dev_name, role, k))
        sd = dict(sd); sd[dev_name] = sd[role]
        agent_desc = f'{role} group of {n_dev}'
        dev_agents = idx_role[k]
    L = load_policy(cfg, ds, seed, src / 'final.pt', env=env, groups=groups, trainable=[dev_name])
    L.load_state_dict(sd, strict_names=False)
    L.train()
    br_days = min(C.EXPLOIT['br_days'], cfg.train_days) if not smoke else 2
    L.exploration_scale, L.entropy_coef, L.lr_scale, L.inter_floor = (C.ADAPT['exploration_scale'], cfg.entropy_floor,
                                                                      C.ADAPT['lr_scale'], cfg.inter_explore_end)
    # 偏离者在验证日上的确定性收益：每 br_check 天检查一次，保留最好的一版（含未训练的原策略），
    # 因此最优响应的收益增量 ≥ 0（最优响应的定义）；选择只用验证集，增量在测试集上报告
    m_idx = env.inter_names.index(dev) if dev.startswith('inter_') else None
    pos = None if dev.startswith('inter_') else k                       # 偏离者在本角色奖励数组中的位置

    from .runner import warm_env
    venv = MarketEnv(cfg, ds, seed=seed)
    L.eval(); warm_env(venv, cfg, ds, 'train', learner=L); warm_env(env, cfg, ds, 'train', learner=L); L.train()   # 先预热价格预期

    def val_payoff():
        n = cfg.val_days if not smoke else 2
        days = np.linspace(0, cfg.val_days - 1, min(C.EXPLOIT.get('val_days', 20), n)).astype(int)
        L.eval(); tot = 0.0
        with torch.no_grad():
            for kd in days:
                d, t, did = split_day(cfg, ds, 'val', int(kd))
                obs = venv.reset(d, t, did); done = False
                while not done:
                    acts = L.act(obs, None, deterministic=True)
                    obs, rew, done, info = venv.step(acts)
                    if dev.startswith('inter_'):
                        tot += float(rew['Intermediary'][m_idx])
                    else:
                        tot += float(np.mean(rew[role][pos]))
        L.train()
        return tot / len(days)

    best_val, best_ep = val_payoff(), 0
    best_sd = {kk: {n: v.clone() for n, v in net.state_dict().items()} for kk, net in L.nets.items()}
    check = C.EXPLOIT.get('br_check', 100) if not smoke else 1
    hist = [dict(day=0, val_payoff=best_val)]
    for ep in range(br_days):
        run_day(env, cfg, ds, 'train', ep, learner=L, train=True)
        L.update()
        if (ep + 1) % check == 0 or ep == br_days - 1:
            v = val_payoff(); hist.append(dict(day=ep + 1, val_payoff=v))
            if v > best_val:
                best_val, best_ep = v, ep + 1
                best_sd = {kk: {n: t.clone() for n, t in net.state_dict().items()} for kk, net in L.nets.items()}
    for kk, net in L.nets.items():
        net.load_state_dict(best_sd[kk])
    pd.DataFrame(hist).to_csv(out / 'br_val_curve.csv', index=False)
    torch.save(L.state_dict(), out / 'br.pt')
    summ = evaluate(cfg, ds, seed, 'marl', out / 'eval_br', learner=L, verbose=verbose)
    # 偏离者收益：BR 策略 vs 原策略（同一评估协议、同一组随机数）
    if dev.startswith('inter_'):
        before = pd.read_csv(src / 'eval_marl' / 'inter.csv').set_index('inter').loc[dev, 'profit']
        after = pd.read_csv(out / 'eval_br' / 'inter.csv').set_index('inter').loc[dev, 'profit']
    else:
        before = pd.read_csv(src / 'eval_marl' / 'agents.csv').set_index('agent').loc[dev_agents, 'surplus_per_day'].mean()
        after = pd.read_csv(out / 'eval_br' / 'agents.csv').set_index('agent').loc[dev_agents, 'surplus_per_day'].mean()
    w0 = json.load(open(src / 'eval_marl' / 'summary.json'))['welfare_per_day']
    res = dict(seed=seed, deviator=dev, agent=agent_desc, payoff_before=float(before), payoff_after=float(after),
               gain=float(after - before), rel_gain=float((after - before) / max(abs(before), 1e-9)),
               welfare_before=w0, welfare_after=summ['welfare_per_day'], br_days=br_days, best_br_day=best_ep)
    with open(out / 'result.json', 'w') as f:
        json.dump(res, f, indent=1)
    (out / 'DONE').write_text('ok')
    return base, seed, 'ok', 0.0


def _exploit_worker(args):
    base, seed, dev, smoke, threads = args
    return _guarded(_exploit_job, (base, seed, dev, smoke), job_dir(base, seed, smoke) / 'exploit' / f'{dev}.log', threads)


def run_exploit(workers=1, smoke=False, threads=1):
    base = C.EXPLOIT['base']
    jobs = [(base, s, d, smoke, threads) for s in get_seeds(base, smoke) for d in C.EXPLOIT['deviators']]
    if workers <= 1:
        for j in jobs:
            print(f'▶ 最优响应 {j[2]} / seed {j[1]}', flush=True)
            _exploit_job(j[0], j[1], j[2], smoke)
    else:
        _run_parallel(_exploit_worker, jobs, workers, lambda j: f'{j[2]} / seed {j[1]}')
    rows = []
    for s in get_seeds(base, smoke):
        for d in C.EXPLOIT['deviators']:
            p = job_dir(base, s, smoke) / 'exploit' / d / 'result.json'
            if p.exists():
                rows.append(json.load(open(p)))
    df = pd.DataFrame(rows)
    df.to_csv(results_root(smoke) / base / 'exploitability.csv', index=False)
    print('✅ 可利用度检验完成'); return df


# ───────────────────────────── 开销与可扩展性 ─────────────────────────────
def run_overhead(smoke=False):
    import torch
    from .data import load_dataset
    from .env import MarketEnv
    from .runner import build_learner, split_day
    out = results_root(smoke) / 'overhead.csv'
    sizes = C.OVERHEAD['sizes'] if not smoke else [40, 80]
    rows = []
    for N in sizes:
        cfg = C.make_config('MAIN', n_consumers=N, train_days=3, val_days=2, test_days=1)
        ds = load_dataset(cfg, DATA)
        env = MarketEnv(cfg, ds, seed=0)
        d, t, did = split_day(cfg, ds, 'train', 0)
        obs = env.reset(d, t, did)
        L = build_learner(cfg, env, 'cpu')
        t_env = t_pol = 0.0; steps = 0
        for k in range(C.OVERHEAD['days']):
            d, t, did = split_day(cfg, ds, 'train', k)
            obs = env.reset(d, t, did); done = False
            while not done:
                a0 = time.perf_counter(); acts = L.act(obs, None, deterministic=False, store=True); a1 = time.perf_counter()
                obs, rew, done, info = env.step(acts); a2 = time.perf_counter()
                L.store_rewards(rew)
                t_pol += a1 - a0; t_env += a2 - a1; steps += 1
            u0 = time.perf_counter(); L.update(); u1 = time.perf_counter()
        rows.append(dict(N=N, clearing_ms_per_step=1e3 * t_env / steps, policy_ms_per_step=1e3 * t_pol / steps,
                         update_s_per_day=(u1 - u0), est_train_s_per_day=(t_env + t_pol) / C.OVERHEAD['days'] + (u1 - u0)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    return df
