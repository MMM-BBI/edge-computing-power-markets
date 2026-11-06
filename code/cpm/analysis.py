"""
结果汇总：把各作业的评估产物整理成论文表格（results/tables/*.csv）。

统计口径（第八节）：单位福利为"比值之比"10⁴·ΣW/ΣD；表中为各训练种子的均值 ± 标准差；
方法间比较对测试日做配对 Wilcoxon 检验（逐日比值只用于检验）。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from . import config as C
from .experiments import results_root, get_seeds, job_dir, get_cfg

METRICS = [('welfare_per_day', 'Welfare (CNY/day)'), ('unit_welfare', 'Unit welfare'),
           ('inter_profit_avg', 'Avg. intermediary profit'), ('buyer_savings_pc', 'Buyer savings'),
           ('prosumer_savings_pc', 'Prosumer savings'), ('p2_vol', 'Retail volume'),
           ('ecc_fallback_vol', 'ECC fallback volume'), ('internal_flow_share', 'Internal flow share'),
           ('ecc_revenue', 'ECC retail revenue'), ('ecc_buyback', 'ECC buy-back spend'),
           ('operator_surplus', 'Operator surplus')]
NETWORK = [('internal_flow_share', 'Internal flow share'), ('throughput', 'Intermediary throughput'),
           ('leakage', 'Leakage to ECC'), ('hhi_retail', 'HHI (retail)'), ('in_degree', 'Aggregation in-degree'),
           ('bidir_share', 'Bidirectional node share'), ('cross_layer_share', 'Cross-layer substitution share'),
           ('share_p2_at_cap', 'Share of retail trades at bid cap'), ('edge_seller_margin', 'Edge value: sellers'),
           ('edge_inter_margin', 'Edge value: intermediaries'), ('edge_buyer_gain', 'Edge value: buyers'),
           ('p2_fill_rate', 'Retail fill rate'), ('fallback_self_per_day', 'Fallback self-supply / day'),
           ('unfilled_eligible_per_day', 'Eligible-but-unfilled bids / day'),
           ('sellall_ecc_per_day', 'Sell-all → ECC events / day'),
           ('p1_price_rel', 'P1 price / ECC retail price'), ('reserve_rel', 'Mean reservation / ECC retail price'),
           ('idle_sold_share', 'Share of idle capacity sold'),
           ('net_edges_p1', 'P1 edges / day'), ('net_edges_p2', 'P2 edges / day'),
           ('net_persist_p1', 'P1 edge persistence (Jaccard, day to day)'),
           ('net_persist_p2', 'P2 edge persistence (Jaccard, day to day)'),
           ('net_active_sellers', 'Active seller share / day'), ('net_seller_degree', 'Seller degree (intermediaries / day)'),
           ('net_buyer_degree', 'Buyer degree (intermediaries / day)'), ('net_inter_in_degree', 'Intermediary in-degree / day'),
           ('net_inter_out_degree', 'Intermediary out-degree / day'), ('net_gini_sellers', 'Gini of seller volume'),
           ('net_gini_buyers', 'Gini of buyer volume'), ('net_top10_sellers', 'Top-10% seller volume share'),
           ('net_cv_p1_daily', 'CV of daily P1 volume'), ('net_cv_p2_daily', 'CV of daily P2 volume')]
METHOD_LABEL = {'marl': 'MARL', 'cda': 'CDA', 'sdr': 'SDR', 'mmr': 'MMR', 'mcafee': 'McAfee'}


def tables_dir(smoke=False):
    d = results_root(smoke) / 'tables'; d.mkdir(parents=True, exist_ok=True)
    return d


def eval_dir(exp, seed, method, smoke=False):
    return job_dir(exp, seed, smoke) / f'eval_{method}'


def load_summaries(exp, method, smoke=False):
    out = []
    for s in get_seeds(exp, smoke):
        p = eval_dir(exp, s, method, smoke) / 'summary.json'
        if p.exists():
            d = json.load(open(p)); d['seed'] = s; out.append(d)
    return out


def load_daily(exp, method, smoke=False):
    """各种子逐日表，按测试日对齐后取种子均值（用于配对检验）。"""
    dfs = []
    for s in get_seeds(exp, smoke):
        p = eval_dir(exp, s, method, smoke) / 'daily.csv'
        if p.exists():
            dfs.append(pd.read_csv(p)[['day', 'W', 'demand']].assign(seed=s))
    if not dfs:
        return None
    df = pd.concat(dfs).groupby('day')[['W', 'demand']].mean().reset_index()
    df['unit_W'] = 1e4 * df['W'] / df['demand']
    return df


def fmt(mean, sd, n):
    if n <= 1 or not np.isfinite(sd):
        return f'{mean:.2f}'
    return f'{mean:.2f} ± {sd:.2f}'


def agg(summs, key):
    v = np.array([s.get(key, np.nan) for s in summs], dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.mean()) if len(v) else np.nan, float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v))


def paired_test(exp_a, m_a, exp_b, m_b, smoke=False):
    a, b = load_daily(exp_a, m_a, smoke), load_daily(exp_b, m_b, smoke)
    if a is None or b is None:
        return np.nan, np.nan
    j = a.merge(b, on='day', suffixes=('_a', '_b'))
    diff = j['unit_W_a'] - j['unit_W_b']
    if len(j) < 2 or np.allclose(diff, 0):
        return float(diff.mean()) if len(j) else np.nan, np.nan
    return float(diff.mean()), float(wilcoxon(j['unit_W_a'], j['unit_W_b']).pvalue)


def table_methods(exp='main', smoke=False):
    """Table 3：四种定价方法 + McAfee 机制对照（main 为单种子；robust 为多种子均值 ± 标准差）。"""
    rows = []
    for m in C.ALL_METHODS:
        S = load_summaries(exp, m, smoke)
        if not S:
            continue
        r = dict(method=METHOD_LABEL[m], n_seeds=(f'{len(S)} (deterministic rule)' if m in ('sdr', 'mmr') else len(S)))
        for k, lab in METRICS:
            mu, sd, n = agg(S, k); r[lab] = fmt(mu, sd, n); r[k + '_mean'] = mu
        if m != 'marl':
            d, p = paired_test(exp, 'marl', exp, m, smoke)
            r['ΔUnit welfare (MARL − method), daily'] = d; r['Wilcoxon p (daily, supplementary)'] = p
            # 主结论用种子层面的比较：每个 MARL 种子与同种子的基准配对（SDR/MMR 为确定性规则，与每个种子比较同一个值）
            Sm = {s['seed']: s['unit_welfare'] for s in load_summaries(exp, 'marl', smoke)}
            Sb = {s['seed']: s['unit_welfare'] for s in S}
            if len(Sm) > 1:
                diffs = np.array([v - Sb.get(k, next(iter(Sb.values()))) for k, v in Sm.items()])
                r['ΔUnit welfare (MARL − method), seed-paired'] = fmt(float(diffs.mean()), float(diffs.std(ddof=1)), len(diffs))
                r['Seeds where MARL is better'] = f'{int((diffs > 0).sum())}/{len(diffs)}'
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir(smoke) / f'table_methods_{exp}.csv', index=False)
    return df


def table_network(exp='main', smoke=False):
    rows = []
    for m in C.ALL_METHODS:
        S = load_summaries(exp, m, smoke)
        if not S:
            continue
        r = dict(method=METHOD_LABEL[m])
        for k, lab in NETWORK:
            mu, sd, n = agg(S, k); r[lab] = fmt(mu, sd, n)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir(smoke) / f'table_network_{exp}.csv', index=False)
    return df


def table_intermediaries(exp='main', method='marl', smoke=False):
    dfs = []
    for s in get_seeds(exp, smoke):
        p = eval_dir(exp, s, method, smoke) / 'inter.csv'
        if p.exists():
            dfs.append(pd.read_csv(p).assign(seed=s))
    if not dfs:
        return None
    df = pd.concat(dfs).groupby('inter').mean(numeric_only=True).reset_index().drop(columns='seed')
    df.to_csv(tables_dir(smoke) / f'table_intermediaries_{exp}_{method}.csv', index=False)
    return df


def _abl_summaries(exp, smoke):
    """消融/归因用：完整方法只取 robust 中 SEEDS_ABL 的种子，与各变体按种子配对。"""
    S = load_summaries(exp, 'marl', smoke)
    if exp == C.ABL_REF:
        keep = set(C.SEEDS_ABL) if not smoke else set(get_seeds(C.ABL_REF, smoke)) & set(C.SEEDS_ABL)
        S = [s for s in S if s['seed'] in keep]
    return S


def seed_paired(S_var, S_ref, key='unit_welfare'):
    """按种子配对的差值：均值、标准差、种子数、变体更差的种子数、配对 t 检验 p（种子数 ≥ 3 时）。"""
    from scipy.stats import ttest_rel
    a = {s['seed']: s.get(key) for s in S_var}; b = {s['seed']: s.get(key) for s in S_ref}
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds], float)
    if not len(d):
        return dict(delta=np.nan, delta_sd=np.nan, n_pairs=0, n_worse=0, p_seed=np.nan)
    p = float(ttest_rel([a[s] for s in seeds], [b[s] for s in seeds]).pvalue) if len(d) >= 3 and d.std() > 0 else np.nan
    return dict(delta=float(d.mean()), delta_sd=float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                n_pairs=len(d), n_worse=int((d < 0).sum()), p_seed=p)


def table_ablation(smoke=False):
    """消融表：各变体 3 个种子的均值 ± 标准差；Δ 为与完整方法同种子配对的差值（均值 ± 标准差）。
    判读：|Δ| 明显大于种子间差异、且 3 个种子方向一致，才算该设计有作用。"""
    rows = []
    ref = _abl_summaries(C.ABL_REF, smoke)
    for exp, lab in C.ABLATIONS:
        S = _abl_summaries(exp, smoke)
        if not S:
            continue
        r = dict(variant=lab, experiment=exp, n_seeds=len(S))
        for k in ('unit_welfare', 'inter_profit_avg', 'buyer_savings_pc', 'prosumer_savings_pc', 'internal_flow_share'):
            mu, sd, n = agg(S, k); r[k] = fmt(mu, sd, n); r[k + '_mean'] = mu
        if exp != C.ABL_REF:
            t = seed_paired(S, ref)
            r['Δ vs full'] = fmt(t['delta'], t['delta_sd'], t['n_pairs']); r['Δ_mean'] = t['delta']
            r['seeds worse'] = f"{t['n_worse']}/{t['n_pairs']}"; r['paired t p'] = t['p_seed']
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(tables_dir(smoke) / 'table_ablation.csv', index=False)
    return df


def table_attribution(smoke=False):
    """2×2 归因：每格为 3 个种子的均值 ± 标准差；主效应与交互效应按种子分别计算后取均值 ± 标准差。"""
    rows, per = [], {}
    for exp, price, qty in C.ATTRIBUTION:
        S = _abl_summaries(exp, smoke)
        if not S:
            continue
        mu, sd, n = agg(S, 'unit_welfare')
        ip, _, _ = agg(S, 'inter_profit_avg')
        rows.append(dict(experiment=exp, price=price, quantity=qty, n_seeds=n, unit_welfare=fmt(mu, sd, n),
                         unit_welfare_mean=mu, inter_profit_avg=ip))
        per[(price, qty)] = {s['seed']: s['unit_welfare'] for s in S}
    df = pd.DataFrame(rows)
    if len(per) == 4:
        seeds = sorted(set.intersection(*[set(v) for v in per.values()]))
        g = {k: np.array([v[s] for s in seeds]) for k, v in per.items()}
        eff = dict(price_effect=(g['learned', 'learned'] + g['learned', 'rule'] - g['rule', 'learned'] - g['rule', 'rule']) / 2,
                   qty_effect=(g['learned', 'learned'] + g['rule', 'learned'] - g['learned', 'rule'] - g['rule', 'rule']) / 2,
                   interaction=g['learned', 'learned'] - g['learned', 'rule'] - g['rule', 'learned'] + g['rule', 'rule'])
        for k, v in eff.items():
            df.attrs[k] = float(v.mean())
        f = lambda v: fmt(float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v))
        df = pd.concat([df, pd.DataFrame([dict(experiment='main effects', price=f"price effect {f(eff['price_effect'])}",
                                               quantity=f"quantity effect {f(eff['qty_effect'])}",
                                               unit_welfare=f"interaction {f(eff['interaction'])}", n_seeds=len(seeds))])])
    df.to_csv(tables_dir(smoke) / 'table_attribution.csv', index=False)
    return df


def axis_def(group_key, smoke=False):
    """敏感性 / 结构参数的 (显示名, 取值点)；默认值取正文实验 main 的实际参数。"""
    from .experiments import get_cfg
    lab = (C.SENS.get(group_key) or C.STRUCT.get(group_key))[0]
    default = getattr(get_cfg('main', smoke, strict=False), group_key)
    if smoke and group_key == 'n_consumers':
        default = 70
    return lab, C.axis_points(group_key, default)


def table_axis(group_key, smoke=False, metrics=('unit_welfare', 'inter_profit_avg', 'buyer_savings_pc',
                                                'prosumer_savings_pc', 'internal_flow_share', 'hhi_retail',
                                                'p1_vol', 'p2_vol', 'net_persist_p1', 'net_persist_p2')):
    title, points = axis_def(group_key, smoke)
    rows = []
    methods = C.SENS_METHODS if group_key in C.SENS else C.ALL_METHODS     # 敏感性只看 MARL；结构实验各方法同台
    for val, exp in points:
        for m in methods:
            S = load_summaries(exp, m, smoke)
            if not S:
                continue
            r = dict(parameter=group_key, value=val, experiment=exp, method=METHOD_LABEL[m])
            for k in metrics:
                r[k] = float(np.mean([s.get(k, np.nan) for s in S]))
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(tables_dir(smoke) / f'table_sens_{group_key}.csv', index=False)
    return df


def table_tuning(smoke=False):
    """调参汇总：随机搜索全部组合（search.csv）与前几名多种子复核（confirm.csv）。"""
    from .experiments import results_root as rr
    d = rr(smoke) / 'tuning'
    if not (d / 'search.csv').exists():
        return None
    df = pd.read_csv(d / 'search.csv')
    if (d / 'confirm.csv').exists():
        c = pd.read_csv(d / 'confirm.csv')
        df = df.merge(c[['trial', 'val_mean', 'val_sd', 'test_mean', 'n_seeds']], on='trial', how='left')
        df['selected'] = df['trial'] == int(c['trial'].iloc[0])
    df.to_csv(tables_dir(smoke) / 'table_tuning.csv', index=False)
    return df


def table_efficiency(exp='main', smoke=False):
    p = results_root(smoke) / exp / 'upper_bound.csv'
    if not p.exists():
        return None
    ub = pd.read_csv(p)
    Wstar = ub['W_star'].sum()
    rows = [dict(method='Upper bound (LP)', unit_welfare=1e4 * Wstar / ub['demand'].sum(), efficiency=1.0)]
    for m in C.ALL_METHODS:
        d = load_daily(exp, m, smoke)
        if d is None:
            continue
        j = d.merge(ub[['day', 'W_star']], on='day')
        rows.append(dict(method=METHOD_LABEL[m], unit_welfare=1e4 * j['W'].sum() / j['demand'].sum(),
                         efficiency=j['W'].sum() / j['W_star'].sum()))
    df = pd.DataFrame(rows); df.to_csv(tables_dir(smoke) / f'table_efficiency_{exp}.csv', index=False)
    return df


SHOCK_METRICS = ('welfare_per_day', 'unit_welfare', 'p1_vol', 'p2_vol', 'internal_flow_share', 'ecc_fallback_vol',
                 'hhi_retail', 'inter_profit_avg', 'buyer_savings_pc', 'prosumer_savings_pc', 'p1_price_rel',
                 'net_edges_p1', 'net_edges_p2', 'net_persist_p1', 'net_persist_p2', 'net_seller_degree',
                 'net_buyer_degree', 'net_gini_sellers', 'net_cv_p1_daily', 'net_cv_p2_daily')


def _shock_family(name):
    if name.startswith('inter_'):
        return 'intermediary'
    if name.startswith('size_'):
        return 'size'
    return name.split('_')[0]


def table_netshock(smoke=False):
    """网络冲击表：每个情景 × 方法的指标，以及相对完整网络（正文实验同一方法）的变化：d_* 为相对变化（%），
    a_* 为绝对变化。基数可能接近 0 的量（如 McAfee 下的中介利润）只看绝对变化。"""
    from .experiments import get_seeds as gs
    base = C.NETSHOCK['base']; seed = gs(base, smoke)[0]
    rows = []
    for m in C.NETSHOCK['methods']:
        p0 = job_dir(base, seed, smoke) / f'eval_{m}' / 'summary.json'
        if not p0.exists():
            continue
        s0 = json.load(open(p0))
        rows.append(dict(scenario='intact', family='intact', method=METHOD_LABEL[m], key=m,
                         **{k: s0.get(k, np.nan) for k in SHOCK_METRICS}, **{f'd_{k}': 0.0 for k in SHOCK_METRICS}, **{f'a_{k}': 0.0 for k in SHOCK_METRICS}))
        for name, _ in C.NETSHOCK['scenarios']:
            p = job_dir(base, seed, smoke) / 'netshock' / name / f'eval_{m}' / 'summary.json'
            if not p.exists():
                continue
            s1 = json.load(open(p))
            r = dict(scenario=name, family=_shock_family(name), method=METHOD_LABEL[m], key=m,
                     **{k: s1.get(k, np.nan) for k in SHOCK_METRICS})
            for k in SHOCK_METRICS:
                a, b = s0.get(k, np.nan), s1.get(k, np.nan)
                r[f'd_{k}'] = 100 * (b - a) / abs(a) if (a is not None and np.isfinite(a) and a != 0) else np.nan
                r[f'a_{k}'] = b - a if (a is not None and b is not None) else np.nan     # 绝对变化（中介利润等可能接近 0 的量用它）
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tables_dir(smoke) / 'table_netshock.csv', index=False)
    return df


def table_recovery(smoke=False):
    """冲击后恢复：每个情景的验证指标轨迹，相对 control 第 0 天（未冲击的正文策略）的比值；
    并汇总"冲击当下的跌幅"与"300 天后恢复到的水平"。"""
    base = C.RECOVERY['base']; seed = get_seeds(base, smoke)[0]
    root = job_dir(base, seed, smoke) / 'recovery'
    pc = root / 'control' / 'curve.csv'
    if not pc.exists():
        return None
    ref = pd.read_csv(pc).iloc[0]
    curves, summ = [], []
    for name in C.RECOVERY['scenarios']:
        p = root / name / 'curve.csv'
        if not p.exists():
            continue
        c = pd.read_csv(p)
        for k in ('welfare', 'p1_vol', 'p2_vol'):
            c[f'rel_{k}'] = c[k] / ref[k] if ref[k] else np.nan
        c['scenario'] = name
        curves.append(c)
        summ.append(dict(scenario=name, **{f'{k}_at_shock': float(c[f'rel_{k}'].iloc[0]) for k in ('welfare', 'p1_vol', 'p2_vol')},
                         **{f'{k}_after': float(c[f'rel_{k}'].iloc[-3:].mean()) for k in ('welfare', 'p1_vol', 'p2_vol')}))
    df = pd.DataFrame(summ)
    if len(df) and (df.scenario == 'control').any():
        ctl = df[df.scenario == 'control'].iloc[0]
        for k in ('welfare', 'p1_vol', 'p2_vol'):            # 再适应收益（双重差分，单位：未冲击水平的比例）
            df[f'{k}_adapt_gain'] = (df[f'{k}_after'] - df[f'{k}_at_shock']) - (ctl[f'{k}_after'] - ctl[f'{k}_at_shock'])
            df[f'{k}_recovered_share'] = np.where(df[f'{k}_at_shock'] < 1 - 1e-3,
                                                  df[f'{k}_adapt_gain'] / (1 - df[f'{k}_at_shock']), np.nan)
    cur = pd.concat(curves) if curves else pd.DataFrame()
    cur.to_csv(tables_dir(smoke) / 'recovery_curves.csv', index=False)
    df.to_csv(tables_dir(smoke) / 'table_recovery.csv', index=False)
    return df


def table_exploit(smoke=False):
    p = results_root(smoke) / C.EXPLOIT['base'] / 'exploitability.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    t = df.groupby('deviator')[['payoff_before', 'payoff_after', 'gain', 'rel_gain']].agg(['mean', 'std']).reset_index()
    t.columns = [' '.join(c).strip() for c in t.columns]
    t.to_csv(tables_dir(smoke) / 'table_exploitability.csv', index=False)
    return t


def table_runtime(smoke=False):
    rows = []
    for exp in C.EXPERIMENTS:
        for s in get_seeds(exp, smoke):
            p = job_dir(exp, s, smoke) / 'train_meta.json'
            if p.exists():
                m = json.load(open(p))
                rows.append(dict(experiment=exp, seed=s, train_minutes=m['train_minutes'], gpu_mem_mb=m.get('gpu_mem_mb', 0),
                                 device=m.get('device')))
    df = pd.DataFrame(rows); df.to_csv(tables_dir(smoke) / 'table_runtime.csv', index=False)
    return df


def all_tables(smoke=False):
    out = {}
    for name, fn in [('methods_main', lambda: table_methods('main', smoke)), ('network_main', lambda: table_network('main', smoke)),
                     ('methods_robust', lambda: table_methods('robust', smoke)),
                     ('network_robust', lambda: table_network('robust', smoke)), ('tuning', lambda: table_tuning(smoke)),
                     ('intermediaries', lambda: table_intermediaries('main', 'marl', smoke)),
                     ('ablation', lambda: table_ablation(smoke)), ('attribution', lambda: table_attribution(smoke)),
                     ('efficiency', lambda: table_efficiency('main', smoke)), ('netshock', lambda: table_netshock(smoke)),
                     ('recovery', lambda: table_recovery(smoke)),
                     ('exploit', lambda: table_exploit(smoke)), ('runtime', lambda: table_runtime(smoke))]:
        try:
            out[name] = fn()
        except Exception as e:                               # 缺少某组结果时跳过，不影响其他表
            print(f'  （跳过 {name}：{e}）')
    for ax in list(C.SENS) + list(C.STRUCT):
        try:
            out[ax] = table_axis(ax, smoke)
        except Exception as e:
            print(f'  （跳过 {ax}：{e}）')
    print(f'✅ 表格已写入 {tables_dir(smoke)}')
    return out


def describe_runs(smoke=False, experiments=None):
    """运行核查表：每个作业实际用的规模、训练天数、已记录回合数、耗时与设备。
    用来确认结果确实来自预期设置（例如主实验应为 N=3000、训练 2800 天、train_log 2800 行）。"""
    rows = []
    for exp in (experiments or C.EXPERIMENTS):
        for s in get_seeds(exp, smoke):
            d = job_dir(exp, s, smoke)
            if not d.exists():
                continue
            r = dict(experiment=exp, seed=s, done=(d / 'DONE').exists())
            jm = d / 'job.json'
            if jm.exists():
                cfg = json.load(open(jm))['cfg']
                r.update(N=cfg['n_consumers'], intermediaries=cfg['n_intermediaries'], train_days_cfg=cfg['train_days'],
                         test_days=cfg['test_days'])
            tl = d / 'train_log.csv'
            if tl.exists():
                r['episodes_logged'] = len(pd.read_csv(tl, usecols=['episode']))
            tm = d / 'train_meta.json'
            if tm.exists():
                m = json.load(open(tm))
                r.update(train_minutes=round(m['train_minutes'], 1), device=m.get('device'),
                         data=Path(m.get('data', '')).name)
            ru = d / 'reused_from.txt'
            r['reused_from'] = ru.read_text() if ru.exists() else ''
            r['evaluated'] = ','.join(sorted(p.name[5:] for p in d.glob('eval_*') if (p / 'DONE').exists()))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir(smoke) / 'run_check.csv', index=False)
    return df


def convergence_table(exp='main', smoke=False, keys=('unit_welfare', 'inter_profit_avg', 'buyer_savings_pc', 'prosumer_savings_pc')):
    """收敛诊断（逐种子、逐指标），只在稳定期（最后 explore_lock_frac 的回合：探索、熵、学习率均已固定）内计算：
      · 训练曲线：稳定期后半段相对前半段的变化、稳定期内的线性斜率（每 100 回合变化占均值的比例）及其 p 值、变异系数；
      · 确定性验证曲线（主要依据）：同样的三项，外加验证峰值出现在第几回合、稳定期均值相对全程峰值的差距。
    判定 converged：|后半段相对变化| < 2%，且斜率不显著（p > 0.05）或 |斜率| < 1%/100 回合。
    稳定期内探索噪声恒定，训练曲线不再因退火"假性上升"，所以平台才能说明策略已经停止变化。"""
    from scipy.stats import linregress
    rows = []
    cfg = get_cfg(exp, smoke, strict=False)

    def stats(ep, y, lock):
        m = ep >= lock
        e, v = ep[m], y[m]
        if len(v) < 4:
            return {}
        h = len(v) // 2
        mu, first, second = v.mean(), v[:h].mean(), v[h:].mean()
        lr = linregress(e, v)
        rel = (second - first) / abs(first) if first else np.nan
        slope = lr.slope * 100 / abs(mu) if mu else np.nan
        return dict(plateau_mean=mu, half_rel_change=rel, slope_per100_rel=slope, slope_p=lr.pvalue,
                    cv=v.std() / abs(mu) if mu else np.nan,
                    converged=bool(abs(rel) < 0.02 and (lr.pvalue > 0.05 or abs(slope) < 0.01)))

    for s in get_seeds(exp, smoke):
        p = job_dir(exp, s, smoke) / 'train_log.csv'
        if not p.exists():
            continue
        l = pd.read_csv(p); n = len(l)
        lock = int(n * (1 - cfg.explore_lock_frac))
        for k in keys:
            rows.append(dict(seed=s, curve='train', metric=k, plateau_from=lock, n_points=int((l.episode >= lock).sum()),
                             **stats(l.episode.values, l[k].values.astype(float), lock)))
        v = job_dir(exp, s, smoke) / 'val_log.csv'
        if v.exists():
            vv = pd.read_csv(v)
            for k in [c for c in vv.columns if c.startswith('val_')]:
                ok = np.isfinite(vv[k].values.astype(float))          # 网络指标（如无二级成交时的 HHI）可能为 NaN
                y, ep = vv[k].values.astype(float)[ok], vv.episode.values[ok]
                st = stats(ep, y, lock)
                if st:
                    pk = int(ep[np.argmax(y)])
                    st.update(peak_episode=pk, peak_value=float(y.max()),
                              plateau_vs_peak=(st['plateau_mean'] - y.max()) / abs(y.max()) if y.max() else np.nan)
                rows.append(dict(seed=s, curve='validation', metric=k, plateau_from=lock,
                                 n_points=int((ep >= lock).sum()), **st))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tables_dir(smoke) / f'convergence_{exp}.csv', index=False)
    return df
