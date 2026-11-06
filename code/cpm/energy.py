"""
算电协同补充实验：对已有正文结果的补充分析（不改变任何已有设定与结果）。

情景（与正文一致的需求–电价耦合：有分时电价就有负荷转移，平坦电价 ξ = 0）
  main   分时电价，ξ = 0.1                      —— 正文实验（已有）
  flat   平坦电价，ξ = 0（需求按 ξ = 0 生成）       —— 已有的 sens_tariff_flat
  pv1.5  峰谷差 × 1.5，ξ = 0.15                 —— 新训练 energy_pv_1.5
  pv2    峰谷差 × 2，  ξ = 0.2                  —— 新训练 energy_pv_2
  全部为 MARL、种子 MAIN_SEED，其余设定与正文相同。

评估（不训练；与正文完全相同的评估协议，额外保存逐时面板）
  · 上述四个情景各用自己训练的策略评估（长期）；
  · 反事实（正文策略，不重训）：
      cf_cost   平坦电价 + 正文需求数据 —— 与正文只差电价，得到电价成本的直接作用；
      cf_flat   平坦电价 + 平坦需求数据 —— 与 cf_cost 只差需求数据，得到需求连锁反应（负荷转移 + 两次随机抽样的差异）；
      cf_pv1.5 / cf_pv2  正文策略面对放大的峰谷差与相应需求；
      rv_cost   分时电价 + 平坦需求数据 + 平坦电价策略    —— 反向路径的第一步；
      rv_demand 分时电价 + 正文需求数据 + 平坦电价策略    —— 反向路径的第二步；
  · 正文条件下的四种规则基准（只用于分时段对照）。

分析
  1 复现核查      本模块对已有作业的评估与正文评估逐项比较，应完全一致；
  2 噪声参照      两套需求数据中纯买方（不受 ξ 影响）的需求差异 = 纯随机抽样差异；正文多种子标准差 = 训练随机性；
  3 情景对比      各情景的主要指标（全天与峰 / 平 / 谷）；
  4 三步分解      总效应（正文 − 平坦）= 电价成本 + 需求调整 + 策略重新学习，正向与反向两条路径取平均，
                  全天与峰 / 平 / 谷分别给出，测试日自助法 95% 区间；中间状态只是分解的过渡点，不作为现实情景解读；
  5 因素拆分      正文内部逐（日, 时, 卡型）回归：一级价 / 价差 ~ 供需紧张度 + 单位电费 + 交互，及偏 R²；
                  并比较两个情景中价格对供需紧张度的斜率（电价是否放大或减弱供需信号）；
  6 电价传导      正文与 cf_cost（同一需求）逐（卡型, 时）的价格差对单位电费差的回归斜率，对照出售底价的机械传导；
  7 分时段成交    正文条件下 MARL 与规则基准在峰 / 平 / 谷的成交、兜底与闲置售出，MARL 新增成交的时段分布。

产出：results/energy/eval/<评估名>/<方法>/、results/energy/tables/*.csv、figures/energy/*
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from . import specs
from .collector import Collector
from .env import MarketEnv
from .experiments import results_root, job_dir, get_cfg, _dataset, run_group, _guarded, _run_parallel
from .runner import load_policy, warm_env, run_day

SEED = C.MAIN_SEED
RULES = ['cda', 'sdr', 'mmr', 'mcafee']
LABEL = {'marl': 'MARL', 'cda': 'CDA', 'sdr': 'SDR', 'mmr': 'MMR', 'mcafee': 'McAfee'}
PERIODS = ['peak', 'flat', 'valley']
N_BOOT = 1000

# 评估名: (环境配置来源实验, 对配置的改动, 策略来源实验, 说明)
CASES = {
    'main': ('main', {}, 'main', 'TOU, ξ = 0.1 (main)'),
    'flat': ('sens_tariff_flat', {}, 'sens_tariff_flat', 'Flat, ξ = 0'),
    'pv1.5': ('energy_pv_1.5', {}, 'energy_pv_1.5', 'Peak–valley × 1.5, ξ = 0.15'),
    'pv2': ('energy_pv_2', {}, 'energy_pv_2', 'Peak–valley × 2, ξ = 0.2'),
    'cf_cost': ('main', {'tariff_mode': 'flat'}, 'main', 'Flat tariff + main demand, main policy'),
    'cf_flat': ('sens_tariff_flat', {}, 'main', 'Flat tariff + flat demand, main policy'),
    'cf_pv1.5': ('energy_pv_1.5', {}, 'main', 'Peak–valley × 1.5, main policy'),
    'cf_pv2': ('energy_pv_2', {}, 'main', 'Peak–valley × 2, main policy'),
    'rv_cost': ('sens_tariff_flat', {'tariff_mode': 'tou'}, 'sens_tariff_flat', 'TOU tariff + flat demand, flat policy'),
    'rv_demand': ('main', {}, 'sens_tariff_flat', 'TOU tariff + main demand, flat policy'),
}
# 三步分解（分时 − 平坦）的两条路径：每一步只改变一个因素，三步之和恰为 正文 − 平坦
#   正向：正文 →(换电价) cf_cost →(换需求) cf_flat →(换策略) flat
#   反向：平坦 →(换电价) rv_cost →(换需求) rv_demand →(换策略) 正文
STEPS = {'electricity cost': (('main', 'cf_cost'), ('rv_cost', 'flat')),
         'demand adjustment': (('cf_cost', 'cf_flat'), ('rv_demand', 'rv_cost')),
         'policy re-learning': (('cf_flat', 'flat'), ('main', 'rv_demand'))}
AMP = [('flat', 'cf_flat', 0.0), ('main', 'main', 1.0), ('pv1.5', 'cf_pv1.5', 1.5), ('pv2', 'cf_pv2', 2.0)]   # (长期, 短期, k)


def case_cfg(case, smoke=False):
    src, over, _, _ = CASES[case]
    cfg = get_cfg(src, smoke)
    return cfg.replace(**over) if over else cfg


def energy_root(smoke=False):
    return results_root(smoke) / 'energy'


def tables_dir(smoke=False):
    d = energy_root(smoke) / 'tables'; d.mkdir(parents=True, exist_ok=True)
    return d


def eval_path(case, method, smoke=False):
    return energy_root(smoke) / 'eval' / case / method


# ───────────────────────────── 逐时面板采集 ─────────────────────────────
class EnergyCollector(Collector):
    """在正文 Collector 的全部产物之外，记录逐（日 × 时 × 卡型）与逐（日 × 时）面板；只读取记录信息，不改变决策与结算。"""

    def __init__(self, env, method):
        super().__init__(env, method)
        cfg = env.cfg
        self.kwh = np.where(env.is_pros, cfg.eta * specs.TDP[env.own0] / 1000.0 / specs.F[env.own0], 0.0)  # kWh / TFLOP·h
        self.com = (env.is_pros & (env.tclass == 1)).astype(float)
        self.rows_hg, self.rows_h = [], []

    def add(self, info):
        super().add(info)
        env = self.env; G = specs.G; h = int(info['hour']); day = int(self.day_acc['day'])
        e_unit = np.where(env.is_pros, env.curves[env.tclass, h % 24], 0.0) * self.kwh     # 元/TFLOP·h（单位算力电费）
        log = info['p1_log']
        if log:
            arr = np.array([(i, g, q) for (m, i, g, q, pb, ps) in log], dtype=np.float64)
            ii, gg, qq = arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]
        else:
            ii = gg = np.zeros(0, int); qq = np.zeros(0)

        def by_g(w):
            return np.bincount(gg, weights=w, minlength=G) if len(gg) else np.zeros(G)
        sold, sold_com = by_g(qq), by_g(qq * self.com[ii])
        sold_e, sold_kwh = by_g(qq * e_unit[ii]), by_g(qq * self.kwh[ii])
        idle = np.where(info['ask_mask'] & env.is_pros, info['idle'], 0.0)                 # 挂牌出售的闲置算力
        offer = np.bincount(env.own0, weights=idle, minlength=G)
        offer_floor = np.bincount(env.own0, weights=idle * info['floor_sell'], minlength=G)
        for g in range(G):
            self.rows_hg.append(dict(day=day, hour=h, gpu=g, p1_price=float(info['p1_price'][g]),
                                     p1_vol=float(info['p1_vol_g'][g]), p2_price=float(info['p2_price'][g]),
                                     p2_vol=float(info['p2_vol_g'][g]), ecc_vol=float(info['ecc_vol_g'][g]),
                                     sold=sold[g], sold_com=sold_com[g], sold_e=sold_e[g], sold_kwh=sold_kwh[g],
                                     offer=offer[g], offer_floor=offer_floor[g]))
        d = info['d']
        self.rows_h.append(dict(day=day, hour=h, period=specs.PERIOD_NAMES[specs.PERIOD[h % 24]], W=info['W'],
                                demand=info['demand'], demand_buyers=float(d[~env.is_pros].sum()),
                                demand_pros=float(d[env.is_pros].sum()), tight=info['demand'] / env.total_cap,
                                p1_vol=info['p1_vol'], p2_vol=info['p2_vol'], ecc_vol=info['ecc_fallback_vol'],
                                idle_total=info['idle_total'], sold_total=info['sold_total']))

    def save(self, out):
        summ = super().save(out)
        pd.DataFrame(self.rows_hg).to_csv(Path(out) / 'panel_hg.csv.gz', index=False)
        pd.DataFrame(self.rows_h).to_csv(Path(out) / 'panel_h.csv.gz', index=False)
        return summ


# ───────────────────────────── 评估 ─────────────────────────────
def energy_eval_job(case, method, smoke=False, verbose=True):
    """与 runner.evaluate 完全相同的评估协议（新建环境 → 验证集最后 warmup_days 天预热 → 逐日评估测试集），采集器换成 EnergyCollector。"""
    out = eval_path(case, method, smoke)
    if (out / 'DONE').exists():
        return f'{case}/{method}', SEED, 'skip', 0.0
    t0 = time.time()
    cfg = case_cfg(case, smoke)
    ds = _dataset(cfg)
    learner, ckpt = None, None
    if method == 'marl':
        ckpt = job_dir(CASES[case][2], SEED, smoke) / 'final.pt'
        if not ckpt.exists():
            raise FileNotFoundError(f'缺少策略 {ckpt}（先运行 run_energy_training）')
        learner = load_policy(cfg, ds, SEED, ckpt)
    env = MarketEnv(cfg, ds, seed=SEED)
    rule = None if method == 'marl' else method
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        warm_env(env, cfg, ds, 'val', learner=learner, rule=rule)
        col = EnergyCollector(env, method)
        for k in range(cfg.test_days):
            run_day(env, cfg, ds, 'test', k, learner=learner, rule=rule, collector=col)
    summ = col.save(out)
    with open(out / 'summary.json', 'w') as f:
        json.dump(summ, f, indent=1, default=float)
    with open(out / 'job.json', 'w') as f:
        json.dump(dict(case=case, method=method, seed=SEED, ckpt=str(ckpt) if ckpt else None, data=ds['path'],
                       cfg=cfg.to_dict()), f, indent=1, default=str)
    (out / 'DONE').write_text('ok')
    if verbose:
        print(f'   [{case}/{method}] 单位福利 {summ["unit_welfare"]:.2f}', flush=True)
    return f'{case}/{method}', SEED, 'ok', (time.time() - t0) / 60


def _worker(args):
    case, method, smoke, threads = args
    return _guarded(energy_eval_job, (case, method, smoke), eval_path(case, method, smoke).parent / f'{method}.log', threads)


def run_energy_training(workers=1, smoke=False, threads=1):
    """只训练两个峰谷差放大情景（单种子）；main 与 sens_tariff_flat 已有，自动跳过。"""
    names = ['main', 'sens_tariff_flat'] + [f'energy_pv_{C._fmt(k)}' for k in C.ENERGY_PV]
    return run_group(names, workers=workers, smoke=smoke, threads=threads)


def run_energy_eval(workers=1, smoke=False, threads=1):
    jobs = [(c, 'marl') for c in CASES] + [('main', m) for m in RULES]
    for c in CASES:
        _dataset(case_cfg(c, smoke))                                    # 主进程先载入数据
    todo = [(c, m, smoke, threads) for c, m in jobs if not (eval_path(c, m, smoke) / 'DONE').exists()]
    print(f'📋 评估作业 {len(todo)} 个', flush=True)
    res = [energy_eval_job(*j[:3]) for j in todo] if workers <= 1 else \
        _run_parallel(_worker, todo, workers, lambda j: f'{j[0]}/{j[1]}')
    bad = [r for r in res if r[2] == 'error']
    if bad:
        raise RuntimeError(f'{len(bad)} 个评估失败，见各目录下的 ERROR.txt：{bad}')
    print('✅ 评估完成', flush=True)
    return res


# ───────────────────────────── 读取与指标 ─────────────────────────────
def load_panel(case, method='marl', smoke=False, kind='hg'):
    p = eval_path(case, method, smoke)
    return pd.read_csv(p / f'panel_{kind}.csv.gz') if (p / 'DONE').exists() else None


def load_summ(case, method='marl', smoke=False):
    p = eval_path(case, method, smoke)
    return json.load(open(p / 'summary.json')) if (p / 'DONE').exists() else None


STATS = ['one', 'hrs', 'W', 'demand', 'p1_num', 'p1_vol', 'p2_num', 'p2_vol', 'spr_num', 'spr_den', 'ecc_vol', 'flow',
         'idle', 'sold_total', 'sold', 'sold_com', 'sold_e', 'sold_kwh', 'rev_p1']
METRICS = {   # 指标 = 系数 × Σ分子 / Σ分母（比值之比，与正文统计口径一致）
    'unit_welfare': ('W', 'demand', 1e4, 'Unit welfare'),
    'welfare_day': ('W', 'one', 1.0, 'Welfare (CNY/day)'),
    'p1_rel': ('p1_num', 'p1_vol', 1.0, 'P1 price / ECC retail'),
    'p2_rel': ('p2_num', 'p2_vol', 1.0, 'P2 price / ECC retail'),
    'spread': ('spr_num', 'spr_den', 1.0, 'Retail spread (CNY/(TFLOP·h))'),
    'p2_vol_day': ('p2_vol', 'one', 1.0, 'Retail volume (TFLOP·h/day)'),
    'ecc_vol_day': ('ecc_vol', 'one', 1.0, 'ECC fallback (TFLOP·h/day)'),
    'p2_vol_h': ('p2_vol', 'hrs', 1.0, 'Retail volume (TFLOP·h per hour)'),
    'ecc_vol_h': ('ecc_vol', 'hrs', 1.0, 'ECC fallback (TFLOP·h per hour)'),
    'internal_share': ('p2_vol', 'flow', 1.0, 'Internal flow share'),
    'idle_sold': ('sold_total', 'idle', 1.0, 'Share of idle capacity sold'),
    'com_share': ('sold_com', 'sold', 1.0, 'Commercial-tariff share of P1 sales'),
    'elec_price_sold': ('sold_e', 'sold_kwh', 1.0, 'Avg. tariff of sold compute (CNY/kWh)'),
    'elec_share_p1': ('sold_e', 'rev_p1', 1.0, 'Electricity cost / P1 price'),
}
# 图中坐标轴名称：与正文图 8 注、支撑材料表 S21 的指标名称一致（METRICS 中的简称只用于表格列名）
FIG_LABEL = {
    'unit_welfare': 'Welfare per unit of computing power',
    'p1_rel': 'Primary price / ECC retail price',
    'p2_rel': 'Secondary price / ECC retail price',
    'idle_sold': 'Share of idle computing power sold',
    'com_share': 'Share of industrial and commercial sellers',
    'elec_share_p1': 'Electricity cost / primary transaction value',
}


def _flab(key):
    return FIG_LABEL.get(key, METRICS[key][3])


KEYS = ['unit_welfare', 'welfare_day', 'p1_rel', 'p2_rel', 'spread', 'p2_vol_day', 'ecc_vol_day', 'internal_share',
        'idle_sold', 'com_share', 'elec_price_sold', 'elec_share_p1']
SUMM_KEYS = ['buyer_savings_pc', 'prosumer_savings_pc', 'inter_profit_avg']


def hour_stats(case, method='marl', smoke=False):
    """逐（日, 时）的充分统计量；缺结果时返回 None。"""
    ph, x = load_panel(case, method, smoke, 'h'), load_panel(case, method, smoke, 'hg')
    if ph is None:
        return None
    pes = specs.P_ES[x['gpu'].values]
    x['p1_num'] = x['p1_price'] * x['p1_vol'] / pes
    x['p2_num'] = x['p2_price'] * x['p2_vol'] / pes
    both = (x['p1_vol'] > 0) & (x['p2_vol'] > 0)
    x['spr_num'] = np.where(both, (x['p2_price'] - x['p1_price']) * x['p2_vol'], 0.0)
    x['spr_den'] = np.where(both, x['p2_vol'], 0.0)
    x['rev_p1'] = x['p1_price'] * x['p1_vol']
    cols = ['p1_num', 'p1_vol', 'p2_num', 'p2_vol', 'spr_num', 'spr_den', 'sold', 'sold_com', 'sold_e', 'sold_kwh', 'rev_p1']
    b = x.groupby(['day', 'hour'])[cols].sum().reset_index()
    a = ph[['day', 'hour', 'period', 'W', 'demand', 'demand_buyers', 'demand_pros', 'tight', 'ecc_vol', 'idle_total',
            'sold_total']].rename(columns={'idle_total': 'idle'})
    m = a.merge(b, on=['day', 'hour'], how='left').fillna(0.0)
    m['one'] = 1.0 / 24.0; m['hrs'] = 1.0; m['flow'] = m['p2_vol'] + m['ecc_vol']
    return m


def _metric(sums, key):
    num, den, k, _ = METRICS[key]
    return k * sums[num] / sums[den] if sums[den] > 0 else np.nan


def metrics_of(hs, keys=KEYS):
    s = hs[STATS].sum()
    return {k: _metric(s, k) for k in keys}


def _robust_sd(smoke=False):
    """正文多种子实验（robust）中 MARL 各指标的种子间标准差：长期结果的训练随机性参照。"""
    from .analysis import load_summaries
    S = load_summaries('robust', 'marl', smoke)
    keys = ['unit_welfare', 'welfare_per_day', 'p1_price_rel', 'internal_flow_share', 'idle_sold_share'] + SUMM_KEYS
    return {k: float(np.std([s[k] for s in S], ddof=1)) for k in keys} if len(S) > 1 else {}


# ───────────────────────────── 1 复现核查 ─────────────────────────────
def check_reproduction(smoke=False, tol=1e-6):
    """用自己训练的策略、在自己的条件下评估的情景（以及正文规则基准），与已有评估逐项比较，应完全一致。"""
    keys = ['unit_welfare', 'welfare_per_day', 'p2_vol', 'ecc_fallback_vol', 'inter_profit_avg', 'idle_sold_share']
    rows = []
    for case, method in [(c, 'marl') for c in ('main', 'flat', 'pv1.5', 'pv2')] + [('main', m) for m in RULES]:
        e = eval_path(case, method, smoke) / 'summary.json'
        ref = job_dir(CASES[case][0], SEED, smoke) / f'eval_{method}' / 'summary.json'
        if not (e.exists() and ref.exists()):
            continue
        a, b = json.load(open(e)), json.load(open(ref))
        diff = max(abs(a[k] - b[k]) / max(abs(b[k]), 1.0) for k in keys)
        rows.append(dict(case=case, method=method, reference=str(ref.parent.relative_to(results_root(smoke))),
                         unit_welfare=a['unit_welfare'], unit_welfare_reference=b['unit_welfare'], max_rel_diff=diff,
                         identical=diff <= tol))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tables_dir(smoke) / 'reproduction_check.csv', index=False)
        print(f"复现核查：{len(df)} 项，{int((~df['identical']).sum())} 项不一致")
    return df


# ───────────────────────────── 2 噪声参照 ─────────────────────────────
def table_noise(smoke=False):
    """两套需求数据（正文 ξ = 0.1 / 平坦 ξ = 0）的差异：纯买方需求不受 ξ 影响，其差异即两次随机抽样的差异（数据噪声尺子）；
    产消者需求的差异 = 负荷转移 + 随机抽样差异。按峰 / 平 / 谷与全天列出相对差（平坦 − 正文）/ 正文；
    另附正文多种子标准差（训练随机性尺子）。"""
    a, b = hour_stats('main', smoke=smoke), hour_stats('flat', smoke=smoke)
    if a is None or b is None:
        return None
    rows = []
    for per in PERIODS + ['all']:
        sa = a if per == 'all' else a[a.period == per]; sb = b if per == 'all' else b[b.period == per]
        r = dict(period=per)
        for col in ('demand_buyers', 'demand_pros', 'demand'):
            r[f'rel_diff_{col}'] = float(sb[col].sum() / sa[col].sum() - 1)
        rows.append(r)
    da, db = a.groupby('day')['demand_buyers'].sum().values, b.groupby('day')['demand_buyers'].sum().values
    df = pd.DataFrame(rows)
    df['buyers_daily_corr'] = float(np.corrcoef(da, db)[0, 1]) if len(da) == len(db) and len(da) > 2 else np.nan
    for k, v in _robust_sd(smoke).items():
        df[f'seed_sd_{k}'] = v
    df.to_csv(tables_dir(smoke) / 'table_noise.csv', index=False)
    return df


# ───────────────────────────── 3 情景对比 ─────────────────────────────
def table_scenarios(smoke=False):
    """各评估（长期情景与正文策略的反事实）的主要指标，全天与峰 / 平 / 谷；含相对正文的差值。"""
    rows = []
    for c in CASES:
        hs = hour_stats(c, smoke=smoke); S = load_summ(c, smoke=smoke)
        if hs is None:
            continue
        for per in ['all'] + PERIODS:
            sub = hs if per == 'all' else hs[hs.period == per]
            rows.append(dict(case=c, label=CASES[c][3], period=per, **metrics_of(sub),
                             **({k: S[k] for k in SUMM_KEYS} if per == 'all' else {})))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    base = df[df.case == 'main'].set_index('period')
    for k in KEYS:
        df[f'd_{k}'] = df[k].values - base.loc[df.period, k].values
    df.to_csv(tables_dir(smoke) / 'table_scenarios.csv', index=False)
    return df


# ───────────────────────────── 4 三步分解 ─────────────────────────────
DEC_KEYS = ['unit_welfare', 'welfare_day', 'p1_rel', 'p2_rel', 'p2_vol_day', 'ecc_vol_day', 'internal_share', 'idle_sold',
            'com_share', 'elec_share_p1']


def table_decomposition(smoke=False, keys=DEC_KEYS):
    """分时电价相对平坦电价的总效应（正文 − 平坦，两者各自自洽：需求与策略都与各自的电价相匹配）分解为三步：
      电价成本     只换电价（需求、策略不变）；
      需求调整     再换成与新电价匹配的需求（负荷转移，含两套数据的随机抽样差异，见 table_noise）；
      策略重新学习 再换成在新电价下学到的策略（含训练随机性）。
    正向（从正文出发）与反向（从平坦出发）各走一遍，两条路径各自三步之和都恰为总效应；报告两条路径及其平均。
    中间状态只是分解的过渡点，不作为现实情景解读；两条路径的电价成本一步若接近，说明结论不依赖用哪套需求与策略。
    全天与峰 / 平 / 谷分别计算；95% 区间对测试日自助重抽样（同一组日期用于全部评估）。"""
    cases = ['main', 'flat', 'cf_cost', 'cf_flat', 'rv_cost', 'rv_demand']
    H = {c: hour_stats(c, smoke=smoke) for c in cases}
    if any(v is None for v in H.values()):
        miss = [c for c, v in H.items() if v is None]
        print(f'  （三步分解缺少评估：{miss}）')
        return None
    days = sorted(set.intersection(*[set(v.day) for v in H.values()]))
    idx = np.random.default_rng(2026).integers(0, len(days), size=(N_BOOT, len(days)))
    kidx = {k: i for i, k in enumerate(STATS)}

    def ratio(x, key):
        num, den, kk, _ = METRICS[key]
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(x[..., kidx[den]] > 0, kk * x[..., kidx[num]] / x[..., kidx[den]], np.nan)
    rows = []
    for per in ['all'] + PERIODS:
        A = {c: (v if per == 'all' else v[v.period == per]).groupby('day')[STATS].sum().reindex(days, fill_value=0.0).values
             for c, v in H.items()}
        P = {c: A[c].sum(0) for c in cases}; B = {c: A[c][idx].sum(1) for c in cases}
        for key in keys:
            pv = {c: float(ratio(P[c], key)) for c in cases}; bv = {c: ratio(B[c], key) for c in cases}
            r = dict(period=per, metric=key, main=pv['main'], flat=pv['flat'], total=pv['main'] - pv['flat'])
            bt = bv['main'] - bv['flat']
            r['total lo'], r['total hi'] = float(np.nanpercentile(bt, 2.5)), float(np.nanpercentile(bt, 97.5))
            for step, ((a1, b1), (a2, b2)) in STEPS.items():
                f, rv = pv[a1] - pv[b1], pv[a2] - pv[b2]
                bf, br = bv[a1] - bv[b1], bv[a2] - bv[b2]
                r[f'{step} (forward)'] = f; r[f'{step} (reverse)'] = rv; r[step] = (f + rv) / 2
                bm = (bf + br) / 2
                r[f'{step} lo'], r[f'{step} hi'] = float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir(smoke) / 'table_decomposition.csv', index=False)
    return df


def table_amplitude(smoke=False, keys=('unit_welfare', 'p1_rel', 'p2_vol_day', 'idle_sold', 'com_share', 'elec_share_p1')):
    """峰谷差幅度 k（0 = 平坦，1 = 正文；ξ = 0.1k）下，全天与峰 / 平 / 谷的指标：长期（各自训练）与短期（正文策略）。"""
    rows = []
    for lc, sc, k in AMP:
        for hz, c in (('retrained', lc), ('main policy', sc)):
            hs = hour_stats(c, smoke=smoke)
            if hs is None:
                continue
            for per in ['all'] + PERIODS:
                rows.append(dict(k=k, horizon=hz, case=c, period=per, **metrics_of(hs if per == 'all' else hs[hs.period == per], keys)))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tables_dir(smoke) / 'table_amplitude.csv', index=False)
    return df


# ───────────────────────────── 5 因素拆分 ─────────────────────────────
def _unit_elec(cfg, w):
    """[G, 24]：卡型 g 在时段 h 的产能加权单位算力电费（元/TFLOP·h）。"""
    cur = specs.tariff_curves(cfg.tariff_mode, cfg.tariff_pv_scale)
    return (w @ cur) * (cfg.eta * specs.TDP / 1000.0 / specs.F)[:, None]


def _cap_weights(smoke):
    """各卡型中居民 / 工商业电价卖方的产能占比 [G, 2]（外生权重）。"""
    cfg = get_cfg('main', smoke)
    env = MarketEnv(cfg, _dataset(cfg), seed=SEED)
    w = np.array([[env.cap[env.is_pros & (env.own0 == g) & (env.tclass == c)].sum() for c in (0, 1)] for g in range(specs.G)])
    return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)


def _cell_panel(case, smoke, w):
    """逐（日, 时, 卡型）面板：一级价 / ECC 零售价、价差 / ECC 零售价、供需紧张度、单位电费 / ECC 零售价、成交量（权重）。"""
    x = load_panel(case, smoke=smoke, kind='hg'); ph = load_panel(case, smoke=smoke, kind='h')
    if x is None:
        return None
    E = _unit_elec(case_cfg(case, smoke), w)
    x = x.merge(ph[['day', 'hour', 'tight']], on=['day', 'hour'])
    pes = specs.P_ES[x.gpu.values]
    x['p1'] = x.p1_price / pes
    x['spr'] = np.where((x.p1_vol > 0) & (x.p2_vol > 0), (x.p2_price - x.p1_price) / pes, np.nan)
    x['elec'] = E[x.gpu.values, x.hour.values % 24] / pes
    return x


def _ols(df, y, X, wcol):
    """加权最小二乘（卡型固定效应 + 自变量），返回系数与 R²。"""
    d = df[np.isfinite(df[y]) & (df[wcol] > 0)]
    if len(d) < len(X) + specs.G + 2:
        return None
    Z = np.c_[pd.get_dummies(d.gpu).values.astype(float), d[X].values] if X else pd.get_dummies(d.gpu).values.astype(float)
    sw = np.sqrt(d[wcol].values)
    beta = np.linalg.lstsq(Z * sw[:, None], d[y].values * sw, rcond=None)[0]
    res = d[y].values - Z @ beta
    yb = np.average(d[y].values, weights=d[wcol].values)
    r2 = 1 - np.sum(d[wcol].values * res ** 2) / np.sum(d[wcol].values * (d[y].values - yb) ** 2)
    return dict(zip(X, beta[-len(X):])) if X else {}, float(r2)


def table_factors(smoke=False):
    """正文内部的因素拆分（逐日 × 时 × 卡型，权重 = 成交量，含卡型固定效应）：
        y = a_g + b1·紧张度 + b2·单位电费 + b3·紧张度×单位电费
      报告标准化系数（自变量的一个标准差带来的 y 变化）与各项的偏 R²（去掉该项后 R² 的下降）。
      识别来源：两个谷段需求高低不同、早晚高峰需求不同、同一时段逐日的随机波动，使供需与电价并不同步。
    另对各情景分别估计 y = a_g + b·紧张度，比较分时 / 平坦 / 峰谷差放大情景中价格对供需信号的反应强度。"""
    w = _cap_weights(smoke)
    rows = []
    x = _cell_panel('main', smoke, w)
    if x is not None:
        x['inter'] = (x.tight - x.tight.mean()) * (x.elec - x.elec.mean())
        for y, wc in (('p1', 'p1_vol'), ('spr', 'p2_vol')):
            full = _ols(x, y, ['tight', 'elec', 'inter'], wc)
            if full is None:
                continue
            coef, r2 = full
            r = dict(analysis='main: factors', y=y, r2_full=r2, r2_card_only=_ols(x, y, [], wc)[1])
            for v, drop in (('tight', ['elec', 'inter']), ('elec', ['tight', 'inter']), ('inter', ['tight', 'elec'])):
                d = x[np.isfinite(x[y]) & (x[wc] > 0)]
                r[f'std_coef_{v}'] = coef[v] * float(d[v].std())
                r[f'partial_r2_{v}'] = r2 - _ols(x, y, drop, wc)[1]
            rows.append(r)
    for c in ('flat', 'main', 'pv1.5', 'pv2'):
        xc = _cell_panel(c, smoke, w)
        if xc is None:
            continue
        for y, wc in (('p1', 'p1_vol'), ('spr', 'p2_vol')):
            res = _ols(xc, y, ['tight'], wc)
            if res is not None:
                rows.append(dict(analysis='slope on tightness', case=c, y=y, slope_tight=res[0]['tight'], r2_full=res[1]))
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tables_dir(smoke) / 'table_factors.csv', index=False)
    return df


# ───────────────────────────── 6 电价传导 ─────────────────────────────
def table_pass_through(smoke=False, n_boot=300):
    """正文与 cf_cost（同一正文策略、同一需求，只差电价）逐（卡型, 时）的价格差对单位电费差的加权回归斜率（含截距）：
      p1 = 一级成交价（市场结果）；floor = 挂牌卖方的平均出售底价（保留价规则的机械传导，作参照）；p2 = 二级成交价。
      权重 = 两者成交（或挂牌）量的较小者；95% 区间对测试日自助重抽样。"""
    w = _cap_weights(smoke)
    A, B = load_panel('main', smoke=smoke), load_panel('cf_cost', smoke=smoke)
    if A is None or B is None:
        return None
    dE = (_unit_elec(case_cfg('main', smoke), w) - _unit_elec(case_cfg('cf_cost', smoke), w)).ravel()
    days = np.array(sorted(set(A.day) & set(B.day)))

    def cube(df, num, vol):
        n = np.zeros((len(days), specs.G, 24)); v = np.zeros_like(n)
        di = np.searchsorted(days, df.day.values); g = df.gpu.values; h = df.hour.values % 24
        np.add.at(n, (di, g, h), (df[num] * (df[vol] if num != 'offer_floor' else 1.0)).values)
        np.add.at(v, (di, g, h), df[vol].values)
        return n, v
    spec = {'p1': ('p1_price', 'p1_vol'), 'floor': ('offer_floor', 'offer'), 'p2': ('p2_price', 'p2_vol')}
    CA = {k: cube(A, *s) for k, s in spec.items()}; CB = {k: cube(B, *s) for k, s in spec.items()}

    def slope(dy, ww):
        ok = np.isfinite(dy) & (ww > 0)
        if ok.sum() < 3:
            return np.nan
        X = np.c_[np.ones(ok.sum()), dE[ok]]; sw = np.sqrt(ww[ok])
        return float(np.linalg.lstsq(X * sw[:, None], dy[ok] * sw, rcond=None)[0][1])

    def est(di):
        o = {}
        with np.errstate(invalid='ignore', divide='ignore'):
            for k in spec:
                (na, va), (nb, vb) = CA[k], CB[k]
                dy = (na[di].sum(0) / va[di].sum(0) - nb[di].sum(0) / vb[di].sum(0)).ravel()
                o[k] = slope(dy, np.minimum(va[di].sum(0), vb[di].sum(0)).ravel())
        return o
    pt = est(np.arange(len(days)))
    rng = np.random.default_rng(7)
    bs = [est(rng.integers(0, len(days), len(days))) for _ in range(n_boot)]
    r = dict(comparison='main vs flat tariff (same demand, main policy)', mean_abs_dE=float(np.abs(dE).mean()))
    for k, v in pt.items():
        bv = np.array([b[k] for b in bs], float)
        r.update({k: v, f'{k}_lo': float(np.nanpercentile(bv, 2.5)), f'{k}_hi': float(np.nanpercentile(bv, 97.5))})
    df = pd.DataFrame([r])
    df.to_csv(tables_dir(smoke) / 'table_pass_through.csv', index=False)
    return df


# ───────────────────────────── 7 分时段成交 ─────────────────────────────
def table_period(smoke=False):
    """正文条件下各定价方法在峰 / 平 / 谷的每小时成交量、兜底量、闲置售出率、价格与所售算力电价；MARL 相对 CDA 新增成交的时段分布。"""
    keys = ['p2_vol_h', 'ecc_vol_h', 'internal_share', 'idle_sold', 'p1_rel', 'p2_rel', 'spread', 'elec_price_sold',
            'unit_welfare']
    rows, H = [], {}
    for m in ['marl'] + RULES:
        hs = hour_stats('main', m, smoke)
        if hs is None:
            continue
        H[m] = hs
        rows += [dict(method=LABEL[m], period=per, **metrics_of(hs if per == 'all' else hs[hs.period == per], keys))
                 for per in PERIODS + ['all']]
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(tables_dir(smoke) / 'table_period_main.csv', index=False)
    if 'marl' in H and 'cda' in H:
        ext = []
        for k, per in enumerate(PERIODS):
            nh = int((specs.PERIOD == k).sum())
            a = metrics_of(H['marl'][H['marl'].period == per], ['p2_vol_h', 'ecc_vol_h'])
            b = metrics_of(H['cda'][H['cda'].period == per], ['p2_vol_h', 'ecc_vol_h'])
            ext.append(dict(period=per, hours=nh, d_p2_vol_per_day=(a['p2_vol_h'] - b['p2_vol_h']) * nh,
                            d_ecc_vol_per_day=(a['ecc_vol_h'] - b['ecc_vol_h']) * nh))
        ext = pd.DataFrame(ext)
        ext['share_of_extra_p2'] = ext.d_p2_vol_per_day / ext.d_p2_vol_per_day.sum()
        ext['share_of_hours'] = ext.hours / 24
        ext.to_csv(tables_dir(smoke) / 'table_period_extra_vs_cda.csv', index=False)
    return df


# ───────────────────────────── 图 ─────────────────────────────
def _fg():
    from . import figures as Fg
    return Fg


def _hourly(case, keys, smoke, method='marl'):
    hs = hour_stats(case, method, smoke)
    if hs is None:
        return None
    return pd.DataFrame([dict(hour=h, **metrics_of(g, keys)) for h, g in hs.groupby('hour')]).set_index('hour').reindex(range(24))


def fig_hourly_tariff(smoke=False, name='Fig_energy_hourly', keys=('p1_rel', 'com_share', 'idle_sold', 'elec_share_p1')):
    """逐时对比：分时电价（正文）与平坦电价（各自的需求与策略）为主；"只换电价"（平坦电价 + 正文需求与策略）作参照。"""
    Fg = _fg(); plt = Fg.plt
    lines = [('main', '-o', '#1f4e79', 1.4, 'TOU tariff (main)'), ('flat', '--s', '#c55a11', 1.4, 'Flat tariff'),
             ('cf_cost', ':', '#7f7f7f', 1.0, 'Flat tariff, main demand & policy (cost only)')]
    D = {c: _hourly(c, keys, smoke) for c, *_ in lines}
    if D['main'] is None or D['flat'] is None:
        return Fg._skip(name)
    fig, axs = plt.subplots(1, len(keys), figsize=(2.9 * len(keys), 2.6))
    for ax, k in zip(axs, keys):
        Fg._shade_periods(ax)
        for c, sty, col, lw, lab in lines:
            if D[c] is not None:
                ax.plot(range(24), D[c][k], sty, ms=2.2, lw=lw, color=col, label=lab)
        ax.set(xlabel='Hour', ylabel=_flab(k), xlim=(-0.5, 23.5)); ax.set_xticks(range(0, 24, 4))
    Fg._panel_labels(axs)
    fig.tight_layout()
    hs_, ls_ = axs[0].get_legend_handles_labels()
    Fg._bottom_legend(fig, hs_, ls_, y=0.0)
    Fg._save(fig, name, smoke, 'energy', {c: v.reset_index() for c, v in D.items() if v is not None})


def fig_amplitude(smoke=False, name='Fig_energy_amplitude', keys=('com_share', 'idle_sold', 'unit_welfare', 'elec_share_p1')):
    """峰谷差倍数 Ψ（论文记号；数据列名仍为 k）的影响，峰段与谷段分开：实线 = 各自训练，虚线 = 正文策略。"""
    Fg = _fg(); plt = Fg.plt
    df = table_amplitude(smoke)
    if df is None or not len(df):
        return Fg._skip(name)
    fig, axs = plt.subplots(1, len(keys), figsize=(2.9 * len(keys), 2.7))
    for ax, key in zip(axs, keys):
        for per, col in (('peak', '#c0504d'), ('valley', '#2f6f8f')):
            for hz, sty in (('retrained', '-o'), ('main policy', ':s')):
                g = df[(df.period == per) & (df.horizon == hz)].sort_values('k')
                ax.plot(g.k, g[key], sty, ms=3, color=col, label=f'{per.capitalize()}, {hz}')
        ax.set(xlabel=r'Peak–valley difference multiplier $\Psi$ ($\xi = 0.1\Psi$)', ylabel=_flab(key)); ax.set_xticks([0, 1, 1.5, 2])
    Fg._panel_labels(axs)
    fig.tight_layout()
    hs_, ls_ = axs[0].get_legend_handles_labels()
    Fg._bottom_legend(fig, hs_, ls_, ncol=4, y=0.0)
    Fg._save(fig, name, smoke, 'energy', df)


def fig_decomposition(smoke=False, name='Fig_energy_decomposition', keys=('unit_welfare', 'com_share', 'idle_sold', 'p1_rel')):
    """三步分解（正反两条路径的平均）：分时 − 平坦 的总效应按峰 / 平 / 谷 / 全天拆成电价成本、需求调整、策略重新学习；
    黑点为总效应，误差线为 95% 区间。"""
    Fg = _fg(); plt = Fg.plt
    df = table_decomposition(smoke)
    if df is None or not len(df):
        return Fg._skip(name)
    steps = [('electricity cost', '#c55a11'), ('demand adjustment', '#548235'), ('policy re-learning', '#9e9e9e')]
    pers = ['peak', 'flat', 'valley', 'all']
    fig, axs = plt.subplots(1, len(keys), figsize=(2.9 * len(keys), 2.8))
    for ax, key in zip(axs, keys):
        d = df[df.metric == key].set_index('period').reindex(pers)
        x = np.arange(len(pers)); w = 0.25
        for j, (st, col) in enumerate(steps):
            y = d[st].values
            ax.bar(x + (j - 1) * w, y, w, color=col, label=st.capitalize(),
                   yerr=np.vstack([y - d[f'{st} lo'].values, d[f'{st} hi'].values - y]), error_kw=dict(lw=0.5, capsize=1.5))
        ax.plot(x, d['total'].values, 'ko', ms=3.5, label='Total (TOU − flat)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(['Peak', 'Flat', 'Valley', 'All']); ax.set_ylabel(f'Δ {_flab(key)}')
    Fg._panel_labels(axs)
    fig.tight_layout()
    hs_, ls_ = axs[0].get_legend_handles_labels()
    Fg._bottom_legend(fig, hs_, ls_, ncol=4, y=0.0)
    Fg._save(fig, name, smoke, 'energy', df)


def fig_period(smoke=False, name='Fig_energy_period_methods'):
    Fg = _fg(); plt = Fg.plt
    df = table_period(smoke)
    if df is None or not len(df):
        return Fg._skip(name)
    d = df[df.period != 'all']; ms = list(dict.fromkeys(d.method))
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.6)); x = np.arange(3); wd = 0.8 / len(ms)
    for ax, k in zip(axs, ('p2_vol_h', 'ecc_vol_h', 'idle_sold')):
        for j, m in enumerate(ms):
            g = d[d.method == m].set_index('period').reindex(PERIODS)
            ax.bar(x + (j - (len(ms) - 1) / 2) * wd, g[k].values, wd, color=Fg.MCOL.get(m.lower(), '#444'), label=m)
        ax.set_xticks(x); ax.set_xticklabels(['Peak', 'Flat', 'Valley']); ax.set_ylabel(_flab(k))
    Fg._panel_labels(axs)
    fig.tight_layout()
    hs_, ls_ = axs[0].get_legend_handles_labels()
    Fg._bottom_legend(fig, hs_, ls_, y=0.0)
    Fg._save(fig, name, smoke, 'energy', df)


# ───────────────────────────── 汇总 ─────────────────────────────
def energy_report(smoke=False):
    """核查 + 全部表 + 全部图（只读结果，可反复运行）。"""
    out = {}
    for name, fn in [('reproduction', check_reproduction), ('noise', table_noise), ('scenarios', table_scenarios),
                     ('decomposition', table_decomposition), ('amplitude', table_amplitude), ('factors', table_factors),
                     ('pass_through', table_pass_through), ('period', table_period)]:
        try:
            out[name] = fn(smoke)
        except Exception as e:
            import traceback
            print(f'  ⚠️ {name} 失败：{e}'); traceback.print_exc()
    for fn in (fig_hourly_tariff, fig_amplitude, fig_decomposition, fig_period):
        try:
            fn(smoke=smoke)
        except Exception as e:
            import traceback
            print(f'  ⚠️ {fn.__name__} 失败：{e}'); traceback.print_exc()
    print(f'✅ 表格：{tables_dir(smoke)}')
    return out
