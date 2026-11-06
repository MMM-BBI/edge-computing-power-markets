"""
论文与补充材料的全部图。

版式：Times New Roman 英文字体；PNG 1000 dpi（另存 PDF 矢量版）；图中不加标题；
多子图组合在每个子图左上角标 (a)、(b)…；卡型配色固定为
78C8B9、9BA7BF、E8A296、A0D8E0、F4C8BD、FFE966、9E9EA0（依次对应 3070、3080、3090、4060、4070、4080、4090）。

输出：figures/<子目录>/<图名>.png / .pdf；该图画出的全部数值存在 figures/<子目录>/data/<图名>[__<部分>].csv，
图的来源（实验、N、训练天数、种子）写在 figures/<子目录>/data/<图名>__meta.csv，重画时直接读取。
子目录：00_data、main、robust、tuning、ablation、efficiency、failure、exploit、sens_<参数>、struct_<参数>、overhead。
缺少对应结果时打印提示并跳过，不抛错。
"""
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

from . import config as C
from . import specs
from .analysis import load_summaries, eval_dir, METHOD_LABEL, table_attribution, axis_def
from .experiments import results_root, figures_root, get_seeds, job_dir, get_cfg, DATA

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Nimbus Roman No9 L', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.bbox': 'tight', 'axes.unicode_minus': False})
DPI = 1000
FIGURES_VERSION = '2026-11-02b'                        # 运行 all_figures 时打印，用于确认 notebook 加载的是新版绘图代码
_FAILED = []
MCOL = {'marl': '#1f4e79', 'cda': '#c55a11', 'sdr': '#548235', 'mmr': '#7f6000', 'mcafee': '#7030a0'}
# 卡型配色：每个卡型一深一浅（浅色 = 填充/背景，深色 = 中位数线、分位线、轮廓、散点），依次对应 3070…4090
CARD_LIGHT = ['#e7abb8', '#f6cebf', '#f0ecd7', '#cee4de', '#b3c8e1', '#d7cae4', '#fff0af']
CARD_DARK = ['#c2294a', '#e8825c', '#d9cf99', '#82baab', '#3d73b2', '#8768a6', '#ffd000']
CARD_HEX = CARD_LIGHT
GCOL = dict(zip(specs.GPU_NAMES, CARD_LIGHT))
GDARK = dict(zip(specs.GPU_NAMES, CARD_DARK))
PCOL = {0: '#F2DADA', 1: '#FFF8E1', 2: '#AFC3D8'}      # 峰（红）、平（黄）、谷（蓝）
LEG_FS = 9                                             # 图下方统一图例的字号
SHOW_INLINE = True
CARD_LAB = [g.replace('RTX', '') for g in specs.GPU_NAMES]


# ───────────────────────────── 通用 ─────────────────────────────
def fig_dir(sub, smoke=False):
    d = figures_root(smoke) / sub
    (d / 'data').mkdir(parents=True, exist_ok=True)
    return d


def _panel_labels(axs):
    for i, ax in enumerate(np.ravel(axs)):
        ax.text(-0.02, 1.02, f'({chr(97 + i)})', transform=ax.transAxes, ha='right', va='bottom', fontsize=10)


def _bottom_legend(fig, handles, labels, ncol=None, y=-0.02):
    """全图统一图例，放在所有子图下方、居中，不与数据重叠（保存时 bbox='tight' 会把它包含进来）。"""
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, y), ncol=ncol or len(labels),
               frameon=False, fontsize=LEG_FS, handlelength=2.2, columnspacing=1.6, markerscale=1.6)


def _save(fig, name, smoke, sub, data=None, meta=None):
    """保存图（PNG 1000 dpi + PDF）与绘图数据（CSV），并在 notebook 中显示。"""
    d = fig_dir(sub, smoke)
    fig.savefig(d / f'{name}.png', dpi=DPI); fig.savefig(d / f'{name}.pdf')
    items = (data.items() if isinstance(data, dict) else [('', data)]) if data is not None else []
    for part, df in items:
        if df is None:
            continue
        fn = f'{name}__{part}.csv' if part else f'{name}.csv'
        df.to_csv(d / 'data' / fn, index=not isinstance(df.index, pd.RangeIndex))
    if meta:
        pd.DataFrame([meta]).to_csv(d / 'data' / f'{name}__meta.csv', index=False)
    plt.close(fig)
    print(f'  🖼 {sub}/{name}.png（绘图数据：{sub}/data/{name}*.csv）')
    if SHOW_INLINE:
        try:
            from IPython import get_ipython
            if get_ipython() is not None:
                from IPython.display import display, Image
                display(Image(filename=str(d / f'{name}.png'), width=900))
        except Exception:
            pass


def _meta(exp, smoke):
    cfg = get_cfg(exp, smoke, strict=False)
    done = [s for s in get_seeds(exp, smoke) if (job_dir(exp, s, smoke) / 'DONE').exists()]
    return dict(experiment=exp, N=cfg.n_consumers, train_days=cfg.train_days, test_days=cfg.test_days,
                seeds=' '.join(map(str, done)), gamma=cfg.gamma, lr=cfg.lr)


def _shade_periods(ax):
    for h in range(24):
        ax.axvspan(h - 0.5, h + 0.5, color=PCOL[specs.PERIOD[h]], lw=0, zorder=0)


def _skip(what):
    print(f'  （跳过 {what}：结果缺失）')


def _seed_files(exp, method, fname, smoke):
    out = []
    for s in get_seeds(exp, smoke):
        p = eval_dir(exp, s, method, smoke) / fname
        if p.exists():
            out.append((s, p))
    return out


# ───────────────────────────── 数据与标定 ─────────────────────────────
def fig_tariff_demand(smoke=False, exp='main'):
    """两条分时电价曲线与日内需求（测试集按角色平均）。"""
    from .data import load_dataset
    cfg = get_cfg(exp, smoke, strict=False); ds = load_dataset(cfg, DATA)
    cur = specs.tariff_curves('tou')
    is_p = np.array([p['agent_type'] == 'Prosumer' for p in ds['profiles']])
    dem = np.asarray(ds['test_demands'])
    data = pd.DataFrame(dict(hour=range(24), period=[specs.PERIOD_NAMES[p] for p in specs.PERIOD],
                             price_residential=cur[0], price_commercial=cur[1],
                             demand_prosumers_total=dem[:, :, is_p].sum(2).mean(0), demand_pure_buyers_total=dem[:, :, ~is_p].sum(2).mean(0),
                             demand_prosumers_per_capita=dem[:, :, is_p].mean(2).mean(0),
                             demand_pure_buyers_per_capita=dem[:, :, ~is_p].mean(2).mean(0)))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    _shade_periods(ax[0])
    ax[0].step(data.hour, data.price_residential, where='mid', label='Residential (single-card)', color='#1f4e79')
    ax[0].step(data.hour, data.price_commercial, where='mid', label='Commercial (multi-card)', color='#c55a11')
    ax[0].set(xlabel='Hour', ylabel='Electricity price (CNY/kWh)', xlim=(-0.5, 23.5)); ax[0].legend(frameon=False)
    _shade_periods(ax[1])
    ax[1].plot(data.hour, data.demand_prosumers_per_capita, label='Prosumers', color='#1f4e79')
    ax[1].plot(data.hour, data.demand_pure_buyers_per_capita, label='Pure buyers', color='#548235')
    ax[1].set(xlabel='Hour', ylabel='Demand per user (TFLOP·h)', xlim=(-0.5, 23.5)); ax[1].legend(frameon=False)
    _panel_labels(ax); fig.tight_layout()
    _save(fig, 'Fig_tariff_demand', smoke, '00_data', data, dict(N=cfg.n_consumers, test_days=cfg.test_days))


def fig_vram(smoke=False, exp='main'):
    from .data import load_dataset
    cfg = get_cfg(exp, smoke, strict=False); ds = load_dataset(cfg, DATA)
    th = np.array(ds['meta']['vram_stats']['tier_hist'], dtype=float)
    data = pd.DataFrame(dict(tier=['≤8', '8–10', '10–12', '12–16', '16–24', '>24'], share_of_tasks=th / th.sum(),
                             n_feasible_card_types=specs.TIER_MASK.sum(1)))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax[0].bar(data.tier, data.share_of_tasks, color='#9BA7BF'); ax[0].set(xlabel='Task VRAM (GB)', ylabel='Share of tasks')
    ax[0].set_yscale('log')
    ax[1].bar(data.tier, data.n_feasible_card_types, color='#78C8B9')
    ax[1].set(xlabel='Task VRAM (GB)', ylabel='Feasible card types |G|')
    _panel_labels(ax); fig.tight_layout()
    _save(fig, 'FigS_vram', smoke, '00_data', data)


# ───────────────────────────── 训练收敛 ─────────────────────────────
TRAIN_KEYS = ['welfare', 'inter_profit_avg', 'buyer_savings_pc', 'prosumer_savings_pc', 'unit_welfare']
UW_LABEL = 'Welfare per unit of\ncomputing power'          # 与正文“welfare per unit of computing power”一致
TRAIN_YLAB = ['Welfare (CNY/day)', 'Avg. intermediary profit (CNY/day)', 'Pure buyer savings (CNY/day)',
              'Prosumer savings (CNY/day)', UW_LABEL]
VAL_MAP = {'welfare': 'val_welfare', 'inter_profit_avg': 'val_inter_profit_avg', 'buyer_savings_pc': 'val_buyer_savings_pc',
           'prosumer_savings_pc': 'val_prosumer_savings_pc', 'unit_welfare': 'val_unit_welfare'}


def fig_training(exp='main', smoke=False, name=None, sub=None):
    """训练曲线：实线 = 训练（含探索噪声，滑动平均）；圆点 = 确定性验证；灰色阴影 = 稳定期（探索、熵、学习率已固定）。
    单种子时只有一条线；多种子（robust）时每个种子一条线。"""
    name = name or 'Fig_training'
    sub = sub or exp
    logs = []
    for s in get_seeds(exp, smoke):
        p = job_dir(exp, s, smoke) / 'train_log.csv'
        if p.exists():
            logs.append(pd.read_csv(p).assign(seed=s))
    if not logs:
        return _skip(f'{exp}/{name}')
    n = min(len(l) for l in logs)
    logs = [l.iloc[:n] for l in logs]
    w = max(5, n // 50)
    raw = pd.concat(logs)[['seed', 'episode'] + TRAIN_KEYS]
    sm = pd.DataFrame({'episode': logs[0]['episode'].values})
    for k in TRAIN_KEYS:
        for l in logs:
            sm[f'{k}_seed{l.seed.iloc[0]}'] = l[k].rolling(w, min_periods=1).mean().values
    vals = []
    for s in get_seeds(exp, smoke):
        p = job_dir(exp, s, smoke) / 'val_log.csv'
        if p.exists() and len(pd.read_csv(p)):
            vals.append(pd.read_csv(p).assign(seed=s))
    cfg = get_cfg(exp, smoke, strict=False)
    lock = int(n * (1 - cfg.explore_lock_frac))
    one = len(logs) == 1
    cols = ['#1f4e79'] if one else [plt.cm.tab10(i) for i in range(len(logs))]
    fig, axs = plt.subplots(1, 5, figsize=(13.0, 2.4))
    for ax, k, yl in zip(axs, TRAIN_KEYS, TRAIN_YLAB):
        for c, l in zip(cols, logs):
            ax.plot(sm.episode, sm[f'{k}_seed{l.seed.iloc[0]}'], color=c, lw=0.9 if one else 0.7)
        for c, v in zip(cols, vals):
            if VAL_MAP[k] in v:
                ax.plot(v['episode'], v[VAL_MAP[k]], 'o', ms=1.8, color='#c55a11' if one else c, alpha=0.9)
        ax.axvspan(lock, n - 1, color='#9E9EA0', alpha=0.15, lw=0)
        ax.set(xlabel='Episode (day)', ylabel=yl)
    # (c) 纯买家节省与 (d) 产消者节省使用相同的纵轴，便于直接比较两类散户的收益
    i_b, i_p = TRAIN_KEYS.index('buyer_savings_pc'), TRAIN_KEYS.index('prosumer_savings_pc')
    lo = min(0.0, axs[i_b].get_ylim()[0], axs[i_p].get_ylim()[0])
    hi = max(axs[i_b].get_ylim()[1], axs[i_p].get_ylim()[1])
    for i in (i_b, i_p):
        axs[i].set_ylim(lo, hi)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    if one:
        hs = [Line2D([], [], color='#1f4e79', lw=1.2), Line2D([], [], ls='none', marker='o', ms=4, color='#c55a11')]
        ls = ['Training (moving average)', f'Validation every {cfg.val_interval} episodes (deterministic policy)']
    else:
        hs = [Line2D([], [], color=c, lw=1.2) for c in cols] + \
             [Line2D([], [], ls='none', marker='o', ms=4, color='#808080')]
        ls = [f'Seed {l.seed.iloc[0]}' for l in logs] + [f'Validation every {cfg.val_interval} episodes (dots in seed color)']
    hs.append(Patch(color='#9E9EA0', alpha=0.3, lw=0)); ls.append('Stabilization phase')
    _panel_labels(axs); fig.tight_layout()
    _bottom_legend(fig, hs, ls, y=0.0)
    _save(fig, name, smoke, sub, {'raw': raw, 'smoothed': sm, 'validation': pd.concat(vals) if vals else None},
          dict(_meta(exp, smoke), smoothing_window=w, exploration_lock_episode=lock))


# ───────────────────────────── 方法对比 ─────────────────────────────
def fig_methods(exp='main', smoke=False, name='Fig_methods'):
    rows = []
    for m in C.ALL_METHODS:
        for s in load_summaries(exp, m, smoke):
            rows.append(dict(method=METHOD_LABEL[m], key=m, seed=s['seed'], unit_welfare=s['unit_welfare'],
                             internal_flow_share=s['internal_flow_share'], welfare_per_day=s['welfare_per_day']))
    if not rows:
        return _skip(f'{exp}/{name}')
    df = pd.DataFrame(rows)
    order = [m for m in C.ALL_METHODS if m in set(df.key)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.5))
    for k, m in enumerate(order):
        d = df[df.key == m]
        for a, col in zip(ax, ['unit_welfare', 'internal_flow_share']):
            a.bar(k, d[col].mean(), yerr=d[col].std(ddof=1) if len(d) > 1 else None, color=MCOL[m], capsize=3)
    for a, lab in zip(ax, [UW_LABEL, 'Internal flow share']):
        a.set_xticks(range(len(order))); a.set_xticklabels([METHOD_LABEL[m] for m in order]); a.set_ylabel(lab)
    _panel_labels(ax); fig.tight_layout()
    _save(fig, name, smoke, exp, df, _meta(exp, smoke))


# ───────────────────────────── 网络与市场行为 ─────────────────────────────
def _band(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.85):
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4,
             MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), fc=color, ec='none', alpha=alpha))


def fig_sankey(exp='main', smoke=False, method='marl', name='Fig_flow_network'):
    """三类主体 + ECC 的日均流量（按卡型着色）：产消者 → 中介 → 买方；ECC 兜底与清算。自供不画（数值见数据）。"""
    fs = _seed_files(exp, method, 'sankey.csv', smoke)
    if not fs:
        return _skip(f'{exp}/{name}')
    all_ = pd.concat([pd.read_csv(p).assign(seed=s) for s, p in fs])
    data = all_.groupby(['source', 'target', 'gpu'])['value'].mean().reset_index()
    df = data[data.gpu != 'all']
    tot = df.groupby(['source', 'target'])['value'].sum()
    scale = 1.0 / max(df['value'].sum(), 1e-9)
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.axis('off')
    gap = 0.04; col = {}
    h = tot.get(('Prosumer', 'Intermediary'), 0) * scale
    col[('L', 'Prosumer')] = (1.0 - h, 1.0)
    y = 1.0 - 0.03
    for n in ['Intermediary', 'ECC']:
        h = (tot.get(('Prosumer', 'Intermediary'), 0) if n == 'Intermediary'
             else tot.get(('ECC', 'PureBuyer'), 0) + tot.get(('ECC', 'Prosumer'), 0)) * scale
        col[('M', n)] = (y - h, y); y -= h + gap + 0.03
    y = 1.0
    for n in ['ECC', 'PureBuyer', 'Prosumer']:
        h = (tot.get(('Intermediary', 'ECC'), 0) if n == 'ECC'
             else tot.get(('Intermediary', n), 0) + tot.get(('ECC', n), 0)) * scale
        col[('R', n)] = (y - h, y); y -= h + gap
    X = {'L': 0.0, 'M': 0.45, 'R': 0.9}; wbox = 0.035
    labels = {('L', 'Prosumer'): 'Prosumers (sell)', ('M', 'Intermediary'): 'Intermediaries', ('M', 'ECC'): 'ECC (fallback)',
              ('R', 'PureBuyer'): 'Pure buyers', ('R', 'Prosumer'): 'Prosumers (buy)', ('R', 'ECC'): 'ECC (salvage)'}
    for (c, n), (y0, y1) in col.items():
        if y1 - y0 <= 0: continue
        ax.add_patch(Rectangle((X[c], y0), wbox, y1 - y0, fc='#4d4d4d'))
        if c == 'L':
            ax.text(X[c] - 0.01, (y0 + y1) / 2, labels[(c, n)], va='center', ha='right', fontsize=9)
        elif c == 'M':
            ax.text(X[c] + wbox / 2, y1 + 0.008, labels[(c, n)], va='bottom', ha='center', fontsize=9)
        else:
            ax.text(X[c] + wbox + 0.01, (y0 + y1) / 2, labels[(c, n)], va='center', ha='left', fontsize=9)
    out_top = {k: v[1] for k, v in col.items()}; in_top = {k: v[1] for k, v in col.items()}

    def flow(src, dst, s_key, d_key):
        for g in specs.GPU_NAMES:
            v = df[(df.source == src) & (df.target == dst) & (df.gpu == g)]['value'].sum() * scale
            if v <= 0: continue
            a0 = out_top[s_key]; b0 = in_top[d_key]
            _band(ax, X[s_key[0]] + wbox, X[d_key[0]], a0, a0 - v, b0, b0 - v, GCOL[g])
            out_top[s_key] -= v; in_top[d_key] -= v
    flow('Prosumer', 'Intermediary', ('L', 'Prosumer'), ('M', 'Intermediary'))
    flow('Intermediary', 'ECC', ('M', 'Intermediary'), ('R', 'ECC'))
    for n in ['PureBuyer', 'Prosumer']:
        flow('Intermediary', n, ('M', 'Intermediary'), ('R', n))
    for n in ['PureBuyer', 'Prosumer']:
        flow('ECC', n, ('M', 'ECC'), ('R', n))
    for g, lab in zip(specs.GPU_NAMES, CARD_LAB):
        ax.plot([], [], color=GCOL[g], lw=6, label=lab)
    ax.legend(ncol=7, frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.12))
    ax.set_xlim(-0.22, 1.15); ax.set_ylim(min(v[0] for v in col.values()) - 0.02, 1.06)
    selfv = data[(data.source == 'Prosumer') & (data.target == 'Self')]['value'].sum()
    _save(fig, name, smoke, exp, {'flows_mean_over_seeds': data, 'flows_by_seed': all_},
          dict(_meta(exp, smoke), method=method, self_supply_per_day_not_drawn=selfv))


def fig_hourly(exp='main', smoke=False, name='Fig_hourly_spread', flat_exp='sens_tariff_flat'):
    """(a) 逐时零售价差：分时电价（正文实验）vs 平坦电价对照；(b) 逐时零售与 ECC 兜底成交量。"""
    fs = _seed_files(exp, 'marl', 'hourly.csv', smoke)
    if not fs:
        return _skip(f'{exp}/{name}')
    h = pd.concat([pd.read_csv(p) for _, p in fs]).groupby(['hour', 'period'], as_index=False).mean(numeric_only=True)
    parts = {'hourly': h}          # 逐时成交量仍写入绘图数据，图中只画正文用到的价差
    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    _shade_periods(ax)
    ax.plot(h.hour, h.spread, '-o', ms=2.5, color='#1f4e79', label='TOU tariff')
    f2 = _seed_files(flat_exp, 'marl', 'hourly.csv', smoke) if flat_exp in C.EXPERIMENTS else []
    if f2:
        h2 = pd.concat([pd.read_csv(p) for _, p in f2]).groupby('hour', as_index=False).mean(numeric_only=True)
        ax.plot(h2.hour, h2.spread, '--s', ms=2.5, color='#c55a11', label='Flat tariff'); parts['hourly_flat'] = h2
    ax.set(xlabel='Hour', ylabel='Retail spread (CNY/(TFLOP·h))', xlim=(-0.5, 23.5))
    ax.set_xticks(range(0, 24, 4))
    fig.tight_layout()
    hs, ls = ax.get_legend_handles_labels()
    _bottom_legend(fig, hs, ls, y=0.0)
    _save(fig, name, smoke, exp, parts, _meta(exp, smoke))


def _wquantile(lo, hi, c, q):
    """由直方图（各区间 [lo, hi) 的成交笔数 c）求分位数，区间内按均匀分布线性插值。"""
    cum = np.concatenate([[0.0], np.cumsum(c)]) / c.sum()
    k = int(np.clip(np.searchsorted(cum, q, side='right') - 1, 0, len(c) - 1))
    frac = 0.0 if c[k] == 0 else (q - cum[k]) / (cum[k + 1] - cum[k])
    return float(lo[k] + np.clip(frac, 0, 1) * (hi[k] - lo[k]))


def _raincloud(ax, df, col, n_pts=250, seed=0):
    """云雨图：左侧半边小提琴（核密度，浅色填充、深色轮廓）+ 中间箱线图（四分位箱、中位数线、1.5 IQR 须）+ 右侧散点。
    原始数据只保存了 0.02 宽的价格直方图，散点按各区间的成交笔数等比例抽取（每卡型至多 n_pts 笔），
    在所属区间内均匀抖动；抽样种子固定，可复现。"""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 401)
    stats, pts = {}, []
    for k, g in enumerate(specs.GPU_NAMES):
        d = df[df.gpu == g].sort_values('bin_lo'); c = d[col].values.astype(float)
        if c.sum() == 0:
            continue
        lo, hi = d.bin_lo.values, d.bin_hi.values; mids = (lo + hi) / 2
        light, dark = GCOL[g], GDARK[g]
        # 半边小提琴：带权高斯核密度（带宽 = 一个价格区间宽度）
        bw = float(np.median(hi - lo))
        dens = (c[None, :] * np.exp(-0.5 * ((grid[:, None] - mids[None, :]) / bw) ** 2)).sum(1)
        q01, q99 = _wquantile(lo, hi, c, 0.005), _wquantile(lo, hi, c, 0.995)
        # 两端各延伸 3 个带宽，使核密度自然衰减到峰值的约 1%（原为 1 个带宽，尾部会被横向截断）；
        # 网格本身限于 [0, 1]，成交价不超过 ECC 零售价，贴近 1 的卡型（如 RTX 4090）在 1.0 处平顶是真实的上界。
        m = (grid >= q01 - 3 * bw) & (grid <= q99 + 3 * bw)
        w = dens[m] / dens[m].max() * 0.36
        x0 = k - 0.08
        ax.fill_betweenx(grid[m], x0 - w, x0, color=light, lw=0, zorder=2)
        ax.plot(x0 - w, grid[m], color=dark, lw=0.8, zorder=3)
        # 箱线图
        q1, med, q3 = (_wquantile(lo, hi, c, q) for q in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        wl = max(_wquantile(lo, hi, c, 0.0), q1 - 1.5 * iqr); wh = min(_wquantile(lo, hi, c, 1.0), q3 + 1.5 * iqr)
        bx = k + 0.0; bwid = 0.10
        ax.add_patch(Rectangle((bx - bwid / 2, q1), bwid, iqr, fc=light, ec=dark, lw=0.9, zorder=4))
        ax.plot([bx - bwid / 2, bx + bwid / 2], [med, med], color=dark, lw=1.6, zorder=5, solid_capstyle='butt')
        ax.plot([bx, bx], [wl, q1], color=dark, lw=0.8, zorder=4); ax.plot([bx, bx], [q3, wh], color=dark, lw=0.8, zorder=4)
        for yv in (wl, wh):
            ax.plot([bx - bwid / 4, bx + bwid / 4], [yv, yv], color=dark, lw=0.8, zorder=4)
        # 散点：按成交笔数等比例抽样，区间内均匀抖动
        n = int(min(n_pts, c.sum()))
        b = rng.choice(len(c), size=n, p=c / c.sum())
        yv = rng.uniform(lo[b], hi[b]); xv = k + 0.12 + rng.uniform(0, 0.22, n)
        ax.scatter(xv, yv, s=2.2, color=dark, alpha=0.45, lw=0, zorder=3, rasterized=True)
        pts.append(pd.DataFrame(dict(gpu=g, x_jitter=xv - k, ratio=yv)))
        stats[g] = dict(n_trades=int(c.sum()), q1=q1, median=med, q3=q3, whisker_lo=wl, whisker_hi=wh,
                        q005=q01, q995=q99, mean=float((mids * c).sum() / c.sum()))
    ax.set_xticks(range(specs.G)); ax.set_xticklabels(CARD_LAB); ax.set_xlim(-0.6, specs.G - 0.4)
    ys = [v for s in stats.values() for v in (s['q005'], s['q995'])]
    if ys:                                    # 纵轴覆盖全部卡型 0.5%–99.5% 分位并留出小提琴尾部（3 个带宽 = 0.06）
        ax.set_ylim(max(0.0, min(ys) - 0.07), min(1.02, max(ys) + 0.07))
    return stats, (pd.concat(pts) if pts else None)


def fig_price(exp='main', smoke=False, name='Fig_price_ratio'):
    """二级成交价 / ECC 零售价（按卡型的云雨图）。出价上限比的直方图仍写入绘图数据，但不再作图（正文未用）。"""
    fs = _seed_files(exp, 'marl', 'price_hist.csv', smoke)
    if not fs:
        return _skip(f'{exp}/{name}')
    raw = pd.concat([pd.read_csv(p) for _, p in fs])
    has_cap = 'count_over_cap' in raw.columns
    agg_cols = ['count', 'count_over_cap'] if has_cap else ['count']
    df = raw.groupby(['gpu', 'bin_lo', 'bin_hi'], as_index=False)[agg_cols].sum()
    caps = [json.load(open(p.parent / 'summary.json'))['share_p2_at_cap'] for _, p in fs]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    s1, pts = _raincloud(ax, df, 'count')
    ax.set(xlabel='Card type', ylabel='Retail price / ECC retail price')
    ax.grid(axis='y', color='#e6e6e6', lw=0.5, zorder=0); ax.set_axisbelow(True)
    rows = [dict(gpu=g, **{f'ratio_PES_{k}': v for k, v in s1[g].items()}) for g in s1]
    fig.tight_layout()
    _save(fig, name, smoke, exp, {'histogram': df, 'stats': pd.DataFrame(rows), 'sampled_points': pts},
          dict(_meta(exp, smoke), share_trades_at_99pct_of_cap=float(np.mean(caps)), points_per_card=250,
               whiskers='1.5 IQR', kde_bandwidth='one bin (0.02)'))


def fig_substitution(exp='main', smoke=False, name='Fig_substitution'):
    """层间替代网络。行 = 买方的最优外部卡型（ECC 上整单最便宜且满足显存），列 = 实际成交卡型；
    格内 = 行内占比（行和为 1）；纵轴括号 = 该行占全部成交量的比例；空白 = 无流量，'<.01' = 不足 1%。
    (a) 只计市场内（P2）成交；(b) 含 ECC 兜底。"""
    parts, mats = {}, {}
    for f in ('subst_market', 'subst_all'):
        fs = _seed_files(exp, 'marl', f'{f}.csv', smoke)
        if not fs:
            return _skip(f'{exp}/{name}')
        m = sum(pd.read_csv(p, index_col=0) for _, p in fs) / len(fs)
        mats[f] = m
        parts[f'{f}_volume_per_day'] = m
        parts[f'{f}_row_share'] = m.div(m.sum(axis=1).replace(0, np.nan), axis=0)
    fig, a = plt.subplots(figsize=(3.9, 3.6))          # 只画市场内成交（正文所用）；含 ECC 兜底的矩阵仍写入绘图数据
    f = 'subst_market'
    m = mats[f]; R = parts[f'{f}_row_share'].values; rowshare = m.sum(axis=1).values / max(m.values.sum(), 1e-9)
    a.imshow(np.nan_to_num(R), cmap='Blues', vmin=0, vmax=1)
    a.set_xticks(range(specs.G)); a.set_xticklabels(CARD_LAB, rotation=45); a.set_yticks(range(specs.G))
    a.set_yticklabels([f'{l} ({100 * rs:.0f}%)' for l, rs in zip(CARD_LAB, rowshare)])
    a.set(xlabel='Card type actually used', ylabel="Buyer's best external card type")
    for i in range(specs.G):
        for j in range(specs.G):
            v = R[i, j]
            if m.values[i, j] > 0 and np.isfinite(v):
                a.text(j, i, f'{v:.2f}' if v >= 0.01 else '<.01', ha='center', va='center', fontsize=6.5,
                       color='w' if v > 0.5 else 'k')
    fig.tight_layout()
    _save(fig, name, smoke, exp, parts, _meta(exp, smoke))


def fig_bidirectional(exp='main', smoke=False, name='Fig_bidirectional'):
    fs = _seed_files(exp, 'marl', 'hourly.csv', smoke)
    if not fs:
        return _skip(f'{exp}/{name}')
    h = pd.concat([pd.read_csv(p) for _, p in fs]).groupby('hour', as_index=False).mean(numeric_only=True)
    data = h[['hour', 'bidir_n', 'bidir_share', 'forced_buy_n']]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    _shade_periods(ax)
    ax.bar(h.hour - 0.2, h.bidir_n, 0.4, color='#9BA7BF', label='Bidirectional prosumers (sell and buy)')
    ax.bar(h.hour + 0.2, h.forced_buy_n, 0.4, color='#E8A296', label='Buying because own cards cannot run the task (VRAM or card count)')
    ax.set(xlabel='Hour', ylabel='Prosumers per hour', xlim=(-0.5, 23.5))
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
    _save(fig, name, smoke, exp, data, _meta(exp, smoke))


def fig_surplus(exp='main', smoke=False, name='Fig_surplus'):
    fs = _seed_files(exp, 'marl', 'agents.csv', smoke)
    if not fs:
        return _skip(f'{exp}/{name}')
    allp = pd.concat([pd.read_csv(p).assign(seed=s) for s, p in fs])
    a = allp.groupby(['agent', 'role', 'gpu', 'n_gpus', 'tariff'], dropna=False, as_index=False)[['surplus_per_day', 'demand_per_day']].mean()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    share = {}
    for k, r in enumerate(['PureBuyer', 'Prosumer']):
        v = a[a.role == r]['surplus_per_day']
        ax[k].hist(v, bins=40, color=['#78C8B9', '#9BA7BF'][k], ec='#4d4d4d', lw=0.2)
        ax[k].axvline(0, color='k', lw=0.8)
        ax[k].set(xlabel=f'{"Pure-buyer" if r == "PureBuyer" else "Prosumer"} surplus vs best external plan (CNY/day)',
                  ylabel='Agents')
        share[f'share_nonnegative_{r}'] = float((v >= -1e-9).mean())
    _panel_labels(ax); fig.tight_layout()
    _save(fig, name, smoke, exp, {'agents': a, 'agents_by_seed': allp}, dict(_meta(exp, smoke), **share))


# ───────────────────────────── 补充材料：调参、敏感性、结构、消融 ─────────────────────────────
def fig_tuning(smoke=False, name='FigS_tuning'):
    """调参：前几个面板为每个超参数取值与验证集单位福利的散点（随机搜索全部组合；方块 = 原默认值，星号 = 最终选定）；
    (g) 前几名组合的多种子复核（验证集均值 ± 标准差）。"""
    from .analysis import table_tuning
    t = table_tuning(smoke)
    if t is None:
        return _skip(name)
    d = results_root(smoke) / 'tuning'
    conf = pd.read_csv(d / 'confirm.csv') if (d / 'confirm.csv').exists() else None
    keys = C.TUNE_KEYS
    ncol = int(np.ceil((len(keys) + 1) / 2))
    fig, axs = plt.subplots(2, ncol, figsize=(2.9 * ncol, 4.8))
    axs = axs.ravel()
    sel = t[t['selected']] if 'selected' in t else t.iloc[:0]
    for a, k in zip(axs, keys):
        a.scatter(t[k], t.val_unit_welfare, s=14, color='#9BA7BF', ec='#4d4d4d', lw=0.3, label='Random trials')
        d0 = t[t.trial == 0]
        a.scatter(d0[k], d0.val_unit_welfare, marker='s', s=26, color='#E8A296', ec='k', lw=0.4, label='Previous default')
        if len(sel):
            a.scatter(sel[k], sel.val_unit_welfare, marker='*', s=110, color='k', label='Selected')
        if C.TUNE_SPACE[k][0] == 'log':
            a.set_xscale('log')
        a.set(xlabel=C.TUNE_LABEL[k], ylabel='Validation unit welfare')
    axs[0].legend(frameon=False, fontsize=7)
    a = axs[len(keys)]
    if conf is not None:
        conf = conf.sort_values('val_mean', ascending=False)
        x = np.arange(len(conf))
        a.bar(x, conf.val_mean, yerr=conf.val_sd.fillna(0), color=['#1f4e79'] + ['#9BA7BF'] * (len(conf) - 1), capsize=3)
        a.set_xticks(x); a.set_xticklabels([f'Trial {int(v):02d}' for v in conf.trial])
        a.set(ylabel='Validation unit welfare (multi-seed)')
        lo = (conf.val_mean - conf.val_sd.fillna(0)).min(); hi = (conf.val_mean + conf.val_sd.fillna(0)).max()
        a.set_ylim(lo - 0.15 * abs(hi - lo + 1e-9), hi + 0.15 * abs(hi - lo + 1e-9))
    else:
        a.axis('off')
    for a in axs[len(keys) + 1:]:
        a.axis('off')
    _panel_labels(axs[:len(keys) + (1 if conf is not None else 0)]); fig.tight_layout()
    _save(fig, name, smoke, 'tuning', {'search': t, 'confirm': conf})


SENS_METRICS = (('unit_welfare', UW_LABEL), ('inter_profit_avg', 'Avg. intermediary profit'),
                ('prosumer_savings_pc', 'Prosumer savings'), ('buyer_savings_pc', 'Pure buyer savings'),
                ('internal_flow_share', 'Internal flow share'))
_SENS_SCALE = {}


def _sens_scale(smoke):
    """全部敏感性图共用的纵轴范围（每个指标一套）与种子噪声。
    各参数的图若各自按数据自动缩放，±1% 的随机波动也会被放大成满幅折线；统一纵轴后，不同参数的影响可直接横向比较。
    噪声取多种子实验（robust，只改种子）中 MARL 各指标的种子间标准差。"""
    if smoke in _SENS_SCALE:
        return _SENS_SCALE[smoke]
    vals = {k: [] for k, _ in SENS_METRICS}
    for g in C.SENS:
        try:
            _, points = axis_def(g, smoke)
        except Exception:
            continue
        for val, exp in points:
            for s in load_summaries(exp, 'marl', smoke):
                for k, _ in SENS_METRICS:
                    if np.isfinite(s.get(k, np.nan)):
                        vals[k].append(s[k])
    sd = {}
    rob = load_summaries('robust', 'marl', smoke) if 'robust' in C.EXPERIMENTS else []
    for k, _ in SENS_METRICS:
        x = [r[k] for r in rob if np.isfinite(r.get(k, np.nan))]
        sd[k] = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    lim = {}
    for k, v in vals.items():
        if not v:
            continue
        lo, hi = float(np.min(v)), float(np.max(v))
        if np.isfinite(sd[k]):
            lo, hi = min(lo, np.median(v) - 2 * sd[k]), max(hi, np.median(v) + 2 * sd[k])
        pad = 0.08 * (hi - lo if hi > lo else abs(hi) + 1e-9)
        lim[k] = (lo - pad, hi + pad)
    _SENS_SCALE[smoke] = (lim, sd)
    return _SENS_SCALE[smoke]


# 敏感性 / 结构实验横轴名称：与论文正文参数表及支撑材料表 S9 的名称、符号一致
AXIS_LABEL = {
    'alpha':            r'Buy-back ratio $\alpha$',
    'beta':             r'Scale elasticity $\beta$',
    'rho_bar':          r'Baseline integration cost rate $\bar{\rho}$',
    'vram_corr':        r'VRAM–size correlation $\rho_{\nu}$',
    'xi':               r'Load shifting intensity $\xi$',
    'eta':              r'Load power ratio $\eta$',
    'hours_life':       r'Depreciation hours $H$',
    'delta':            r'Buyer savings margin $\Omega$',
    'forecast_noise':   r'Demand signal noise $\sigma_f$',
    'no_gpu_ratio':     'Share of pure buyers',
    'reserve_scale':    r'Fair share scaling (multiplier of $f_i$)',
    'ref_speed':        r'Expectation updating speed (mean of $\upsilon_i$)',
    'la_scale':         r'Loss aversion scaling (multiplier of $\chi_i$)',
    'n_consumers':      r'Number of retail participants $|N|$',
    'n_intermediaries': r'Number of intermediaries $|M|$',
}


def fig_axis(group_key, smoke=False, metrics=None):
    """敏感性：只画 MARL（本文模型），全部敏感性图的同一指标共用纵轴，并以灰带标出默认值 ±2 个种子间标准差；
    结构实验（N、|M|）：各定价方法同图，图例统一放在下方。横轴为参数取值（含正文默认值）。"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    is_sens = group_key in C.SENS
    methods = C.SENS_METHODS if is_sens else C.ALL_METHODS
    if metrics is None:
        metrics = (SENS_METRICS if is_sens else
                   (('unit_welfare', UW_LABEL), ('inter_profit_avg', 'Avg. intermediary profit'),
                    ('internal_flow_share', 'Internal flow share'), ('hhi_retail', 'HHI of retail flows')))
    title, points = axis_def(group_key, smoke)
    title = AXIS_LABEL.get(group_key, title)            # 横轴名称与符号统一为论文写法（支撑材料表 S9、图 S8–S20）
    rows = []
    for x, (val, exp) in enumerate(points):
        for m in methods:
            for s in load_summaries(exp, m, smoke):
                r = dict(parameter=group_key, value=str(val), x=x, experiment=exp, method=METHOD_LABEL[m], key=m,
                         seed=s['seed'], **{k: s.get(k, np.nan) for k, _ in metrics})
                if 'hhi_retail' in r and not (s.get('p2_vol', 0) > 0):
                    r['hhi_retail'] = np.nan                          # 没有零售成交时集中度无定义
                rows.append(r)
    if not rows:
        return _skip(group_key)
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 2.4))
    dflt = [x for x, (v, e) in enumerate(points) if e == 'main']
    if is_sens:
        lim, sd = _sens_scale(smoke)
    for m in methods:
        d = df[df.key == m].groupby('x', as_index=False).mean(numeric_only=True)
        if not len(d): continue
        for a, (k, lab) in zip(axs, metrics):
            a.plot(d.x, d[k], marker='o', ms=3, color=MCOL[m], label=METHOD_LABEL[m], zorder=3)
            if is_sens and dflt:
                y0 = float(d.set_index('x').reindex(dflt)[k].iloc[0])
                a.plot(dflt, [y0], 'o', ms=6, mfc='none', mec='k', mew=0.8, zorder=4)
                if np.isfinite(sd.get(k, np.nan)):
                    a.axhspan(y0 - 2 * sd[k], y0 + 2 * sd[k], color='#9E9EA0', alpha=0.18, lw=0, zorder=1)
    for a, (k, lab) in zip(axs, metrics):
        a.set_xticks(range(len(points))); a.set_xticklabels([('%g' % v) if not isinstance(v, str) else v for v, _ in points])
        a.set(xlabel=title, ylabel=lab)
        if is_sens and k in lim:
            a.set_ylim(*lim[k])
    pref = 'sens' if is_sens else 'struct'
    _panel_labels(axs); fig.tight_layout()
    if is_sens:
        hs = [Line2D([], [], color=MCOL['marl'], marker='o', ms=4), Line2D([], [], ls='none', marker='o', ms=7, mfc='none', mec='k'),
              Patch(color='#9E9EA0', alpha=0.35, lw=0)]
        ls = ['MARL', 'Default value (main experiment)', 'Default ± 2 seed SD']
        _bottom_legend(fig, hs, ls, y=-0.02)
    else:
        hs, ls = axs[0].get_legend_handles_labels()
        _bottom_legend(fig, hs, ls, y=-0.02)
    _save(fig, f'Fig_{pref}_{group_key}', smoke, f'{pref}_{group_key}', df,
          dict(parameter=group_key, default_experiment='main', methods=','.join(methods),
               **({f'ylim_{k}': str(lim.get(k)) for k, _ in metrics} if is_sens else {}),
               **({f'seed_sd_{k}': sd.get(k) for k, _ in metrics} if is_sens else {})))


def fig_ablation(smoke=False, name='Fig_ablation'):
    rows = []
    from .analysis import _abl_summaries
    for exp, lab in C.ABLATIONS:
        for s in _abl_summaries(exp, smoke):
            rows.append(dict(variant=lab, experiment=exp, seed=s['seed'], unit_welfare=s['unit_welfare'],
                             inter_profit_avg=s['inter_profit_avg']))
    if not rows:
        return _skip(name)
    df = pd.DataFrame(rows)
    order = [lab for _, lab in C.ABLATIONS if lab in set(df.variant)]
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 2.6), gridspec_kw=dict(width_ratios=[2, 1]))
    for k, lab in enumerate(order):
        v = df[df.variant == lab].unit_welfare
        ax[0].bar(k, v.mean(), color='#1f4e79' if k == 0 else '#9BA7BF',
                  yerr=v.std(ddof=1) if len(v) > 1 else None, capsize=2, error_kw=dict(lw=0.6))
        ax[0].plot(np.full(len(v), k), v.values, 'o', ms=2, color='k', alpha=0.6)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=25, ha='right'); ax[0].set_ylabel(UW_LABEL)
    t = table_attribution(smoke)
    t4 = t[t.price.isin(['learned', 'rule'])] if len(t) else t
    if len(t4) == 4:
        g = t4.set_index(['price', 'quantity'])['unit_welfare_mean']
        M = np.array([[g['learned', 'learned'], g['learned', 'rule']], [g['rule', 'learned'], g['rule', 'rule']]], float)
        ax[1].imshow(M, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax[1].text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', color='w' if M[i, j] > M.mean() else 'k')
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(['Learned quantity', 'Rule quantity'])
        ax[1].set_yticks([0, 1]); ax[1].set_yticklabels(['Learned price', 'Rule price'])
        ax[1].spines[:].set_visible(False)
        _panel_labels(ax)
    else:
        ax[1].axis('off')
    fig.tight_layout()
    _save(fig, name, smoke, 'ablation', {'ablation': df, 'attribution': t})


def fig_efficiency(exp='main', smoke=False, name='Fig_efficiency'):
    from .analysis import table_efficiency
    t = table_efficiency(exp, smoke)
    if t is None:
        return _skip(name)
    tt = t[t.method != 'Upper bound (LP)']
    inv = {v: k for k, v in METHOD_LABEL.items()}
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.bar(tt.method, tt.efficiency, color=[MCOL[inv[m]] for m in tt.method])
    ax.set(ylabel='Realised / upper-bound welfare', ylim=(min(0, tt.efficiency.min() * 1.1), 1))
    _save(fig, name, smoke, 'efficiency', t, _meta(exp, smoke))


def fig_exploit(smoke=False, name='Fig_exploitability'):
    p = results_root(smoke) / C.EXPLOIT['base'] / 'exploitability.csv'
    if not p.exists():
        return _skip(name)
    df = pd.read_csv(p)
    g = df.groupby('deviator')['rel_gain'].mean().reindex(C.EXPLOIT['deviators']).dropna()
    labels = {'inter_0': 'Intermediary 1', 'inter_1': 'Intermediary 2', 'inter_2': 'Intermediary 3',
              'prosumer': 'Prosumer group (5%)', 'buyer': 'Pure-buyer group (5%)'}
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    ax.bar([labels.get(i, i) for i in g.index], 100 * g.values, color='#9BA7BF', ec='#4d4d4d', lw=0.4)
    ax.axhline(0, color='k', lw=0.8)
    ax.set(ylabel='Unilateral best-response gain (%)'); plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    _save(fig, name, smoke, 'exploit', df, _meta(C.EXPLOIT['base'], smoke))


def fig_overhead(smoke=False, name='Fig_overhead'):
    p = results_root(smoke) / 'overhead.csv'
    if not p.exists():
        return _skip(name)
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    ax.loglog(d.N, d.clearing_ms_per_step, 'o-', color='#1f4e79', label='Market clearing per step')
    ax.loglog(d.N, d.policy_ms_per_step, 's-', color='#c55a11', label='Policy inference per step')
    ax.loglog(d.N, 1e3 * d.update_s_per_day / 24, '^-', color='#548235', label='PPO update per step (amortised)')
    ax.set(xlabel='Number of retail agents N', ylabel='Time (ms)'); ax.legend(frameon=False)
    _save(fig, name, smoke, 'overhead', d)


def fig_surplus_dynamics(exp='main', smoke=False, name='Fig_surplus_dynamics'):
    """福利分配随训练的变化（确定性验证曲线）：
    (a) 三方剩余的堆叠面积（买方节省 + 产消者节省 + 中介利润 = 福利）：总量先升后平，构成可以变化；
    (b) 一级市场成交价 / ECC 零售价，与产消者卖算力的人均剩余（解释产消者收益的走势）。"""
    vals = []
    for s in get_seeds(exp, smoke):
        p = job_dir(exp, s, smoke) / 'val_log.csv'
        if p.exists():
            vals.append(pd.read_csv(p).assign(seed=s))
    if not vals:
        return _skip(f'{exp}/{name}')
    cfg = get_cfg(exp, smoke, strict=False)
    n_pb = int(cfg.n_consumers * cfg.no_gpu_ratio); n_pr = cfg.n_consumers - n_pb; M = cfg.n_intermediaries
    v = pd.concat(vals).groupby('episode', as_index=False).mean(numeric_only=True)
    d = pd.DataFrame({'episode': v.episode, 'buyers': v.val_buyer_savings_pc * n_pb,
                      'prosumers': v.val_prosumer_savings_pc * n_pr, 'intermediaries': v.val_inter_profit_avg * M,
                      'welfare': v.val_welfare})
    has_p1 = 'val_p1_price_rel' in v
    if has_p1:
        d['p1_price_rel'] = v.val_p1_price_rel; d['pros_sell_surplus_pc'] = v.val_pros_sell_surplus_pc
        if 'val_reserve_rel' in v: d['reserve_rel'] = v.val_reserve_rel
    lock = int(cfg.train_days * (1 - cfg.explore_lock_frac))
    # 只画三方剩余的堆叠（一级成交价与保留价的走势仍写入绘图数据，正文与支撑材料只引用数值）
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.axvspan(lock, cfg.train_days, color='#9E9EA0', alpha=0.15, lw=0, label='Stabilization phase', zorder=0)
    ax.stackplot(d.episode, d.buyers, d.prosumers, d.intermediaries, colors=['#78C8B9', '#9BA7BF', '#E8A296'],
                 labels=['Pure buyers', 'Prosumers', 'Intermediaries'], alpha=0.9, zorder=1)
    ax.plot(d.episode, d.welfare, color='#1f1f1f', lw=0.9, label='Social welfare', zorder=2)
    ax.set(xlabel='Episode (day)', ylabel='Validation surplus (CNY/day)')
    ax.set_xlim(0, max(cfg.train_days, float(d.episode.max()))); ax.set_ylim(0, float(d.welfare.max()) * 1.38)   # 原点重合；上方留出图例空间
    ax.margins(x=0, y=0)
    hs, ls = ax.get_legend_handles_labels()
    order = [ls.index(x) for x in ('Pure buyers', 'Prosumers', 'Intermediaries', 'Social welfare', 'Stabilization phase')]
    ax.legend([hs[i] for i in order], [ls[i] for i in order], ncol=2, loc='upper left', frameon=False,
              fontsize=LEG_FS, columnspacing=1.5, handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout()
    _save(fig, name, smoke, exp, d, dict(_meta(exp, smoke), exploration_lock_episode=lock))


def experiment_figures(exp, smoke=False):
    """正文全套市场行为图（训练曲线、方法对比、流量、价差、价格、替代、双向节点、剩余）。"""
    for fn in (fig_training, fig_surplus_dynamics, fig_methods, fig_sankey, fig_hourly, fig_price, fig_substitution,
               fig_bidirectional, fig_surplus, fig_network_structure, fig_network_formation):
        try:
            fn(exp=exp, smoke=smoke)
        except Exception as e:
            _fail(f'{exp}/{fn.__name__}', e)


def robust_figures(smoke=False):
    fig_network_formation('robust', smoke, name='FigS_robust_network_formation')
    fig_training('robust', smoke, name='FigS_robust_training')
    fig_methods('robust', smoke, name='FigS_robust_methods')


def _fail(what, e):
    """某张图失败时打印完整报错；该图的旧 PNG 不会被覆盖，所以必须让失败显眼。"""
    import traceback
    print(f'  ⚠ {what} 失败（旧图未更新）：{e!r}')
    traceback.print_exc()
    _FAILED.append(what)


def all_figures(smoke=False):
    global SHOW_INLINE
    _FAILED.clear()
    print(f'绘图代码版本 {FIGURES_VERSION}（{__file__}）')
    keep, SHOW_INLINE = SHOW_INLINE, False
    try:
        for fn in (fig_tariff_demand, fig_vram):
            try: fn(smoke=smoke)
            except Exception as e: _fail(fn.__name__, e)
        experiment_figures('main', smoke)
        for fn in (robust_figures, fig_tuning, fig_ablation, fig_efficiency, fig_exploit, fig_overhead, fig_network_structure,
                   fig_netshock_agents, fig_netshock_inter, fig_layer_criticality, fig_size_transfer, fig_recovery):
            try: fn(smoke=smoke)
            except Exception as e: _fail(fn.__name__, e)
        for k in list(C.SENS) + list(C.STRUCT):
            try: fig_axis(k, smoke)
            except Exception as e: _fail(k, e)
    finally:
        SHOW_INLINE = keep
    if _FAILED:
        print(f'❌ 以下 {len(_FAILED)} 张图失败，对应的旧图仍在原位置：{_FAILED}')
    print(f'✅ 图与绘图数据已写入 {figures_root(smoke)}（每个子目录下的 data/ 为绘图数据）')


# ───────────────────────────── 网络结构与网络冲击 ─────────────────────────────
def _net_eval(path):
    p = Path(path) / 'summary.json'
    return json.load(open(p)) if p.exists() else None


def fig_network_structure(exp='main', smoke=False, name='Fig_network_structure'):
    """正文实验中各定价方法形成的交易网络：(a) 散户每天交易的中介数（度分布）；(b) 逐日交易关系的持续性
    （相邻两天边集合的 Jaccard 系数）；(c) 卖方成交量的洛伦兹曲线（集中度）。"""
    seed = get_seeds(exp, smoke)[0]
    deg, pers = [], []
    for m in C.ALL_METHODS:
        d = job_dir(exp, seed, smoke) / f'eval_{m}'
        if not (d / 'network_degrees.csv').exists():
            continue
        g = pd.read_csv(d / 'network_degrees.csv'); g['method'] = METHOD_LABEL[m]; g['key'] = m; deg.append(g)
        nd = pd.read_csv(d / 'network_daily.csv')
        pers.append(dict(method=METHOD_LABEL[m], key=m, P1=nd['persist_p1'].mean(), P2=nd['persist_p2'].mean()))
    if not deg:
        return _skip(name)
    deg = pd.concat(deg); pers = pd.DataFrame(pers)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.6))
    sub = deg[deg.kind == 'seller']
    ks = sorted(sub.degree.unique()); w = 0.8 / max(len(pers), 1)
    for j, (m, gm) in enumerate(sub.groupby('key', sort=False)):
        frac = gm.set_index('degree')['count'] / gm['count'].sum()
        ax[0].bar(np.array(ks) + (j - len(pers) / 2 + 0.5) * w, frac.reindex(ks).fillna(0).values, w,
                  color=MCOL[m], label=METHOD_LABEL[m])
    ax[0].set(xlabel='Intermediaries a seller trades with per day', ylabel='Share of active sellers')
    ax[0].set_xticks(ks); ax[0].legend(frameon=False, fontsize=7)
    x = np.arange(len(pers))
    ax[1].bar(x - 0.2, pers.P1, 0.4, color='#9BA7BF', label='Primary market (P1)')
    ax[1].bar(x + 0.2, pers.P2, 0.4, color='#E8A296', label='Retail market (P2)')
    ax[1].set_xticks(x); ax[1].set_xticklabels(pers.method)
    ax[1].set(ylabel='Day-to-day edge persistence'); ax[1].legend(frameon=False, fontsize=7)
    _panel_labels(ax); fig.tight_layout()
    _save(fig, name, smoke, exp, {'degrees': deg, 'persistence': pers}, _meta(exp, smoke))


def _shock_table(smoke):
    from .analysis import table_netshock
    t = table_netshock(smoke)
    return t if t is not None and len(t) else None


def fig_netshock_agents(smoke=False, name='Fig_netshock_agents'):
    """产消者退出（供给冲击）与纯买方退出（需求冲击）：横轴为退出比例，纵轴为相对完整网络的变化（%）。
    第一行产消者退出、第二行纯买方退出；列：一级成交量、二级成交量、福利。"""
    t = _shock_table(smoke)
    if t is None:
        return _skip(name)
    fam = {'prosumer': 'Prosumer exit (%)', 'buyer': 'Pure-buyer exit (%)'}
    cols = [('d_p1_vol', 'P1 volume change (%)'), ('d_p2_vol', 'P2 volume change (%)'), ('d_welfare_per_day', 'Welfare change (%)')]
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 4.8))
    data = []
    for r, (f, xl) in enumerate(fam.items()):
        for m in C.ALL_METHODS:
            d = t[(t.key == m) & ((t.family == f) | (t.scenario == 'intact'))].copy()
            if not len(d): continue
            d['x'] = d.scenario.map(lambda n: 0 if n == 'intact' else int(n.split('_')[-1]))
            d = d.sort_values('x'); data.append(d.assign(row=f))
            for a, (k, lab) in zip(axs[r], cols):
                a.plot(d.x, d[k], 'o-', ms=3, color=MCOL[m], label=METHOD_LABEL[m])
        for a, (k, lab) in zip(axs[r], cols):
            a.axhline(0, color='k', lw=0.6); a.set(xlabel=xl, ylabel=lab)
    axs[0, 0].legend(frameon=False, fontsize=7)
    _panel_labels(axs.ravel()); fig.tight_layout()
    _save(fig, name, smoke, 'network', pd.concat(data) if data else None)


def fig_netshock_inter(smoke=False, name='Fig_netshock_inter'):
    """中介退出与进入：各情景相对完整网络的变化（%）。"""
    t = _shock_table(smoke)
    if t is None:
        return _skip(name)
    order = [('inter_remove_two', 'Remove 2 largest'), ('inter_remove_largest', 'Remove largest'),
             ('inter_remove_smallest', 'Remove smallest'), ('inter_entry_1', '+1 entrant'), ('inter_entry_2', '+2 entrants')]
    cols = [('d_welfare_per_day', 'Welfare change (%)'), ('d_p2_vol', 'P2 volume change (%)'),
            ('d_ecc_fallback_vol', 'ECC fallback change (%)'), ('a_inter_profit_avg', 'Δ avg. intermediary profit (per day)')]
    fig, axs = plt.subplots(1, 4, figsize=(12.0, 2.8))
    ms = [m for m in C.ALL_METHODS if m in set(t.key)]
    w = 0.8 / max(len(ms), 1); x = np.arange(len(order)); data = []
    for j, m in enumerate(ms):
        d = t[t.key == m].set_index('scenario').reindex([o for o, _ in order]); data.append(d.reset_index())
        for a, (k, lab) in zip(axs, cols):
            a.bar(x + (j - len(ms) / 2 + 0.5) * w, d[k].values, w, color=MCOL[m], label=METHOD_LABEL[m])
    for a, (k, lab) in zip(axs, cols):
        a.axhline(0, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels([l for _, l in order], rotation=30, ha='right')
        a.set(ylabel=lab)
    axs[0].legend(frameon=False, fontsize=7)
    _panel_labels(axs); fig.tight_layout()
    _save(fig, name, smoke, 'network', pd.concat(data) if data else None)


def fig_layer_criticality(smoke=False, name='Fig_layer_criticality'):
    """卡型层的关键性：逐一切断每个卡型层后福利的下降（%）与流向 ECC 的增加（%）。"""
    t = _shock_table(smoke)
    if t is None:
        return _skip(name)
    layers = [g.replace('RTX', '') for g in specs.GPU_NAMES]
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 2.7))
    ms = [m for m in C.ALL_METHODS if m in set(t.key)]
    w = 0.8 / max(len(ms), 1); x = np.arange(len(layers)); data = []
    for j, m in enumerate(ms):
        d = t[t.key == m].set_index('scenario').reindex([f'layer_cut_{l}' for l in layers]); data.append(d.reset_index())
        axs[0].bar(x + (j - len(ms) / 2 + 0.5) * w, -d['d_welfare_per_day'].values, w, color=MCOL[m], label=METHOD_LABEL[m])
        axs[1].bar(x + (j - len(ms) / 2 + 0.5) * w, d['d_ecc_fallback_vol'].values, w, color=MCOL[m])
    for a, lab in zip(axs, ['Welfare drop (%)', 'ECC fallback increase (%)']):
        a.axhline(0, color='k', lw=0.6); a.set_xticks(x); a.set_xticklabels(layers); a.set(xlabel='Card-type layer removed', ylabel=lab)
    axs[0].legend(frameon=False, fontsize=7)
    _panel_labels(axs); fig.tight_layout()
    _save(fig, name, smoke, 'network', pd.concat(data) if data else None)


def fig_size_transfer(smoke=False, name='Fig_size_transfer'):
    """市场规模：正文策略直接用于更小/更大的市场（不重训，短期）与在该规模下重新训练（长期均衡）对比。"""
    t = _shock_table(smoke)
    from .analysis import table_axis
    try:
        ax_t = table_axis('n_consumers', smoke)
    except Exception:
        ax_t = None
    if t is None or ax_t is None or not len(ax_t):
        return _skip(name)
    rows = []
    for n_, d in t[t.family == 'size'].groupby('scenario'):
        N = int(n_.split('_')[1])
        for _, r in d.iterrows():
            rows.append(dict(N=N, method=r['method'], key=r['key'], mode='transfer', unit_welfare=r['unit_welfare'],
                             internal_flow_share=r['internal_flow_share']))
    for _, r in t[t.scenario == 'intact'].iterrows():          # 正文策略在 N = 3000 本身（迁移曲线的基准点）
        rows.append(dict(N=int(get_cfg(C.NETSHOCK['base'], smoke, strict=False).n_consumers), method=r['method'], key=r['key'],
                         mode='transfer', unit_welfare=r['unit_welfare'],
                         internal_flow_share=r['internal_flow_share']))
    for _, r in ax_t.iterrows():
        key = {v: k for k, v in METHOD_LABEL.items()}[r['method']]
        rows.append(dict(N=int(float(r['value'])), method=r['method'], key=key, mode='retrained',
                         unit_welfare=r['unit_welfare'], internal_flow_share=r['internal_flow_share']))
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.6))
    for a, k, lab in zip(axs, ['unit_welfare', 'internal_flow_share'], [UW_LABEL, 'Internal flow share']):
        for m in ['marl', 'cda']:
            for mode, ls in (('retrained', '-'), ('transfer', '--')):
                d = df[(df.key == m) & (df['mode'] == mode)].sort_values('N')
                if m != 'marl' and mode == 'transfer':
                    continue
                lbl = f"{METHOD_LABEL[m]} ({'policy from N = 3000' if mode == 'transfer' else 'trained at each N'})" if m == 'marl' else METHOD_LABEL[m]
                a.plot(d.N, d[k], ls, marker='o', ms=3, color=MCOL[m], label=lbl)
        a.set_xscale('log'); a.set(xlabel='Number of retail agents N', ylabel=lab)
        ns = sorted(df.N.unique()); a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns])
        a.xaxis.set_minor_locator(plt.NullLocator())
    axs[0].legend(frameon=False, fontsize=7)
    _panel_labels(axs); fig.tight_layout()
    _save(fig, name, smoke, 'network', df)


def fig_recovery(smoke=False, name='Fig_recovery'):
    """冲击后的恢复（MARL 在冲击后的网络上继续学习）：相对未冲击网络的一级/二级成交量与福利。"""
    from .analysis import table_recovery
    table_recovery(smoke)
    p = results_root(smoke) / 'tables' / 'recovery_curves.csv'
    if not p.exists():
        return _skip(name)
    c = pd.read_csv(p)
    if not len(c):
        return _skip(name)
    lab = {'control': 'No shock', 'inter_remove_largest': 'Largest intermediary exits', 'inter_entry_1': 'One entrant',
           'prosumer_exit_30': '30% prosumers exit', 'buyer_exit_30': '30% pure buyers exit',
           'layer_cut_4090': 'RTX 4090 layer cut'}
    cols = ['#9E9EA0', '#1f4e79', '#548235', '#c55a11', '#7030a0', '#b8860b']
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.7))
    for (sc, d), col in zip(c.groupby('scenario', sort=False), cols):
        for a, k in zip(axs, ['rel_p1_vol', 'rel_p2_vol', 'rel_welfare']):
            a.plot(d.day, d[k], 'o-', ms=2.5, lw=0.9, color=col, label=lab.get(sc, sc))
    for a, yl in zip(axs, ['P1 volume (relative to intact)', 'P2 volume (relative to intact)', 'Welfare (relative to intact)']):
        a.axhline(1, color='k', lw=0.5, ls=':'); a.set(xlabel='Days after shock (continued learning)', ylabel=yl)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=len(l), loc='lower center', bbox_to_anchor=(0.5, -0.06))
    _panel_labels(axs); fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, name, smoke, 'network', c)


def fig_network_formation(exp='main', smoke=False, name='Fig_network_formation'):
    """交易网络在训练中的形成（确定性验证）：一级/二级每天的边数（散户–中介交易关系）、活跃卖方比例、
    零售流量在中介间的集中度 HHI。灰色阴影为稳定期。多种子实验每个种子一条线。"""
    vals = []
    for s in get_seeds(exp, smoke):
        p = job_dir(exp, s, smoke) / 'val_log.csv'
        if p.exists():
            v = pd.read_csv(p)
            if 'val_net_edges_p1' in v:
                vals.append(v.assign(seed=s))
    if not vals:
        return _skip(f'{exp}/{name}')
    cfg = get_cfg(exp, smoke, strict=False)
    lock = int(cfg.train_days * (1 - cfg.explore_lock_frac))
    keys = [('val_net_edges_p1', 'P1 edges per day'), ('val_net_edges_p2', 'P2 edges per day'),
            ('val_net_active_sellers', 'Active seller share'), ('val_net_hhi_retail', 'HHI of retail flows')]
    one = len(vals) == 1
    cols = ['#1f4e79'] if one else [plt.cm.tab10(i) for i in range(len(vals))]
    fig, axs = plt.subplots(1, 4, figsize=(11.0, 2.5))
    for a, (k, lab) in zip(axs, keys):
        for c, v in zip(cols, vals):
            a.plot(v.episode, v[k], 'o-', ms=1.8, lw=0.8, color=c, label=None if one else f'Seed {v.seed.iloc[0]}')
        a.axvspan(lock, cfg.train_days - 1, color='#9E9EA0', alpha=0.15, lw=0)
        a.set(xlabel='Episode (day)', ylabel=lab)
    if not one:
        axs[-1].legend(frameon=False, fontsize=7)
    _panel_labels(axs); fig.tight_layout()
    _save(fig, name, smoke, exp, pd.concat(vals), dict(_meta(exp, smoke), exploration_lock_episode=lock))
