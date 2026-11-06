"""
数据与标定自检（设计文档第二、六、七节），结果写入 results/tables/calibration_*.csv。
  · 电价：复现 2024-12 蒙西官方工商业分时价格；两条曲线的峰平谷值与峰谷比；
  · 成本前提：c^run 全时段高于 P^EB、且不超过 P^ES 的 37%（默认参数）；
  · 显存：任务显存 > 8 GB / > 24 GB 的比例，产消者任务超出自有卡显存的比例；
  · 保留价：φ_i 的分位数与两个实证锚点（中位数 0.20；P(φ > 0.404) = 16.2%），θ_i、λ_i 的范围，
    以及"无中介化市场"反事实下保留价 / ECC 零售价的分布（按卡型）。
"""
import numpy as np
import pandas as pd

from . import specs
from .experiments import get_cfg, DATA, results_root


def calibration_report(smoke=False):
    from .data import load_dataset
    out = results_root(smoke) / 'tables'; out.mkdir(parents=True, exist_ok=True)
    cfg = get_cfg('main', smoke, strict=False)
    err = specs.tariff_selfcheck()
    cur = specs.tariff_curves('tou')
    tariff = pd.DataFrame(dict(hour=range(24), period=[specs.PERIOD_NAMES[p] for p in specs.PERIOD],
                               residential=cur[0], commercial=cur[1]))
    tariff.to_csv(out / 'calibration_tariff.csv', index=False)
    print(f'电价自检：复现 2024-12 官方工商业分时价格，最大误差 {err:.1e} 元/kWh')
    for k, name in enumerate(specs.TARIFF_CLASSES):
        c = cur[k]
        print(f'  {name:12s} 峰 {c[6]:.4f}  平 {c[9]:.4f}  谷 {c[0]:.4f}  峰谷比 {c.max() / c.min():.2f}')
    prem = pd.DataFrame(specs.cost_premise_table(cfg.alpha, cfg.eta, cfg.hours_life))
    prem.to_csv(out / 'calibration_cost_premise.csv', index=False)
    print(f"成本前提：min c^run/P^EB = {prem.min_c_over_PEB.min():.2f}（应 > 1），"
          f"max c^run/P^ES = {prem.max_c_over_PES.max():.2f}")
    for a in (0.05, 0.2, 0.4):
        p = pd.DataFrame(specs.cost_premise_table(a, cfg.eta, cfg.hours_life))
        print(f'  α = {a}: P^EB 高于 c^run 的(卡型, 电价类别)组合 {(p.min_c_over_PEB < 1).sum()} / {len(p)}')
    ds = load_dataset(cfg, DATA)
    vs = ds['meta']['vram_stats']
    pd.DataFrame([vs]).to_csv(out / 'calibration_vram.csv', index=False)
    print(f"显存：>8 GB {vs['share_gt8']:.1%}，>24 GB {vs['share_gt24']:.1%}，"
          f"产消者任务超出自有卡显存 {vs['prosumer_own_infeasible']:.1%}")
    from .env import MarketEnv
    from scipy.stats import beta as _beta
    env = MarketEnv(cfg, ds, seed=0)
    P = env.is_pros
    phi = env.phi[P]
    rows = [dict(item='phi_quantiles', p10=np.quantile(phi, .1), p25=np.quantile(phi, .25), p50=np.median(phi),
                 p75=np.quantile(phi, .75), p90=np.quantile(phi, .9), mean=phi.mean()),
            dict(item='anchor_median_target_0.20', value=float(_beta.median(cfg.reserve_a, cfg.reserve_b) * cfg.reserve_scale)),
            dict(item='anchor_share_above_0.404_target_0.162', value=float((phi > 0.404).mean())),
            dict(item='theta_range', lo=float(env.theta_la[P].min()), hi=float(env.theta_la[P].max())),
            dict(item='lambda_range', lo=float(env.lam[P].min()), hi=float(env.lam[P].max()))]
    ext = []
    for h in range(24):                                  # 反事实保留价（P̂ = P^ES，无卖价经验）
        elec = env.curves[env.tclass, h]
        c = specs.run_cost(env.own0, elec, cfg.eta, cfg.hours_life)
        r = c + env.phi * np.maximum(0, specs.P_ES[env.own0] - c)
        ext.append(pd.DataFrame(dict(gpu=np.array(specs.GPU_NAMES)[env.own0[P]], hour=h,
                                     c_rel=(c / specs.P_ES[env.own0])[P], r_rel=(r / specs.P_ES[env.own0])[P])))
    ext = pd.concat(ext).groupby('gpu')[['c_rel', 'r_rel']].describe(percentiles=[.1, .5, .9])
    pd.DataFrame(rows).to_csv(out / 'calibration_reservation.csv', index=False)
    ext.to_csv(out / 'calibration_reservation_by_gpu.csv')
    print(f"保留价：φ 中位数 {np.median(phi):.3f}（锚点 0.20），φ > 0.404 的比例 {(phi > 0.404).mean():.1%}（锚点 16.2%），"
          f"θ ∈ [{env.theta_la[P].min():.2f}, {env.theta_la[P].max():.2f}]，λ ∈ [{env.lam[P].min():.2f}, {env.lam[P].max():.2f}]")
    print('  反事实保留价 / ECC 零售价（中位数，按卡型）：' +
          '，'.join(f"{g[3:]} {v:.2f}" for g, v in ext[('r_rel', '50%')].items()))
    return tariff, prem, vs
