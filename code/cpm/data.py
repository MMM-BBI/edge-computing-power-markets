"""
合成数据生成（设计文档第二、七节）：散户画像、逐时算力需求、逐任务显存分档。

与原稿相比的改动
  · 删除亲和集 / 锚点；删除无效的 burst_* 参数；day_volatility 用论文取值 0.15；
  · 负荷转移 ξ 只作用于产消者（纯买方不付电费），时段改用第七节的峰平谷时段；
  · 每个任务按 PAI 实测经验分布抽单卡显存（与任务规模相关 0.18），以 ≤ 24 GB 为条件，存为分档；
  · 画像增加 tariff_class（单卡 = 居民电价，多卡 = 工商业电价）；
  · 需求与显存用两条独立随机流：改变显存参数 b 时需求矩阵完全不变（敏感性只改一个因素）；
  · 逐时生成改为向量化，全规模数据约 1–2 分钟生成完。
数据集缓存于 data/<data_key>/，同一组数据参数只生成一次；需求矩阵以 mmap 方式读取，多进程共享内存。
"""
import json
from pathlib import Path

import numpy as np

from . import specs

BASE_CURVE = np.array([0.30, 0.26, 0.24, 0.26, 0.34, 0.42, 0.55, 0.72,
                       0.88, 0.92, 0.86, 0.74, 0.66, 0.62, 0.66, 0.74,
                       0.84, 0.90, 0.82, 0.78, 0.72, 0.62, 0.50, 0.38])


def generate_profiles(cfg, rng):
    """散户画像：纯买方（无卡，名义档位只用于刻画任务规模）与产消者（1/2/4 张同型卡）。"""
    N = cfg.n_consumers
    n_pb = int(N * cfg.no_gpu_ratio)
    profiles = []
    for i in range(N):
        is_pb = i < n_pb
        g = int(rng.choice(specs.G, p=specs.GPU_PROBS))
        gamma = float(np.exp(rng.normal(np.log(cfg.task_gamma), cfg.task_sigma)))
        if is_pb:
            n_gpus, base = 0, gamma * specs.F[g]
        else:
            u = rng.random()
            n_gpus = 1 if u < 0.75 else (2 if u < 0.9375 else 4)
            base = gamma * n_gpus * specs.F[g]
        profiles.append(dict(agent_type='PureBuyer' if is_pb else 'Prosumer',
                             gpu=g if not is_pb else -1, nominal_gpu=g, n_gpus=n_gpus,
                             base_demand=base, tariff_class=specs.tariff_class(n_gpus) if not is_pb else -1))
    rng.shuffle(profiles)
    for i, p in enumerate(profiles):
        p['id'] = i
    return profiles


def generate_day(cfg, profiles_arr, rng):
    """生成一天 [24, N] 的需求（TFLOPS·h）。三状态：不活跃 / 常规 D=γ·C / 偶发大任务。"""
    T, N = cfg.steps_per_day, len(profiles_arr['base'])
    out = np.zeros((T, N), dtype=np.float32)
    is_pros = profiles_arr['is_prosumer']
    day_shock = float(np.exp(rng.normal(0.0, cfg.day_volatility)))
    hour_shocks = np.exp(rng.normal(0.0, cfg.hour_volatility, size=T))
    peak_shift = int(rng.integers(-2, 3))                      # 日内高峰 ±2 h 漂移
    c_lo, c_hi = BASE_CURVE.min(), BASE_CURVE.max()
    for step in range(T):
        hour = step % 24
        norm_c = (BASE_CURVE[(hour - peak_shift) % 24] - c_lo) / (c_hi - c_lo)
        per = specs.PERIOD[hour]
        shift = (1.0 - cfg.xi) if per == 0 else ((1.0 + cfg.xi) if per == 2 else 1.0)
        tier_mult = np.where(is_pros, shift, 1.0)              # ξ 只作用于产消者
        p_big = cfg.big_task_share_lo + (cfg.big_task_share_hi - cfg.big_task_share_lo) * norm_c
        p_reg = np.clip((cfg.p_active_lo + (cfg.p_active_hi - cfg.p_active_lo) * norm_c) * tier_mult, 0.02, 0.95)
        p_reg = np.minimum(p_reg, 1.0 - p_big)
        hsh = hour_shocks[step] * day_shock
        r = rng.random(N)
        big = r < p_big
        reg = (~big) & (r < p_big + p_reg)
        d = np.zeros(N)
        d[big] = rng.uniform(cfg.big_task_min, cfg.big_task_max, size=int(big.sum())) * hsh
        d[reg] = profiles_arr['base'][reg] * rng.uniform(0.85, 1.15, size=int(reg.sum())) * cfg.demand_scale * hsh
        out[step] = d
    return out


# PAI 2020 实测的单卡显存经验分布（每卡峰值显存，GB → 累积概率），来自本地标定 notebook 的输出：
# 分位数 p50 1.46、p75 4.26、p90 7.94、p95 9.53、p99 15.33；超过 8/10/12/16/24 GB 的比例 9.86%/4.56%/3.52%/0.42%/0.07%
PAI_CDF = np.array([[1.46, 0.50], [4.26, 0.75], [7.94, 0.90], [8.0, 0.9014], [9.53, 0.95], [10.0, 0.9544],
                    [12.0, 0.9648], [15.33, 0.99], [16.0, 0.9958], [24.0, 0.9993]])


def pai_quantile(u):
    """PAI 经验分布的分位函数：在 (Φ⁻¹(F), ln m) 坐标上分段线性插值，中位数以下按第一段斜率外推。"""
    from scipy.stats import norm
    zq, lm = norm.ppf(PAI_CDF[:, 1]), np.log(PAI_CDF[:, 0])
    z = norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))
    out = np.interp(z, zq, lm)
    lo = z < zq[0]
    out[lo] = lm[0] + (z[lo] - zq[0]) * (lm[1] - lm[0]) / (zq[1] - zq[0])
    hi = z > zq[-1]
    out[hi] = lm[-1] + (z[hi] - zq[-1]) * (lm[-1] - lm[-2]) / (zq[-1] - zq[-2])
    return np.exp(out)


def draw_vram(cfg, d, rng):
    """逐任务单卡显存需求（GB）；无需求处返回 0。
    边缘分布直接取 PAI 实测经验分布（不再用对数正态近似：后者的上尾比实测厚得多）；
    与任务规模的相关用高斯 copula 刻画，相关系数取 PAI 实测的 corr(ln 算力, ln 显存) = 0.18（vram_corr）；
    以 m ≤ vram_cap（24 GB，消费级卡最大显存）为条件抽取：超过 24 GB 的作业属于数据中心，不进入边端市场。"""
    from scipy.stats import norm, rankdata
    m = np.zeros_like(d, dtype=np.float32)
    act = d > 0
    n = int(act.sum())
    if n == 0:
        return m
    x = norm.ppf(rankdata(np.log(d[act])) / (n + 1))          # 当天任务规模的正态得分
    r = cfg.vram_corr
    z = r * x + np.sqrt(1 - r * r) * rng.standard_normal(n)
    u = norm.cdf(z)
    if cfg.vram_cap:
        u = u * _cdf_at(cfg.vram_cap)                      # 条件于 m ≤ 24 GB（截断，不是压到 24）
    m[act] = np.minimum(pai_quantile(u), cfg.vram_cap or np.inf)
    return m


def _cdf_at(v):
    from scipy.stats import norm
    zq, lm = norm.ppf(PAI_CDF[:, 1]), np.log(PAI_CDF[:, 0])
    return float(norm.cdf(np.interp(np.log(v), lm, zq)))


def profiles_to_arrays(profiles):
    return dict(
        is_prosumer=np.array([p['agent_type'] == 'Prosumer' for p in profiles]),
        base=np.array([p['base_demand'] for p in profiles], dtype=np.float64),
    )


def build_dataset(cfg, root: Path, verbose=True):
    """生成并保存数据集；已存在则直接返回目录。"""
    out = Path(root) / cfg.data_key()
    if (out / 'DONE').exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    rng_prof = np.random.default_rng(cfg.data_seed)
    rng_dem = np.random.default_rng(cfg.data_seed + 1)
    rng_vram = np.random.default_rng(cfg.data_seed + 2)
    profiles = generate_profiles(cfg, rng_prof)
    arr = profiles_to_arrays(profiles)
    n_train = cfg.train_days + cfg.val_days
    T, N = cfg.steps_per_day, cfg.n_consumers
    stats = dict(n_tasks=0, gt8=0, gt24=0, own_infeasible=0, pros_tasks=0)
    tier_hist = np.zeros(specs.N_TIERS, dtype=np.int64)
    own_g = np.array([p['gpu'] for p in profiles])
    is_p = arr['is_prosumer']
    if verbose:
        print(f'⏳ 生成数据集 {out.name}：{N} 散户，训练+验证 {n_train} 天，测试 {cfg.test_days} 天')
    for split, n_days in (('train', n_train), ('test', cfg.test_days)):
        dem = np.lib.format.open_memmap(out / f'{split}_demands.npy', mode='w+', dtype=np.float32, shape=(n_days, T, N))
        tier = np.lib.format.open_memmap(out / f'{split}_tiers.npy', mode='w+', dtype=np.uint8, shape=(n_days, T, N))
        for k in range(n_days):
            d = generate_day(cfg, arr, rng_dem)
            m = draw_vram(cfg, d, rng_vram)
            t = specs.vram_to_tier(m)
            dem[k] = d; tier[k] = t
            if split == 'train' and k < 200:                    # 显存统计（前 200 天足够）
                act = d > 0
                stats['n_tasks'] += int(act.sum())
                stats['gt8'] += int((m[act] > 8).sum()); stats['gt24'] += int((m[act] > 24).sum())
                tier_hist += np.bincount(t[act], minlength=specs.N_TIERS)
                pa = act & is_p[None, :]
                stats['pros_tasks'] += int(pa.sum())
                feas = specs.TIER_MASK[t, np.where(own_g >= 0, own_g, 0)[None, :].repeat(T, 0)]
                stats['own_infeasible'] += int((pa & ~feas).sum())
        dem.flush(); tier.flush(); del dem, tier
    meta = dict(data_key=cfg.data_key(), n_consumers=N, n_train_days=n_train, train_days=cfg.train_days,
                val_days=cfg.val_days, test_days=cfg.test_days,
                vram_stats=dict(share_gt8=stats['gt8'] / max(stats['n_tasks'], 1),
                                share_gt24=stats['gt24'] / max(stats['n_tasks'], 1),
                                prosumer_own_infeasible=stats['own_infeasible'] / max(stats['pros_tasks'], 1),
                                tier_hist=tier_hist.tolist()))
    with open(out / 'profiles.json', 'w') as f:
        json.dump(profiles, f)
    with open(out / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    (out / 'DONE').write_text('ok')
    if verbose:
        vs = meta['vram_stats']
        print(f"✅ 数据集完成：显存 >8 GB 占 {vs['share_gt8']:.1%}，>24 GB 占 {vs['share_gt24']:.1%}，"
              f"产消者任务超出自有卡显存 {vs['prosumer_own_infeasible']:.1%}")
    return out


def load_dataset(cfg, root: Path):
    path = build_dataset(cfg, root, verbose=True)
    with open(path / 'profiles.json') as f:
        profiles = json.load(f)
    with open(path / 'meta.json') as f:
        meta = json.load(f)
    return dict(path=str(path), profiles=profiles, meta=meta,
                train_demands=np.load(path / 'train_demands.npy', mmap_mode='r'),
                train_tiers=np.load(path / 'train_tiers.npy', mmap_mode='r'),
                test_demands=np.load(path / 'test_demands.npy', mmap_mode='r'),
                test_tiers=np.load(path / 'test_tiers.npy', mmap_mode='r'))
