"""
全部可调设置集中在本文件：默认参数（设计文档 9.2 参数总表）、实验标准、实验清单。

统一标准（控制变量）
  · 所有实验都用同一个标准设置 STANDARD：N = 3000、训练 2800 天、验证 200 天、测试 100 天、
    同一数据集、同一评估协议（新建环境 + 验证集最后 20 天预热 + 共同随机数）。
  · 正文：只用一个种子 MAIN_SEED。MARL 与全部对比方法在同一种子、同一数据、同一训练天数下比较；
    正文所有图都基于这一次运行。
  · 补充材料，每类实验只改变一个因素，其余与正文完全一致：
      调参（阶段一）  算法超参数的联合随机搜索（拉丁超立方）+ 前几名多种子复核，按验证集选择，
                      选中的值写入 results/tuning/best_params.json，之后所有实验（包括正文）自动使用；
      多种子          只改变种子（SEEDS_ROBUST）；
      消融 / 学习类基线 / 2×2 归因   只改变对应的设计，3 个种子（SEEDS_ABL，与多种子实验的种子 0–2 配对比较）；
      敏感性          只改变一个经济、标定或行为参数，单种子；
      结构实验        只改变市场规模 N 或中介数 |M|，单种子。
  · 经济、标定与行为参数（α、β、ρ̄、b、ξ、η、H、δ、预测噪声、纯买方比例、保留价强度 κ、预期速度 λ̄、损失厌恶强度）描述的是市场本身，
    只做敏感性检验，不按结果"优化"；只有算法超参数参与调参。
  SMOKE 预设只用于检查代码能否跑通，数值无意义。
"""
import hashlib
import json
from copy import deepcopy

# ─────────────────────────── 默认参数 ───────────────────────────
DEFAULTS = dict(
    # —— 市场规模与数据划分 ——
    n_consumers=3000, no_gpu_ratio=0.30, n_intermediaries=3,
    train_days=2800, val_days=200, test_days=100, steps_per_day=24,
    data_seed=42,
    # —— 需求过程（合成数据；与原稿一致，day_volatility 改为论文取值 0.15）——
    demand_scale=0.85, task_gamma=0.85, task_sigma=0.80,
    p_active_lo=0.10, p_active_hi=0.68,
    big_task_share_lo=0.001, big_task_share_hi=0.005, big_task_min=80.0, big_task_max=200.0,
    hour_volatility=0.22, day_volatility=0.15,
    xi=0.10,                    # 产消者负荷转移强度 ξ（只作用于产消者）
    # —— 显存需求（第二节 2.2）：边缘分布 = PAI 实测经验分布；与任务规模的相关 = 实测 0.18（高斯 copula）——
    vram_corr=0.18,             # corr(ln 任务算力, ln 单卡显存)，PAI 实测值
    vram_cap=24.0,              # 单卡显存需求上限（消费级卡最大 24 GB）；以 m ≤ 24 GB 为条件抽取
    # —— 价格与成本 ——
    alpha=0.10,                 # 回购价 / 零售价
    eta=0.93,                   # 负载功耗系数
    hours_life=43800.0,         # 折旧小时数 H
    tariff_mode='tou',          # 'tou' 分时；'flat' 平坦电价（电价渠道对照）
    tariff_pv_scale=1.0,        # 算电协同补充实验：峰谷幅度缩放 k（日均电价不变；1 = 原分时电价）
    delta=0.0,                  # 买方节省保证金 δ
    rho_bar=0.035, beta=0.30,   # 中介整合成本率 ρ_m = ρ̄ (θ_m/θ̄)^(-β)
    theta=None,                 # 中介规模系数；None 时按中介数自动取 linspace(1.3, 0.7, M)
    forecast_noise=0.20,        # 公共需求信号的乘性对数正态噪声 σ
    signal_hist_days=10,        # 中介收购信号的历史窗口（天）
    # —— 基准规则参数 ——
    mmr_mid_pos=0.5, cda_beta=0.15, cda_r_lo=0.05, cda_r_hi=0.95,
    # —— 学习算法（IPPO：散户按角色共享参数，各中介独立网络，critic 只看本地观测）——
    critic='local',             # 'local' = IPPO；'central' = 集中式 critic（MAPPO 学习类基线）
    share_intermediaries=False, # True：三家中介共享一张网络（合谋对照）
    gamma=0.5, gae_lambda=0.95,     # γ、学习率、熵、裁剪、PPO 轮数由阶段一联合调参确定（best_params.json 覆盖这里的默认值）
    clip=0.2, ppo_epochs=3,
    lr=1e-4, lr_floor=0.05, value_coef=0.5, entropy_coef=0.02, entropy_floor=0.001,
    max_grad_norm=0.5, hidden=256, n_layers=3, min_std=0.05,
    explore_floor=0.05, inter_explore_start=0.15, inter_explore_end=0.05,
    explore_lock_frac=0.30,     # 最后 30% 回合为稳定期：探索噪声、熵系数、学习率都固定在下限，用来检验收敛平台
    reward_scale=0.1,
    # —— 产消者的保留价（出售意愿）：r⁰_i,t = c_i,t + φ_i · max(0, P̂_i,t − c_i,t)；r_i,t = r⁰ + θ_i · max(0, ê_i,t − r⁰) ——
    # φ_i：可接受的最低剩余份额（"出价太低不如留在手里"），个体异质，Beta(a,b) 由最后通牒博弈实证锚定：
    #      低于 20% 的分配约一半被拒（中位数 0.20），平均出价 40.4% 时拒绝率 16.2%（Oosterbeek et al., 2004 元分析）
    # P̂_i,t：对本卡型该时段转售价的适应性预期（逐日指数平滑观察到的二级市场成交价，初值为 ECC 零售价），
    #      平滑速度 λ_i ~ U(0.5, 1.5)·ref_speed，个体异质；reserve_scale = 0 即旧模型（无保留价）
    # 损失厌恶：若市场价低于自己对卖价的预期 ê_i,t（逐日平滑观察到的一级市场成交价），保留价再上调 θ_i·(ê − r)，
    #      θ_i ~ U(0.25, 0.35)·la_scale，取自 Genesove & Mayer (2001)：面临名义损失的卖方把要价提高预期损失的 25–35%
    reserve_scale=1.0, reserve_a=1.29, reserve_b=4.24, ref_speed=0.3, la_scale=1.0,
    reward_mode='relative',     # 'relative' = U − U^base；'absolute' = U（奖励消融）
    inter_qty_map='unit',       # 中介收购强度 v_g 的映射：'unit' = (a+1)/2 ∈ [0,1]（与价格动作一致，初值 0.5，不会卡死在 0）；
                                #   'relu' = max(0,a)（旧设计，a<0 时该层收购量恒为 0、梯度消失；消融对照，初值同样设为 0.5）
    # —— 2×2 归因：中介的定价 / 数量由学习还是规则给出 ——
    inter_price_rule=None,      # None = 学习；'cda' = 自适应报价规则
    inter_qty_rule=None,        # None = 学习；'newsvendor' = 报童规则
    # —— 训练与评估流程 ——
    val_interval=50, val_days_per_eval=40, warmup_days=20,   # 预热 20 天：填满需求信号窗口（10 天）并让价格预期收敛（最慢 λ=0.15 时残差约 4%）
    device='auto', torch_threads=0,
)

PRESETS = {
    'MAIN': dict(val_interval=50, val_days_per_eval=10),      # 唯一的实验标准（STANDARD）
    'SMOKE': dict(n_consumers=40, train_days=6, val_days=12, test_days=3,
                  val_interval=2, val_days_per_eval=2, warmup_days=2),
}

# 后加入的参数：取默认值时不计入作业哈希（experiments.cfg_hash），已有作业的复用登记保持不变
ADDED_KEYS = ('tariff_pv_scale',)

# 只影响数据集的参数（决定数据缓存目录）
DATA_KEYS = ['n_consumers', 'no_gpu_ratio', 'train_days', 'val_days', 'test_days', 'steps_per_day',
             'data_seed', 'demand_scale', 'task_gamma', 'task_sigma', 'p_active_lo', 'p_active_hi',
             'big_task_share_lo', 'big_task_share_hi', 'big_task_min', 'big_task_max',
             'hour_volatility', 'day_volatility', 'xi',
             'vram_corr', 'vram_cap']


class Config:
    def __init__(self, **kw):
        d = deepcopy(DEFAULTS)
        for k, v in kw.items():
            if k not in d:
                raise KeyError(f'未知参数: {k}')
            d[k] = v
        self.__dict__.update(d)

    def to_dict(self):
        return deepcopy(self.__dict__)

    def replace(self, **kw):
        d = self.to_dict(); d.update(kw)
        return Config(**d)

    def data_key(self):
        sub = {k: self.__dict__[k] for k in DATA_KEYS}
        h = hashlib.md5(json.dumps(sub, sort_keys=True).encode()).hexdigest()[:10]
        return f"N{self.n_consumers}_T{self.train_days}_{h}"

    @property
    def thetas(self):
        if self.theta is not None:
            return list(self.theta)
        M = self.n_intermediaries
        if M == 1:
            return [1.0]
        return [1.3 - 0.6 * k / (M - 1) for k in range(M)]   # 1.3 … 0.7，均值恒为 1


def make_config(preset='MAIN', **over):
    d = deepcopy(PRESETS[preset]); d.update(over)
    return Config(**d)


# ─────────────────────────── 种子 ───────────────────────────
MAIN_SEED = 0                    # 正文：单种子
SEEDS_ROBUST = [0, 1, 2, 3, 4]   # 补充材料：多种子稳健性（只改变种子）
SEEDS_ABL = [0, 1, 2]            # 消融与 2×2 归因：3 个种子，与完整方法的同种子结果配对（种子间差异约 0.4–2 分，单种子分不开）

ALL_METHODS = ['marl', 'cda', 'sdr', 'mmr', 'mcafee']
BASELINES = ['cda', 'sdr', 'mmr', 'mcafee']


def _exp(group, label, methods, seeds=None, **over):
    return dict(group=group, preset='MAIN', label=label, seeds=list(seeds or [MAIN_SEED]), methods=list(methods),
                over=over)


def _fmt(v):
    return ('%g' % v) if isinstance(v, float) else str(v)


# ─────────────────────────── 阶段一：调参（只调算法超参数，按验证集选择） ───────────────────────────
# 联合随机搜索：在 5 个超参数的联合空间里用拉丁超立方抽样 N_TRIALS 组（Bergstra & Bengio, JMLR 2012：
# 同样的训练次数下，随机组合比网格和逐个调更容易覆盖到起作用的维度），外加一组原默认值作对照；
# 再对验证集排名前 TOP_K 的组合补跑 CONFIRM_SEEDS 两个种子，按 3 个种子的验证均值定最终参数，
# 避免选到"碰巧这个种子好"的组合。
TUNE_SPACE = {                     # (抽样方式, 下界, 上界)
    'gamma':           ('uniform', 0.0, 0.99),
    'lr':              ('log', 3e-5, 1e-3),
    'entropy_coef':    ('log', 1e-3, 5e-2),
    'clip':            ('uniform', 0.1, 0.3),
    'ppo_epochs':      ('int', 3, 10),
}
TUNE_KEYS = list(TUNE_SPACE)
N_TRIALS = 24                      # 随机组合数（另加 1 组默认值）
TUNE_SAMPLER_SEED = 2026
TOP_K = 3
CONFIRM_SEEDS = [1, 2]
TUNE_LABEL = {'gamma': 'Discount factor γ', 'lr': 'Learning rate', 'entropy_coef': 'Entropy coefficient',
              'clip': 'PPO clip range', 'ppo_epochs': 'PPO epochs per update'}


def tuning_trials():
    """返回 [(试验编号, 参数字典)]：0 号为原默认值，其余为拉丁超立方抽样（固定随机种子，可复现）。"""
    from scipy.stats import qmc
    trials = [(0, {k: DEFAULTS[k] for k in TUNE_KEYS})]
    U = qmc.LatinHypercube(d=len(TUNE_KEYS), seed=TUNE_SAMPLER_SEED).random(N_TRIALS)
    for t, u in enumerate(U, 1):
        p = {}
        for (k, (kind, lo, hi)), x in zip(TUNE_SPACE.items(), u):
            if kind == 'log':
                p[k] = float('%.3g' % (lo * (hi / lo) ** x))
            elif kind == 'int':
                p[k] = int(min(hi, lo + int(x * (hi - lo + 1))))
            else:
                p[k] = round(float(lo + x * (hi - lo)), 3)
        trials.append((t, p))
    return trials


# 敏感性只检验本文模型（MARL）：基准方法只作为主实验与网络实验中的定价对照，不对它们做敏感性分析
SENS_METHODS = ['marl']
# ─────────────────────────── 补充材料：敏感性（经济与标定参数，只检验不优化） ───────────────────────────
SENS = {
    'alpha':          ('α (buy-back / retail price)', [0.05, 0.15, 0.2, 0.3, 0.4]),
    'beta':           ('β (scale elasticity of integration cost)', [0.0, 0.15, 0.45, 0.6]),
    'rho_bar':        ('ρ̄ (base integration-cost rate)', [0.02, 0.05]),
    'vram_corr':      ('ρ (correlation of task size and VRAM)', [0.0, 0.1, 0.3, 0.5]),
    'xi':             ('ξ (prosumer load shifting)', [0.0, 0.05, 0.15, 0.2, 0.3]),
    'eta':            ('η (load power factor)', [0.8, 1.0]),
    'hours_life':     ('H (depreciation hours)', [26280.0, 61320.0]),
    'delta':          ('δ (buyer saving margin)', [0.02, 0.05, 0.1]),
    'forecast_noise': ('σ (public demand-signal noise)', [0.0, 0.1, 0.4]),
    'no_gpu_ratio':   ('Share of pure buyers', [0.2, 0.4]),
    'reserve_scale':  ('κ (strength of prosumer reservation, 0 = none)', [0.0, 0.5, 1.5]),
    'ref_speed':      ('λ̄ (speed of price-expectation updating)', [0.1, 0.6]),
    'la_scale':       ('Loss-aversion strength (scale of θ, 0 = none)', [0.0, 2.0]),
}
# ─────────────────────────── 算电协同补充实验（energy.py） ───────────────────────────
# 与正文一致的耦合：有分时电价就有负荷转移，平坦电价 ξ = 0（即已有的 sens_tariff_flat）；
# 峰谷差放大 k 倍时负荷转移按同一比例增大，ξ = 0.1·k；需求按原方法生成，其余设定与正文相同，单种子，只训练 MARL。
ENERGY_PV = [1.5, 2.0]
# ─────────────────────────── 补充材料：结构实验 ───────────────────────────
STRUCT = {
    'n_consumers':      ('Market size N', [300, 1000, 2000, 5000]),
    'n_intermediaries': ('Number of intermediaries |M|', [1, 2, 4, 5, 8]),
}


def _build():
    E = {}
    for t, p in tuning_trials():
        E[f'tune_t{t:02d}'] = dict(_exp('tune', f'trial {t:02d}', ['marl'], **p), tune_trial=t)
    E['main'] = _exp('main', 'Main experiment (single seed)', ALL_METHODS)
    E['robust'] = _exp('robust', 'Multi-seed robustness (seed is the only change)', ALL_METHODS, seeds=SEEDS_ROBUST)
    # 消融 / 学习类基线 / 2×2 归因（SEEDS_ABL 三个种子，只改对应设计，其余参数与正文实验完全一致）
    A = dict(seeds=SEEDS_ABL)
    E['abl_gamma099'] = _exp('ablation', 'γ = 0.99 (common default)', ['marl'], gamma=0.99, **A)
    E['abl_relu_qty'] = _exp('ablation', 'Quantity map max(0, a)', ['marl'], inter_qty_map='relu', **A)
    E['abl_reward_abs'] = _exp('ablation', 'Absolute reward (no U^base)', ['marl'], reward_mode='absolute', **A)
    E['abl_mappo'] = _exp('ablation', 'MAPPO (centralised critic)', ['marl'], critic='central', **A)
    E['abl_shared_inter'] = _exp('ablation', 'Shared intermediary network', ['marl'], share_intermediaries=True, **A)
    E['x_rulep_learnq'] = _exp('attribution', 'Rule price × learned quantity', ['marl'], inter_price_rule='cda', **A)
    E['x_learnp_ruleq'] = _exp('attribution', 'Learned price × rule quantity', ['marl'], inter_qty_rule='newsvendor', **A)
    E['x_rulep_ruleq'] = _exp('attribution', 'Rule price × rule quantity', ['marl'],
                              inter_price_rule='cda', inter_qty_rule='newsvendor', **A)
    for k, (lab, vals) in SENS.items():
        for v in vals:
            E[f'sens_{k}_{_fmt(v)}'] = _exp(f'sens_{k}', f'{k} = {_fmt(v)}', SENS_METHODS, **{k: v})
    E['sens_tariff_flat'] = _exp('sens_xi', 'Flat tariff (ξ = 0)', SENS_METHODS, tariff_mode='flat', xi=0.0)
    for k, (lab, vals) in STRUCT.items():
        for v in vals:
            E[f'struct_{k}_{v}'] = _exp(f'struct_{k}', f'{k} = {v}', ALL_METHODS, **{k: v})
    for k in ENERGY_PV:                       # 算电协同补充实验：峰谷差放大
        E[f'energy_pv_{_fmt(k)}'] = _exp('energy', f'Peak–valley × {_fmt(k)}, ξ = {_fmt(0.1 * k)}', ['marl'],
                                         tariff_pv_scale=k, xi=round(0.1 * k, 10))
    return E


EXPERIMENTS = _build()


def axis_points(group_key, default_value):
    """某个敏感性/结构参数的全部取值点 [(取值, 实验名)]，默认值对应正文实验 main。"""
    if group_key in SENS:
        vals = SENS[group_key][1]; pref = 'sens'
    else:
        vals = STRUCT[group_key][1]; pref = 'struct'
    pts = [(v, f'{pref}_{group_key}_{_fmt(v)}') for v in vals] + [(default_value, 'main')]
    pts = sorted(pts, key=lambda x: float(x[0]))
    if group_key == 'xi':
        pts.append(('flat', 'sens_tariff_flat'))
    return pts


# 消融与归因的"完整方法"取多种子实验 robust 中与 SEEDS_ABL 相同的种子（种子 0 即正文实验，均为复用，不重复训练）
ABL_REF = 'robust'
ATTRIBUTION = [(ABL_REF, 'learned', 'learned'), ('x_rulep_learnq', 'rule', 'learned'),
               ('x_learnp_ruleq', 'learned', 'rule'), ('x_rulep_ruleq', 'rule', 'rule')]
ABLATIONS = [(ABL_REF, 'Full method (IPPO)'), ('abl_gamma099', 'γ = 0.99'), ('abl_relu_qty', 'Quantity max(0, a)'),
             ('abl_reward_abs', 'Absolute reward'),
             ('abl_mappo', 'MAPPO (central critic)'), ('abl_shared_inter', 'Shared intermediary net')]

# 再学习协议（可利用度的最优响应重训与冲击后的恢复共用同一套设置）：在已训练策略上重新打开适度探索
# （探索倍数 0.2，即训练初期的 1/5；学习率取调参值的一半；熵系数取下限），让策略有能力对新环境作出调整
ADAPT = dict(exploration_scale=0.2, lr_scale=0.5)
# 可利用度：在正文策略上，冻结其余主体，只让一个主体做最优响应重训
EXPLOIT = dict(base='main', br_days=1000, br_check=100, val_days=20, retail_group_frac=0.05, deviators=['inter_0', 'inter_1', 'inter_2', 'prosumer', 'buyer'])
# 网络冲击：一律以正文实验（main，种子 0）的已训练策略为基准，不重训，只改变网络本身（节点、连接、规模），
# 全部定价方法同台评估。中介按规模系数 θ 编号：inter_0 最大（θ = 1.3），inter_2 最小（θ = 0.7）
SHOCK_FRACS = [0.1, 0.2, 0.3, 0.5]
NETSHOCK = dict(base='main', methods=ALL_METHODS, scenarios=(
    [('inter_remove_largest', dict(exclude_inters=[0])), ('inter_remove_smallest', dict(exclude_inters=[2])),
     ('inter_remove_two', dict(exclude_inters=[0, 1])),
     ('inter_entry_1', dict(entrants=1)), ('inter_entry_2', dict(entrants=2))]
    + [(f'prosumer_exit_{int(f * 100)}', dict(drop_role='Prosumer', drop_frac=f)) for f in SHOCK_FRACS]
    + [(f'buyer_exit_{int(f * 100)}', dict(drop_role='PureBuyer', drop_frac=f)) for f in SHOCK_FRACS]
    + [(f'layer_cut_{g[3:]}', dict(blocked_layers=[g])) for g in ['RTX3070', 'RTX3080', 'RTX3090', 'RTX4060',
                                                                  'RTX4070', 'RTX4080', 'RTX4090']]
    + [(f'size_{n}', dict(n_consumers=n)) for n in [300, 1000, 2000, 5000]]))
# 冲击后的恢复：MARL 在冲击后的网络上按再学习协议（ADAPT）继续学习，跟踪一级/二级成交量与福利；
# control 为同样再学习但不加冲击，用双重差分扣除再学习本身带来的变化
RECOVERY = dict(base='main', days=300, check=25, val_days=10,
                scenarios=['control', 'inter_remove_largest', 'inter_entry_1', 'prosumer_exit_30', 'buyer_exit_30',
                           'layer_cut_4090'])
# 开销与可扩展性
OVERHEAD = dict(sizes=[300, 1000, 3000, 10000], days=2)
