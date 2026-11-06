"""
中介化算力交易网络的市场环境（设计文档第二至八节的最终版规则）。

每个时段三阶段：
  P1 一级市场：产消者 → 中介，按卡型分别做统一价双向拍卖（可分割）；
  P2 二级市场：中介 → 买方，整单、单一卡型、单一来源，按买方偏好分轮出清；
  P3 外部兜底：未成交买方（可回退自供者先自供）按 G_{i,t} 内整单最便宜的卡型向 ECC 购买；
     中介余货按 P^EB 卖给 ECC；产消者闲置仅当 P^EB ≥ 其当前保留价（≥ c^run）时才卖给 ECC。

主要规则（对应设计文档）
  · 可行卡型 G_{i,t} 由显存分档决定（2.3）；自供条件 g_i ∈ G_{i,t} 且卡数足够（2.4，F4）；
  · 买方出价上限 min{P^ES_g, (1−δ)B/Q_g}，保留硬件的产消者再加 Θ/Q_g（4.1、4.2）；
  · 中介一级出价 ∈ [P^EB, P^ES − o_m]，零售保留价 ≥ p^P1 + o_m（4.3，F2）；
  · 每个报价只有一个定位系数（4.4）；中介收购量 = θ̃_m · Ŝ_t · v_g / max(1, Σv)，v_g = (a+1)/2 ∈ [0,1]（2.4，F1）；
  · 产消者保留价 r_i,t = r⁰ + θ_i·max(0, ê_i,t − r⁰)，r⁰ = c_i,t + φ_i·max(0, P̂_i,t − c_i,t)：出售（给中介或 ECC）
    的底价为 max(r_i,t, P^EB)；P̂ 与 ê 是对转售价、卖价的适应性预期（逐日按 λ_i 平滑），跨天保留；
    四种规则基准与 MARL 受同一约束（SDR/MMR 的供给只计愿意卖的闲置，McAfee 如实报保留价）；
  · 奖励：散户 U − U^base（第八节，F5），中介为利润；W = Σ(U − U^base) + ΣΠ + 运营方盈余（仅 McAfee 交易缩减时非零）；
    U 与 W 都是货币口径，
    保留价只约束出售决策、不进入效用；U^base 是"无中介化市场"的反事实（P̂ = P^ES、无卖价经验），
    与定价方法和动作都无关，所以不卖不会被惩罚，各方法的福利基准相同；
  · 公共信号噪声由 (data_seed, 日, 小时) 确定性生成（共同随机数，F6）。
"""
from collections import deque

import numpy as np

from . import specs
from .mechanism import clear_divisible, clear_indivisible

ROLES = ('Prosumer', 'PureBuyer', 'Intermediary')
RULES = ('cda', 'sdr', 'mmr', 'mcafee')


def action_dims(n_inter=None):
    return {'Prosumer': 3 + 2 * specs.G, 'PureBuyer': 2 * specs.G, 'Intermediary': 3 * specs.G}


class MarketEnv:
    def __init__(self, cfg, dataset, seed=0, exclude_inters=(), blocked_layers=(), entrants=0, drop_agents=None):
        """网络冲击（只在评估/恢复实验中使用，训练时全部为默认值）：
        exclude_inters  移除的中介编号（其收购能力随之消失，其余中介的份额不重新归一化）；
        entrants        新进入的中介数（规模系数 θ = 1，即中等规模；其收购能力是新增的）；
        blocked_layers  切断的卡型层；
        drop_agents     退出市场的散户编号（不再有需求，也不再提供算力）。"""
        self.cfg = cfg
        prof = dataset['profiles']
        self.N = len(prof)
        self.is_pros = np.array([p['agent_type'] == 'Prosumer' for p in prof])
        self.idx_pros = np.where(self.is_pros)[0]
        self.idx_pb = np.where(~self.is_pros)[0]
        self.own = np.array([p['gpu'] for p in prof])                    # −1 = 无卡
        self.own0 = np.where(self.own >= 0, self.own, 0)
        self.n_gpus = np.array([p['n_gpus'] for p in prof], dtype=np.float64)
        self.F_own = np.where(self.is_pros, specs.F[self.own0], 1.0)
        self.cap = self.n_gpus * np.where(self.is_pros, specs.F[self.own0], 0.0)
        self.dropped = np.zeros(self.N, bool)
        if drop_agents is not None and len(drop_agents):
            self.dropped[np.asarray(drop_agents, dtype=int)] = True
            self.cap = np.where(self.dropped, 0.0, self.cap)
        self.tclass = np.array([max(p['tariff_class'], 0) for p in prof])
        self.curves = specs.tariff_curves(cfg.tariff_mode, cfg.tariff_pv_scale)   # [2, 24]
        self.peb = specs.p_eb(cfg.alpha)
        self.blocked = np.zeros(specs.G, dtype=bool)
        for g in blocked_layers:
            self.blocked[specs.GIDX[g] if isinstance(g, str) else int(g)] = True
        # —— 中介：规模系数 θ、份额 θ̃、整合成本率 ρ_m（第三节）——
        thetas_all = np.array(cfg.thetas, dtype=np.float64)
        theta_bar = thetas_all.mean()
        theta_sum0 = thetas_all.sum()                                   # 原市场的总规模：各中介收购能力 ∝ 自身 θ
        n0 = len(thetas_all)
        thetas_all = np.concatenate([thetas_all, np.ones(int(entrants))])
        keep = [k for k in range(len(thetas_all)) if k not in set(exclude_inters)]
        self.inter_names = [f'inter_{k}' for k in keep]
        self.theta = thetas_all[keep]
        self.M = len(keep)
        # 份额相对原市场总规模计算：完整市场时与原定义完全相同；移除中介 = 其收购能力消失，新进入 = 新增收购能力
        self.share = self.theta / theta_sum0
        self.n_base_inters = n0
        self.rho = cfg.rho_bar * (self.theta / theta_bar) ** (-cfg.beta)
        self.o = self.rho[:, None] * specs.P_ES[None, :]                  # [M, G] 元/TFLOPS·h
        self.o_min = self.o.min(axis=0) if self.M else np.full(specs.G, np.inf)
        # 静态供给结构（报童规则与观测用）
        cap_by_g = np.bincount(self.own0[self.is_pros], weights=self.cap[self.is_pros], minlength=specs.G)
        self.supply_share = cap_by_g / max(cap_by_g.sum(), 1e-9)
        self.total_cap = max(self.cap.sum(), 1e-9)
        # —— 跨日持续的状态 ——
        H = cfg.signal_hist_days
        self.addr_hist = [deque([np.zeros(specs.G)] * H, maxlen=H) for _ in range(24)]
        self.fill = np.zeros((self.M, specs.G)); self.leftover = np.zeros((self.M, specs.G))
        self.nv_p1 = np.zeros(specs.G); self.nv_p2 = np.zeros(specs.G)          # 报童规则：跨天保留的成交价平滑
        self.nv_st = np.ones((self.M, specs.G))                                  # 报童规则：售出率平滑
        cda_rng = np.random.default_rng(int(seed) + 2024)          # CDA 激进度初值按种子固定
        self.cda_r_retail = cda_rng.uniform(0.2, 0.8, size=self.N)
        self.cda_r_inter = cda_rng.uniform(0.2, 0.8, size=self.M)
        # —— 产消者保留价的个体参数（独立随机流，不改变数据集）——
        rrng = np.random.default_rng([int(cfg.data_seed), 4242])
        self.phi = np.where(self.is_pros, np.minimum(cfg.reserve_scale * rrng.beta(cfg.reserve_a, cfg.reserve_b, self.N), 0.95), 0.0)
        self.lam = np.where(self.is_pros, cfg.ref_speed * rrng.uniform(0.5, 1.5, self.N), 0.0).clip(0.0, 1.0)
        self.ref_p = np.repeat(specs.P_ES[self.own0][:, None], 24, axis=1)      # 对转售价的预期 P̂（按 24 个时段）
        self.theta_la = np.where(self.is_pros, cfg.la_scale * rrng.uniform(0.25, 0.35, self.N), 0.0)
        self.ref_p1 = np.zeros((self.N, 24))                                      # 对自己卖价的预期 ê（尚无观察时为 0，不起作用）
        self._reset_day_history()

    # ───────────────────────────── 生命周期 ─────────────────────────────
    def copy_state_from(self, other):
        """复制跨日持续的状态（产消者价格预期、需求信号历史、中介的私有信号），用于验证环境与训练环境同步。
        规则基准自己的状态（报童价格平滑、CDA 激进度）不复制：验证只评估学习策略。"""
        self.ref_p = other.ref_p.copy(); self.ref_p1 = other.ref_p1.copy()
        self.addr_hist = [deque(list(q), maxlen=q.maxlen) for q in other.addr_hist]
        self.fill = other.fill.copy(); self.leftover = other.leftover.copy()

    def _reset_day_history(self):
        self.last_p1 = np.zeros(specs.G); self.last_p2 = np.zeros(specs.G); self.last_vol = np.zeros(specs.G)

    def reset(self, demands_day, tiers_day, day_id):
        """开始新的一天。demands_day/tiers_day: [24, N]；day_id 为全局日序号（共同随机数用）。"""
        self.dem = np.asarray(demands_day, dtype=np.float64)
        if self.dropped.any():
            self.dem = np.where(self.dropped[None, :], 0.0, self.dem)
        self.tiers = np.asarray(tiers_day)
        self.day_id = int(day_id)
        self.hour = 0
        self._reset_day_history()
        self._prepare_hour()
        return self.observe()

    def _noise(self, hour):
        if self.cfg.forecast_noise <= 0:
            return 1.0
        rng = np.random.default_rng([self.cfg.data_seed, 777, self.day_id, hour])
        return float(np.exp(rng.normal(0.0, self.cfg.forecast_noise)))

    def _prepare_hour(self):
        """计算本时段与动作无关的量：需求、可行集、整单支出、B、运行成本、U^base、公共信号。"""
        h = self.hour; cfg = self.cfg
        d = self.dem[h]
        self.d = d
        self.active = d > 1e-9
        self.Gm = specs.TIER_MASK[self.tiers[h]] & ~self.blocked[None, :]
        self.Q = specs.whole_card_qty(d)                                  # [N, G]
        ecc = self.Q * specs.P_ES[None, :]
        ecc_feas = np.where(specs.TIER_MASK[self.tiers[h]], ecc, np.inf)  # ECC 不受层切断影响
        self.best_g = ecc_feas.argmin(axis=1)
        self.B = np.where(self.active, ecc_feas.min(axis=1), 0.0)
        self.elec = np.where(self.is_pros, self.curves[self.tclass, h % 24], 0.0)
        self.c = np.where(self.is_pros, specs.run_cost(self.own0, self.elec, cfg.eta, cfg.hours_life), 0.0)
        # 保留价：低于它宁可不卖（卡留在手里）；只约束出售决策，不计入成本与福利
        r0 = self.c + self.phi * np.maximum(0.0, self.ref_p[:, h % 24] - self.c)          # 公平份额：转售剩余的 φ_i
        r0 = r0 + self.theta_la * np.maximum(0.0, self.ref_p1[:, h % 24] - r0)            # 损失厌恶：低于预期卖价时惜售
        self.reserve = np.where(self.is_pros, r0, 0.0)
        self.floor_sell = np.minimum(np.maximum(self.reserve, self.peb[self.own0]), specs.P_ES[self.own0])
        self.cards_own = np.where(self.is_pros, np.ceil(d / self.F_own - 1e-9), 0.0)
        own_feas = specs.TIER_MASK[self.tiers[h], self.own0]
        self.can_self = self.is_pros & self.active & own_feas & (self.cards_own <= self.n_gpus)
        # U^base（第八节）：纯买方 −B；产消者 max{外部方案, 自跑}
        # 卖给 ECC 回购（P^EB）同样受保留价约束：只有 P^EB ≥ 当前保留价时才会回购卖出（实际清算与 Θ 用它）。
        self.salv_ok = self.is_pros & (self.peb[self.own0] >= self.reserve - 1e-12)
        # U^base 是"没有中介化市场"的反事实：此时没有二级市场成交价可参照，预期 P̂ 停在 ECC 零售价、也没有卖价经验（ê 无定义），
        # 保留价为 r^ext = c + φ_i·(P^ES − c)。它与定价方法、与训练进程都无关，所以各方法的福利基准相同、可直接比较；
        # 且 U^base 不依赖动作，"不卖"不会在奖励中被惩罚（只是一个与动作无关的基准平移）。
        r_ext = self.c + self.phi * np.maximum(0.0, specs.P_ES[self.own0] - self.c)
        salv_base = self.is_pros & (self.peb[self.own0] >= r_ext - 1e-12)
        margin = np.where(salv_base, self.peb[self.own0] - self.c, 0.0)
        ext = -self.B + self.cap * margin
        selfopt = np.where(self.can_self,
                           -self.cards_own * self.F_own * self.c + (self.cap - self.cards_own * self.F_own) * margin,
                           -np.inf)
        self.U_base = np.where(self.is_pros, np.maximum(ext, selfopt), -self.B)
        self.margin = margin
        # 公共信号（第二节 2.4：总量 + 卡型占比）
        noise = self._noise(h)
        hist = np.array(self.addr_hist[h % 24])                           # [H, G]
        tot = hist.sum(axis=1)
        self.S_samples = tot * noise
        self.S_hat = float(tot.mean()) * noise
        s = hist.sum(axis=0)
        self.addr_share = s / s.sum() if s.sum() > 0 else self.supply_share.copy()
        self.noisy_total = float(d.sum()) * noise
        self.tightness = self.noisy_total / self.total_cap

    # ───────────────────────────── 观测 ─────────────────────────────
    def global_features(self):
        h = self.hour % 24
        per = np.zeros(3); per[specs.PERIOD[h]] = 1.0
        vs = self.last_vol / self.last_vol.sum() if self.last_vol.sum() > 0 else np.zeros(specs.G)
        return np.concatenate([[np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)], per,
                               [self.noisy_total / self.N / 20.0, self.tightness],
                               self.last_p1 / specs.P_ES, self.last_p2 / specs.P_ES, vs]).astype(np.float32)

    def central_state(self):
        return np.concatenate([self.global_features(), [self.S_hat / self.N / 20.0], self.addr_share]).astype(np.float32)

    def observe(self):
        gf = self.global_features()
        d, cap = self.d, self.cap
        urg = np.where(self.is_pros, np.maximum(0.0, (d - cap) / np.maximum(cap, 1e-9)), self.active.astype(float))
        own_oh = np.zeros((self.N, specs.G)); own_oh[self.idx_pros, self.own[self.idx_pros]] = 1.0
        ecc_ratio = np.where(self.active[:, None], np.minimum(self.Q * specs.P_ES / np.maximum(self.B, 1e-9)[:, None] / 5.0, 1.0), 0.0)
        per_g = np.stack([self.Gm.astype(float), ecc_ratio], axis=2).reshape(self.N, -1)
        base = np.stack([d / 100.0, cap / 100.0, self.elec / 0.75, np.minimum(urg, 5.0), self.c / 0.05, self.reserve / 0.05,
                         self.can_self.astype(float), self.is_pros.astype(float)], axis=1)
        retail = np.concatenate([base, own_oh, per_g, np.repeat(gf[None, :], self.N, 0)], axis=1).astype(np.float32)
        inter = []
        for m in range(self.M):
            per = np.stack([self.addr_share, self.supply_share, self.fill[m], self.leftover[m]], axis=1).reshape(-1)
            inter.append(np.concatenate([gf, [self.S_hat / self.N / 20.0, self.share[m], self.rho[m] / 0.05], per]))
        inter = np.array(inter, dtype=np.float32).reshape(self.M, -1) if self.M else np.zeros((0, len(gf) + 3 + 4 * specs.G), np.float32)
        return {'Prosumer': retail[self.idx_pros], 'PureBuyer': retail[self.idx_pb], 'Intermediary': inter}

    # ───────────────────────────── 散户决策 ─────────────────────────────
    def _bid_caps(self, need, retains):
        """出价上限矩阵 [N, G]（4.1、4.2）与最低有效价 lo = P^EB + min_m o_m。"""
        cap = np.minimum(specs.P_ES[None, :], (1.0 - self.cfg.delta) * self.B[:, None] / np.maximum(self.Q, 1e-9))
        theta_on = self.is_pros & retains & self.can_self
        Theta = self.cards_own * self.F_own * np.where(self.salv_ok, self.peb[self.own0], self.c)   # 自跑的机会成本（含可放弃的回购收入）
        cap = np.where(theta_on[:, None], np.minimum(cap, Theta[:, None] / np.maximum(self.Q, 1e-9)), cap)
        lo = self.peb + self.o_min
        valid = self.Gm & (need[:, None] > 1e-9) & (cap > lo[None, :] + 1e-12)
        return cap, lo, valid

    def _decode_retail(self, actions):
        N, G = self.N, specs.G
        sell = np.zeros(N, bool); selfa = np.zeros(N, bool); ws = np.zeros(N)
        phi = np.zeros((N, G)); wb = np.zeros((N, G))
        a = np.clip(np.asarray(actions['Prosumer'], dtype=np.float64), -1, 1)
        if len(self.idx_pros):
            sell[self.idx_pros] = a[:, 0] > 0
            selfa[self.idx_pros] = a[:, 1] > 0
            ws[self.idx_pros] = (a[:, 2] + 1) / 2
            phi[self.idx_pros] = a[:, 3::2][:, :G]
            wb[self.idx_pros] = (a[:, 4::2][:, :G] + 1) / 2
        b = np.clip(np.asarray(actions['PureBuyer'], dtype=np.float64), -1, 1)
        if len(self.idx_pb):
            phi[self.idx_pb] = b[:, 0::2][:, :G]
            wb[self.idx_pb] = (b[:, 1::2][:, :G] + 1) / 2
        is_self = self.can_self & selfa
        idle = np.where(self.is_pros, self.cap - np.where(is_self, self.cards_own * self.F_own, 0.0), 0.0)
        ask_mask = self.is_pros & sell & (idle > 1e-9) & ~self.blocked[self.own0]
        ask_price = self.floor_sell + ws * (specs.P_ES[self.own0] - self.floor_sell)
        need = np.where(self.active & ~is_self, self.d, 0.0)
        retains = ~sell
        cap, lo, valid = self._bid_caps(need, retains)
        price = lo[None, :] + wb * (cap - lo[None, :])
        order = np.argsort(-np.where(valid, phi, -np.inf), axis=1)
        nvalid = valid.sum(axis=1)
        return dict(is_self=is_self, idle=idle, ask_mask=ask_mask, ask_price=ask_price, need=need,
                    sell=sell, valid=valid, price=price, order=order, nvalid=nvalid, cap=cap)

    def _cda_quote(self, r, limit, ref, side, g):
        q = ref + r * (limit - ref)
        if side == 'sell':
            return np.minimum(np.maximum(q, limit), specs.P_ES[g])
        return np.minimum(np.maximum(q, self.peb[g]), limit)

    def _unified_prices(self, rule):
        """SDR / MMR 全市场统一价（按卡型分别计算；进口价 = P^ES，出口价 = P^EB）。
          SDR（Liu et al., IEEE TPWRS 2017）：SDR = S/D；SDR ≥ 1 时买卖价都等于出口价；
                0 ≤ SDR < 1 时 p_s = P^ES·P^EB / ((P^ES − P^EB)·SDR + P^EB)，p_b = p_s·SDR + P^ES·(1 − SDR)。无可调参数。
          MMR（Long et al., IEEE PES GM 2017）：中间价 p_mid = (P^ES + P^EB)/2；
                S ≥ D 时 p_b = p_mid，p_s = (D·p_mid + (S − D)·P^EB)/S（多余部分按出口价）；
                S < D 时 p_s = p_mid，p_b = (S·p_mid + (D − S)·P^ES)/D（不足部分按进口价）。唯一参数是中间价位置 0.5。
        供给 S 只计"在该价格下愿意卖"的闲置算力（出售底价 ≤ p_s），与出售决策同口径。两种规则下 p_s 都随 S 单调不增、
        意愿供给随价格单调不减，故 g(p) = p − p_s(S(p)) 单调递增，用二分法求唯一不动点。"""
        cfg = self.cfg
        idle_t = np.where(self.can_self, self.cap - self.cards_own * self.F_own, self.cap) * self.is_pros
        need_t = self.active & ~self.can_self
        # 每个买方的整单需求只计一次：按可行卡型的供给占比分摊到各层（单一商品时退化为原文献的 S/D；
        # 若在每个可行层都全额计入，显存要求低的任务（约 90%）会在 7 个层被重复计算，D 被高估数倍）
        feas = self.Gm & need_t[:, None]
        wgt = feas * self.supply_share[None, :]
        wsum = wgt.sum(axis=1, keepdims=True)
        wgt = np.where(wsum > 0, wgt / np.maximum(wsum, 1e-12), feas / np.maximum(feas.sum(axis=1, keepdims=True), 1))
        demand = (self.Q * wgt).sum(axis=0)
        pp = np.zeros(specs.G); pc = np.zeros(specs.G)
        for g in range(specs.G):
            PE, PI = self.peb[g], specs.P_ES[g]
            sel = self.is_pros & (self.own0 == g)
            fl, cap_g = self.floor_sell[sel], idle_t[sel]
            supply_at = lambda p: float(cap_g[fl <= p + 1e-12].sum())
            D = demand[g]

            def price(S):
                if rule == 'sdr':
                    if D <= 1e-9 or S >= D:
                        return PE, PE
                    r = S / D
                    p_s = (PI * PE) / ((PI - PE) * r + PE)
                    return p_s, p_s * r + PI * (1.0 - r)
                mid = PE + cfg.mmr_mid_pos * (PI - PE)
                if D <= 1e-9:
                    return PE, mid
                if S >= D:
                    return (D * mid + (S - D) * PE) / S, mid
                return mid, (S * mid + (D - S) * PI) / D

            lo, hi = PE, PI
            for _ in range(40):
                m = 0.5 * (lo + hi)
                if m - price(supply_at(m))[0] < 0:
                    lo = m
                else:
                    hi = m
            p_p, p_c = price(supply_at(hi))
            p_p = min(max(p_p, PE), PI); p_c = min(max(p_c, p_p), PI)
            pp[g], pc[g] = p_p, p_c
        return pp, pc

    def _decode_retail_rule(self, rule):
        """规则基准下散户的报价与参与决策（与原基准逻辑一致，口径统一到 G_{i,t} 与 4.1 上限）。"""
        N, G = self.N, specs.G
        if rule in ('sdr', 'mmr'):
            self._pp, self._pc = self._unified_prices(rule)
        own = self.own0
        if rule in ('sdr', 'mmr'):
            p_own = self._pp[own]
        elif rule == 'cda':
            ref = np.where(self.last_p1[own] > 0, self.last_p1[own], self.floor_sell)
            p_own = self._cda_quote(self.cda_r_retail, self.floor_sell, ref, 'sell', own)
        else:                                                             # mcafee：如实报价
            p_own = self.floor_sell.copy()
        unit_margin = np.where(p_own >= self.floor_sell - 1e-12, np.maximum(p_own - self.c, 0.0), 0.0)   # 低于保留价不卖
        util_self = np.where(self.can_self, -self.cards_own * self.F_own * self.c
                             + (self.cap - self.cards_own * self.F_own) * unit_margin, -np.inf)
        util_rent = self.cap * unit_margin - self.B
        is_self = self.can_self & (util_self >= util_rent)
        idle = np.where(self.is_pros, self.cap - np.where(is_self, self.cards_own * self.F_own, 0.0), 0.0)
        ask_mask = self.is_pros & (idle > 1e-9) & (p_own >= self.floor_sell - 1e-12) & ~self.blocked[own]   # 价格不低于保留价才出售
        need = np.where(self.active & ~is_self, self.d, 0.0)
        sell = ask_mask.copy()
        cap, lo, valid = self._bid_caps(need, ~sell)
        if rule in ('sdr', 'mmr'):
            # 统一价规则下中介是"价格跟随者"：零售挂牌价 = 规则买方价 + 本中介的整合成本（成本转嫁），买方接受不高于
            # 自己出价上限的挂牌价，按挂牌价成交（见 step 中 P2 的 posted 撮合）。这里用最低整合成本的挂牌价判断可行与排序
            price = np.repeat((self._pc + self.o_min)[None, :], N, 0)
        elif rule == 'cda':
            ref = np.where(self.last_p2[None, :] > 0, self.last_p2[None, :], cap)
            price = self._cda_quote(self.cda_r_retail[:, None], cap, ref, 'buy', np.arange(G)[None, :])
        else:
            price = cap.copy()
        # 参与条件与命题 2 一致：整单支出不劣于最优外部方案（上限 cap 已含 (1−δ) 与 Θ）
        valid = valid & (price <= cap + 1e-12)
        if rule == 'mcafee':                                              # 如实报价：按 ECC 成本偏好排序，可多轮
            key = np.where(valid, self.Q * specs.P_ES[None, :], np.inf)
            order = np.argsort(key, axis=1); nvalid = valid.sum(axis=1)
        else:                                                             # 按总支出由低到高排序，可多轮（与 MARL、McAfee 相同的二级市场规则）
            tot = np.where(valid, price * self.Q, np.inf)
            order = np.argsort(tot, axis=1); nvalid = valid.sum(axis=1)
        return dict(is_self=is_self, idle=idle, ask_mask=ask_mask, ask_price=p_own, need=need,
                    sell=sell, valid=valid, price=price, order=order, nvalid=nvalid, cap=cap)

    # ───────────────────────────── 中介决策 ─────────────────────────────
    NV_GATE = 0.3     # 售出率低于此值才按比例减少收购（高于此值说明能卖掉，报童分位数已包含了库存风险，不重复折减）
    NV_FLOOR = 0.02   # 收购量折减的下限：长期卖不掉时几乎停收，但保留极小的试探量

    def _nv_gate(self, st):
        return min(1.0, max(self.NV_FLOOR, st / self.NV_GATE)) if self.NV_GATE > 0 else 1.0

    def _newsvendor_qty(self, rule=None):
        """报童规则（全部规则基准的中介收购量，也是 2×2 归因中的"规则数量"）。
        q_{m,g} = θ̃_m · w_g · F̂^{-1}(cr_{m,g}) · s_{m,g}
          w_g    = 0.9·历史可成交占比 + 0.1·供给占比（后者避免从未成交的卡型永远不被收购）；
          cr     = (p2 − p1 − o)/(p2 − o − P^EB)：SDR/MMR 直接用本时段规则价（中介按 p_s 收购、按 p_b + o 挂牌，故 cr = (p_b − p_s)/(p_b − P^EB)）；
                   其余规则用跨天保留的成交价指数平滑 p̄1、p̄2（尚无成交时取 0.5）；
          s      = 售出率门控：本中介在该卡型上售出率（售出/收购）的指数平滑 ŝ（初值 1）低于 0.3 时按 ŝ/0.3 减少收购，
                   下限 0.02（长期卖不掉时几乎停收但保留试探量）；ŝ ≥ 0.3 时不折减（库存风险已由报童分位数处理）。"""
        w = 0.9 * self.addr_share + 0.1 * self.supply_share
        q = np.zeros((self.M, specs.G))
        uni = rule in ('sdr', 'mmr') and hasattr(self, '_pc')
        for m in range(self.M):
            for g in range(specs.G):
                o = self.o[m, g]
                if uni:
                    den = self._pc[g] - self.peb[g]
                    cr = (self._pc[g] - self._pp[g]) / den if den > 1e-12 else 0.5
                else:
                    p1, p2 = self.nv_p1[g], self.nv_p2[g]
                    den = p2 - o - self.peb[g]
                    cr = (p2 - p1 - o) / den if (p1 > 0 and p2 > 0 and den > 1e-12) else 0.5
                cr = min(max(cr, 0.05), 0.95)
                q[m, g] = self.share[m] * w[g] * float(np.quantile(self.S_samples, cr)) * self._nv_gate(self.nv_st[m, g])
        return q

    def _decode_inter(self, actions, rule):
        """返回 qty [M,G]、P1 出价 bid [M,G]，以及 P2 定价方式。"""
        M, G = self.M, specs.G
        # McAfee 对照：散户如实报价（该机制下的占优策略）；中介的估值取决于转售，不存在"如实"报价，
        # 故沿用与 CDA 基准相同的自适应报价规则，使两者只差在出清机制上。
        price_rule = ('cda' if rule == 'mcafee' else rule) if rule is not None else self.cfg.inter_price_rule
        qty_rule = 'newsvendor' if rule is not None else self.cfg.inter_qty_rule
        cap_bid = specs.P_ES[None, :] - self.o                             # 出价上限 P^ES − o_m（F2）
        a = np.clip(np.asarray(actions['Intermediary'], dtype=np.float64), -1, 1) if actions is not None and M else None
        if qty_rule == 'newsvendor':
            target = self._newsvendor_qty(rule if rule is not None else self.cfg.inter_price_rule)
        else:
            v = (a[:, 0::3] + 1) / 2 if self.cfg.inter_qty_map == 'unit' else np.maximum(0.0, a[:, 0::3])
            target = self.share[:, None] * self.S_hat * v / np.maximum(1.0, v.sum(axis=1, keepdims=True))
        qty = np.ceil(target / specs.F[None, :] - 1e-9) * specs.F[None, :]
        qty[:, self.blocked] = 0.0
        if price_rule is None:
            wb = (a[:, 1::3] + 1) / 2
            bid = self.peb[None, :] + wb * (cap_bid - self.peb[None, :])
        elif price_rule == 'cda':
            ref = np.where(self.last_p1[None, :] > 0, self.last_p1[None, :], cap_bid)
            bid = self._cda_quote(self.cda_r_inter[:, None], cap_bid, ref, 'buy', np.arange(G)[None, :])
        elif price_rule in ('sdr', 'mmr'):
            bid = np.minimum(np.repeat(self._pp[None, :], M, 0), cap_bid)
        else:
            raise ValueError(price_rule)
        ws = (a[:, 2::3] + 1) / 2 if price_rule is None else None
        return qty, bid, price_rule, ws

    def _inter_asks(self, inv, cost, price_rule, ws):
        floor = np.where(inv > 1e-9, cost / np.maximum(inv, 1e-9), 0.0) + self.o   # ≥ p^P1 + o，且 ≤ P^ES
        floor = np.minimum(floor, specs.P_ES[None, :])
        if price_rule is None:
            ask = floor + ws * (specs.P_ES[None, :] - floor)
        elif price_rule == 'cda':
            ref = np.where(self.last_p2[None, :] > 0, self.last_p2[None, :], floor)
            ask = self._cda_quote(self.cda_r_inter[:, None], floor, ref, 'sell', np.arange(specs.G)[None, :])
        elif price_rule in ('sdr', 'mmr'):
            ask = np.maximum(self._pc[None, :] + self.o, floor)      # 规则买方价 + 整合成本（成本转嫁），不低于成本底价
        else:
            ask = floor
        return ask

    # ───────────────────────────── 一步 ─────────────────────────────
    def step(self, actions=None, rule=None, record=False):
        """actions: {'Prosumer','PureBuyer','Intermediary': 动作数组}；rule ∈ RULES 时忽略 actions。"""
        cfg = self.cfg; G = specs.G; N, M = self.N, self.M
        mech = 'mcafee' if rule == 'mcafee' else 'mid'
        posted = (rule in ('sdr', 'mmr')) or (rule is None and cfg.inter_price_rule in ('sdr', 'mmr'))
        if rule is None and (cfg.inter_price_rule in ('sdr', 'mmr')):
            self._pp, self._pc = self._unified_prices(cfg.inter_price_rule)
        dec = self._decode_retail(actions) if rule is None else self._decode_retail_rule(rule)
        qty, bid, price_rule, ws = self._decode_inter(actions, rule)

        pay = np.zeros(N); recv = np.zeros(N); sold = np.zeros(N)
        inv = np.zeros((M, G)); cost = np.zeros((M, G))
        p1_price = np.zeros(G); p1_vol = np.zeros(G)
        op_surplus = 0.0
        p1_log = []
        # —— P1：一级市场 ——
        ask_ids = np.where(dec['ask_mask'])[0]
        for g in range(G):
            if self.blocked[g]:
                continue
            bids = [(bid[m, g], qty[m, g], m) for m in range(M) if qty[m, g] > 1e-9]
            sel = ask_ids[self.own0[ask_ids] == g]
            asks = [(dec['ask_price'][i], dec['idle'][i], i) for i in sel]
            pref, trades = clear_divisible(bids, asks, mech)
            if not trades:
                continue
            p1_price[g] = pref
            for m, i, q, pb, ps in trades:
                inv[m, g] += q; cost[m, g] += q * pb
                recv[i] += q * ps; sold[i] += q; p1_vol[g] += q
                op_surplus += q * (pb - ps)
                if record: p1_log.append((m, i, g, q, pb, ps))
        unsold_idle = np.where(self.is_pros, dec['idle'] - sold, 0.0)
        pros_sell_surplus = float(((recv - self.c * sold) * self.is_pros).sum())   # 产消者在一级市场卖算力的剩余

        # —— P2：二级市场（整单、单一卡型、单一来源，偏好分轮）——
        ask_p = self._inter_asks(inv, cost, price_rule, ws)
        avail = inv.copy()
        settled = np.zeros(N, bool)
        p2_val = np.zeros(G); p2_vol = np.zeros(G); p2_rev = np.zeros(M)
        p2_qty_m = np.zeros((M, G))
        eligible_unfilled = set()
        p2_log = []
        need = dec['need']; nvalid = dec['nvalid']; order = dec['order']; price = dec['price']
        buyers = np.where(need > 1e-9)[0]
        max_r = int(nvalid[buyers].max()) if len(buyers) else 0
        for r in range(max_r):
            cand = buyers[(~settled[buyers]) & (nvalid[buyers] > r)]
            if not len(cand):
                break
            gs = order[cand, r]
            for g in range(G):
                bi = cand[gs == g]
                if not len(bi):
                    continue
                asks = [(ask_p[m, g], avail[m, g], m) for m in range(M) if avail[m, g] > 1e-9]
                if not asks:
                    continue
                if posted:      # 统一价规则：买方接受不高于出价上限的挂牌价，按所选中介的挂牌价成交
                    bids = [(dec['cap'][i, g], self.Q[i, g], i) for i in bi]
                    pref, trades, unf = clear_indivisible(bids, asks, 'posted')
                else:
                    bids = [(price[i, g], self.Q[i, g], i) for i in bi]
                    pref, trades, unf = clear_indivisible(bids, asks, mech)
                eligible_unfilled.update(unf)
                for i, m, q, pb, ps in trades:
                    avail[m, g] -= q; settled[i] = True
                    pay[i] += q * pb; p2_rev[m] += q * ps; p2_qty_m[m, g] += q
                    p2_val[g] += q * pb; p2_vol[g] += q
                    op_surplus += q * (pb - ps)
                    if record: p2_log.append((i, m, g, q, pb, ps))
        p2_price = np.where(p2_vol > 0, p2_val / np.maximum(p2_vol, 1e-9), 0.0)

        # —— P3：外部兜底与清算 ——
        rem = (need > 1e-9) & ~settled
        fb_self = rem & self.can_self & (unsold_idle >= self.cards_own * self.F_own - 1e-9)   # "卖全部"可撤回
        unsold_idle = np.where(fb_self, unsold_idle - self.cards_own * self.F_own, unsold_idle)
        ecc_buy = rem & ~fb_self
        ecc_g = self.best_g
        ecc_qty = np.where(ecc_buy, self.Q[np.arange(N), ecc_g], 0.0)
        pay += np.where(ecc_buy, self.B, 0.0)
        self_used = np.where(dec['is_self'] | fb_self, self.cards_own * self.F_own, 0.0)
        salv_p = self.salv_ok & (unsold_idle > 1e-9)
        salv_qty_p = np.where(salv_p, unsold_idle, 0.0)
        recv += salv_qty_p * self.peb[self.own0]
        salv_inter = np.maximum(avail, 0.0)
        salv_rev = (salv_inter * self.peb[None, :]).sum(axis=1)

        # —— 效用、奖励、福利 ——
        U = recv - pay - self.c * (sold + self_used + salv_qty_p)
        rel = U - self.U_base
        Pi = p2_rev - cost.sum(axis=1) + salv_rev - (p2_qty_m * self.o).sum(axis=1)
        W = float(rel.sum() + Pi.sum() + op_surplus)    # McAfee 交易缩减的预算盈余归市场运营方，计入总剩余（与 LP 上界同口径）
        r_retail = rel if cfg.reward_mode == 'relative' else U
        rewards = {'Prosumer': r_retail[self.idx_pros], 'PureBuyer': r_retail[self.idx_pb], 'Intermediary': Pi}

        sellall_ecc = self.is_pros & (sold > 1e-9) & ecc_buy
        ecc_by_g = np.bincount(ecc_g[ecc_buy], weights=ecc_qty[ecc_buy], minlength=G)
        info = dict(
            hour=self.hour % 24, W=W, demand=float(self.d.sum()),
            rel_pros=float(rel[self.idx_pros].sum()), rel_pb=float(rel[self.idx_pb].sum()),
            inter_profit=Pi.copy(), p1_vol=float(p1_vol.sum()), p2_vol=float(p2_vol.sum()),
            pros_sell_surplus=pros_sell_surplus,
            idle_total=float(dec['idle'][self.is_pros].sum()), sold_total=float(sold[self.is_pros].sum()),
            reserve_rel=float((self.reserve / specs.P_ES[self.own0])[self.is_pros & (dec['idle'] > 1e-9)].mean()) if (self.is_pros & (dec['idle'] > 1e-9)).any() else 0.0,
            ecc_fallback_vol=float(ecc_qty.sum()), ecc_revenue=float(self.B[ecc_buy].sum()),
            salvage_inter_vol=float(salv_inter.sum()), salvage_pros_vol=float(salv_qty_p.sum()),
            ecc_buyback=float(salv_rev.sum() + (salv_qty_p * self.peb[self.own0]).sum()),
            self_vol=float(self_used.sum()), operator_surplus=op_surplus,
            n_fallback_self=int(fb_self.sum()), n_unfilled_eligible=int(len(eligible_unfilled - set(np.where(settled)[0]))),
            n_sellall_ecc=int(sellall_ecc.sum()),
            sellall_ecc_loss=float((-rel[sellall_ecc]).clip(min=0).sum()),
            n_buyers_active=int((need > 1e-9).sum()), n_p2_settled=int(settled.sum()),
            p1_price=p1_price, p2_price=p2_price, p1_vol_g=p1_vol, p2_vol_g=p2_vol, ecc_vol_g=ecc_by_g,
            elec_res=float(self.curves[0, self.hour % 24]), elec_com=float(self.curves[1, self.hour % 24]),
        )
        if record:
            info.update(p1_log=p1_log, p2_log=p2_log, rel=rel, U=U, inv=inv, p2_qty_m=p2_qty_m,
                        salv_inter=salv_inter, ecc_buy=ecc_buy, ecc_g=ecc_g, ecc_qty=ecc_qty, best_g=self.best_g.copy(),
                        sold=sold, is_self=dec['is_self'] | fb_self, c=self.c.copy(), active=self.active.copy(),
                        d=self.d.copy(), can_self=self.can_self.copy(), self_used=self_used, salv_p=salv_qty_p,
                        cap=dec['cap'], o=self.o, inter_names=list(self.inter_names),
                        # 算电协同补充分析（energy.py）用：本时段的保留价、出售底价与挂牌闲置（只读记录，不参与任何计算）
                        reserve=self.reserve.copy(), floor_sell=self.floor_sell.copy(),
                        idle=dec['idle'].copy(), ask_mask=dec['ask_mask'].copy())

        # —— 自适应报价代理（CDA）更新 ——
        if rule in ('cda', 'mcafee') or (rule is None and cfg.inter_price_rule == 'cda'):
            self._update_cda(rule, dec, qty, inv, p2_qty_m, sold, settled)

        # —— 历史与私有信号 ——
        self.addr_hist[self.hour % 24].append(p2_vol + ecc_by_g)
        bought = inv
        with np.errstate(invalid='ignore', divide='ignore'):
            self.fill = np.where(bought > 1e-6, np.minimum(1.0, p2_qty_m / np.maximum(bought, 1e-9)), self.fill)
            self.leftover = np.where(bought > 1e-6, np.minimum(1.0, salv_inter / np.maximum(bought, 1e-9)), self.leftover)
        self.last_p1 = p1_price; self.last_p2 = p2_price; self.last_vol = p1_vol + p2_vol
        self.nv_p1 = np.where(p1_vol > 0, np.where(self.nv_p1 > 0, 0.7 * self.nv_p1 + 0.3 * p1_price, p1_price), self.nv_p1)
        self.nv_p2 = np.where(p2_vol > 0, np.where(self.nv_p2 > 0, 0.7 * self.nv_p2 + 0.3 * p2_price, p2_price), self.nv_p2)
        st = np.where(inv > 1e-9, np.minimum(1.0, p2_qty_m / np.maximum(inv, 1e-9)), np.nan)
        self.nv_st = np.where(np.isnan(st), self.nv_st, np.clip(0.7 * self.nv_st + 0.3 * np.nan_to_num(st), 0.0, 1.0))
        # 适应性预期：观察到本卡型本时段的二级市场成交价后，按个体速度 λ_i 更新 P̂（无成交则不更新）
        hh = self.hour % 24
        seen = self.is_pros & (p2_price[self.own0] > 0)
        self.ref_p[seen, hh] += self.lam[seen] * (p2_price[self.own0[seen]] - self.ref_p[seen, hh])
        seen1 = self.is_pros & (p1_price[self.own0] > 0)
        first = seen1 & (self.ref_p1[:, hh] <= 0)
        self.ref_p1[first, hh] = p1_price[self.own0[first]]
        upd = seen1 & ~first
        self.ref_p1[upd, hh] += self.lam[upd] * (p1_price[self.own0[upd]] - self.ref_p1[upd, hh])

        self.hour += 1
        done = self.hour >= cfg.steps_per_day
        if not done:
            self._prepare_hour()
            obs = self.observe()
        else:
            obs = None
        return obs, rewards, done, info

    def _update_cda(self, rule, dec, qty, inv, p2_qty_m, sold, settled):
        """AA 式自适应：本期成交 → 激进度收敛；挂单未成交 → 更激进（只改报价代理状态，不改结算）。"""
        b, lo, hi = self.cfg.cda_beta, self.cfg.cda_r_lo, self.cfg.cda_r_hi
        if rule == 'cda':                          # 散户激进度（McAfee 下散户如实报价，不更新）
            posted = dec['ask_mask'] | (dec['nvalid'] > 0)
            traded = (sold > 1e-9) | settled
            r = self.cda_r_retail
            r = np.where(posted & traded, r - b * (r - lo), np.where(posted, r + b * (hi - r), r))
            self.cda_r_retail = np.clip(r, lo, hi)
        posted_m = (qty.sum(axis=1) > 1e-9) | (inv.sum(axis=1) > 1e-9)
        traded_m = (inv.sum(axis=1) > 1e-9) | (p2_qty_m.sum(axis=1) > 1e-9)
        r = self.cda_r_inter
        r = np.where(posted_m & traded_m, r - b * (r - lo), np.where(posted_m, r + b * (hi - r), r))
        self.cda_r_inter = np.clip(r, lo, hi)
