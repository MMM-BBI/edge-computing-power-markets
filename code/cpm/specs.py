"""
卡型参数、显存分档、电价曲线与产消者运行成本（设计文档第二、六、七节）。

全部物理/价格常数只在本文件定义一次，其余模块一律从这里引用。
"""
import math
import numpy as np

# ─────────────────────────── 卡型表（第九节 9.2 唯一参数表） ───────────────────────────
# power: FP32 单卡算力 TFLOPS；tdp: W；msrp: 元；vram: GB；ecc_sell: ECC 整卡每小时零售价（元/卡·h）
CARDS = {
    'RTX3070': dict(power=20.31, tdp=220.0, msrp=3899.0,  vram=8,  ecc_sell=0.78),
    'RTX3080': dict(power=29.77, tdp=320.0, msrp=5499.0,  vram=10, ecc_sell=0.93),
    'RTX3090': dict(power=35.58, tdp=350.0, msrp=11999.0, vram=24, ecc_sell=1.66),
    'RTX4060': dict(power=15.11, tdp=115.0, msrp=2399.0,  vram=8,  ecc_sell=0.58),
    'RTX4070': dict(power=29.15, tdp=200.0, msrp=4799.0,  vram=12, ecc_sell=1.11),
    'RTX4080': dict(power=48.74, tdp=320.0, msrp=9499.0,  vram=16, ecc_sell=1.86),
    'RTX4090': dict(power=82.58, tdp=450.0, msrp=12999.0, vram=24, ecc_sell=2.10),
}
GPU_NAMES = list(CARDS.keys())
G = len(GPU_NAMES)
GIDX = {g: i for i, g in enumerate(GPU_NAMES)}
# 持卡分布（Steam 硬件调查口径，顺序同 GPU_NAMES）
GPU_PROBS = np.array([0.17, 0.13, 0.04, 0.32, 0.18, 0.07, 0.09])

F = np.array([CARDS[g]['power'] for g in GPU_NAMES])             # TFLOPS
TDP = np.array([CARDS[g]['tdp'] for g in GPU_NAMES])
MSRP = np.array([CARDS[g]['msrp'] for g in GPU_NAMES])
VRAM = np.array([CARDS[g]['vram'] for g in GPU_NAMES])
P_ES = np.array([CARDS[g]['ecc_sell'] for g in GPU_NAMES]) / F    # 元/TFLOPS·h
IDX_4090 = GIDX['RTX4090']


def p_eb(alpha: float) -> np.ndarray:
    """ECC 回购价 P^EB = α·P^ES（元/TFLOPS·h）。"""
    return alpha * P_ES


# ─────────────────────────── 显存分档（第二节 2.3） ───────────────────────────
# 任务显存需求只通过"哪些卡可用"进入模型，故数据集中按阈值存为分档：
#   0: m ≤ 8   1: (8,10]   2: (10,12]   3: (12,16]   4: (16,24]；需求以 m ≤ 24 GB 为条件抽取，分档 5 不会出现
VRAM_EDGES = np.array([8.0, 10.0, 12.0, 16.0, 24.0])
N_TIERS = 6
TIER_MASK = np.zeros((N_TIERS, G), dtype=bool)          # 可行卡型集合 G_{i,t} 按分档查表
for _t, _upper in enumerate(VRAM_EDGES):                # 分档 t 的显存上界 → 显存不低于该值的卡可用
    TIER_MASK[_t] = VRAM >= _upper - 1e-9
TIER_MASK[5, IDX_4090] = True                           # 仅作保护（vram_cap=None 时才可能出现）


def vram_to_tier(m: np.ndarray) -> np.ndarray:
    """显存需求（GB）→ 分档 0..5。"""
    return np.searchsorted(VRAM_EDGES, m, side='left').astype(np.uint8)


# ─────────────────────────── 电价曲线（第七节） ───────────────────────────
PEAK = {6, 7, 18, 19, 20, 21}                      # 06–08、18–22 时，6 h
VALLEY = {0, 1, 2, 3, 11, 12, 13, 14, 15}           # 00–04、11–16 时，9 h；其余 9 h 为平段
PERIOD = np.array([0 if h in PEAK else (2 if h in VALLEY else 1) for h in range(24)])  # 0 峰 1 平 2 谷
PERIOD_NAMES = ['peak', 'flat', 'valley']

# 工商业：代理购电价 × 峰平谷比 1.68:1:0.48（大风季），加不浮动的线损、输配电价、系统运行费、政府性基金
# （内蒙古自治区发改委《关于完善蒙西电网工商业分时电价政策的通知（试行）》，2023-12-28 发布，2024-02-01 起试行）
_RATIO = np.array([1.68, 1.00, 0.48])
P_AGT = float(np.mean([0.2985, 0.3377, 0.3294]))    # 代理购电价格（三个已核实月份均值）
SYS_OP = float(np.mean([0.0069, 0.0209]))           # 系统运行费用
TD, FUNDS, LOSS_RATE = 0.1561, 0.022425, 0.0281
# 居民：一档电价 0.415 作平段，峰 +0.11、谷 −0.12（蒙西居民分时听证方案一）
RES_FLAT, RES_PEAK_ADD, RES_VALLEY_ADD = 0.415, 0.11, -0.12
TARIFF_CLASSES = ['residential', 'commercial']      # 0 = 个人（单卡），1 = 工作室（多卡）


def commercial_price(h: int, p_agt=P_AGT, sys_op=SYS_OP, loss=None) -> float:
    loss = p_agt * LOSS_RATE if loss is None else loss
    return p_agt * _RATIO[PERIOD[h]] + loss + TD + sys_op + FUNDS


def residential_price(h: int) -> float:
    p = PERIOD[h]
    return RES_FLAT + (RES_PEAK_ADD if p == 0 else RES_VALLEY_ADD if p == 2 else 0.0)


def tariff_curves(mode: str = 'tou', pv_scale: float = 1.0) -> np.ndarray:
    """返回 [2, 24] 电价矩阵（行 0 居民、行 1 工商业，元/kWh）。
    mode='tou' 为分时曲线；mode='flat' 为各自日均值的平坦电价（电价渠道识别对照，E8）。
    pv_scale（算电协同补充实验）：峰谷幅度缩放 e_h = ē + k·(e_h − ē)，日均电价 ē 不变（k = 0 即平坦电价，k = 1 即原分时电价）；
    取默认值 1 时不做任何运算，输出与原版逐位相同。"""
    c = np.array([[residential_price(h) for h in range(24)],
                  [commercial_price(h) for h in range(24)]], dtype=np.float64)
    if mode == 'flat':
        c = np.repeat(c.mean(axis=1, keepdims=True), 24, axis=1)
    elif mode != 'tou':
        raise ValueError(mode)
    if pv_scale != 1.0:
        mu = c.mean(axis=1, keepdims=True)
        c = mu + float(pv_scale) * (c - mu)
        if (c <= 0).any():
            raise ValueError(f'电价出现非正值（pv_scale={pv_scale}）')
    return c


def tariff_class(n_gpus: int) -> int:
    """单卡持有者 = 个人（居民电价）；多卡持有者 = 工作室（工商业电价）。"""
    return 0 if n_gpus <= 1 else 1


def tariff_selfcheck() -> float:
    """复现 2024-12 蒙西官方工商业分时价格（峰/平/谷），返回最大绝对误差。"""
    official = {6: 0.696305, 9: 0.493325, 11: 0.338105}
    return max(abs(commercial_price(h, 0.2985, 0.0069, 0.0094) - v) for h, v in official.items())


# ─────────────────────────── 运行成本（第六节） ───────────────────────────
def run_cost(g_idx, elec_price, eta: float, hours_life: float):
    """c^run = [e·η·TDP/1000 + MSRP/H] / F  （元/TFLOPS·h）。g_idx 与 elec_price 可为数组。
    全项目唯一的运行成本函数（F7）。"""
    g_idx = np.asarray(g_idx)
    return (np.asarray(elec_price) * eta * TDP[g_idx] / 1000.0 + MSRP[g_idx] / hours_life) / F[g_idx]


def cost_premise_table(alpha: float, eta: float, hours_life: float, tariff_mode='tou'):
    """各卡型、各电价类别下 c^run 相对 P^EB 与 P^ES 的范围（第六节"前提"自检）。"""
    curves = tariff_curves(tariff_mode)
    rows = []
    peb = p_eb(alpha)
    for gi, g in enumerate(GPU_NAMES):
        for ci, cname in enumerate(TARIFF_CLASSES):
            c = run_cost(gi, curves[ci], eta, hours_life)
            rows.append(dict(gpu=g, tariff=cname, c_min=c.min(), c_max=c.max(),
                             min_c_over_PEB=c.min() / peb[gi], max_c_over_PES=c.max() / P_ES[gi]))
    return rows


def whole_card_qty(d, g_idx=None):
    """Q_g(d) = ⌈d/F_g⌉·F_g。d 为 [N] 时返回 [N, G]；给定 g_idx 时返回对应卡型的量。"""
    d = np.asarray(d, dtype=np.float64)
    if g_idx is None:
        return np.ceil(d[..., None] / F - 1e-9) * F
    return np.ceil(d / F[g_idx] - 1e-9) * F[g_idx]


def cards_needed(d, g_idx):
    return np.ceil(np.asarray(d, dtype=np.float64) / F[g_idx] - 1e-9)
