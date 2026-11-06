"""
训练与统一评估（设计文档第八节）。

  Trainer.train()   逐日训练；每 val_interval 回合在验证集上做确定性评估（只用于监控收敛）；
                    训练结束保存最后一帧 final.pt（正文用最后一帧，不按验证分择优）。
  evaluate(...)     所有方法（MARL 与全部规则基准）共用的唯一评估函数：
                    新建环境 → 用验证集最后 warmup_days 天预热（填满需求信号历史窗口、让价格预期收敛）→ 逐日评估测试集；
                    公共信号噪声由 (data_seed, 日, 小时) 确定，四种方法同一天的随机数完全相同。
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import specs
from .collector import Collector
from .env import MarketEnv, action_dims
from .learner import PPOLearner, Group, default_groups


def pick_device(cfg):
    if cfg.device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg.device


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_day(cfg, ds, split, k):
    if split == 'test':
        return ds['test_demands'][k], ds['test_tiers'][k], cfg.train_days + cfg.val_days + k
    off = 0 if split == 'train' else cfg.train_days
    return ds['train_demands'][off + k], ds['train_tiers'][off + k], off + k


def warm_env(env, cfg, ds, split, learner=None, rule=None):
    """评估前预热：在 split 的最后 warmup_days 天上运行（不记录），填满需求信号历史窗口，
    并让产消者对价格的适应性预期（P̂、ê）从初值收敛到当前策略下的水平。所有评估路径共用。"""
    n = cfg.train_days if split == 'train' else cfg.val_days
    with torch.no_grad():
        for k in range(max(n - cfg.warmup_days, 0), n):
            run_day(env, cfg, ds, split, k, learner=learner, rule=rule)


def build_learner(cfg, env, device, groups=None, trainable=None):
    obs = env.observe()
    obs_dims = {r: o.shape[1] for r, o in obs.items()}
    groups = groups or default_groups(env, cfg.share_intermediaries)
    return PPOLearner(cfg, groups, obs_dims, action_dims(), central_dim=len(env.central_state()),
                      device=device, trainable=trainable)


def run_day(env, cfg, ds, split, k, learner=None, rule=None, train=False, collector=None):
    """跑一天。learner 给出动作（rule=None）或用规则基准（rule ∈ RULES）。返回 (W, 需求, 各角色奖励和)。"""
    d, t, did = split_day(cfg, ds, split, k)
    obs = env.reset(d, t, did)
    if collector is not None:
        collector.start_day(did)
    tot = dict(W=0.0, D=0.0, inter=0.0, pros=0.0, pb=0.0, p1=0.0, p2=0.0, ecc=0.0, p1_rel=0.0, p2_rel=0.0, pros_sell=0.0, sellall_loss=0.0, reserve_rel=0.0, steps=0)
    done = False
    while not done:
        if rule is None:
            cs = env.central_state() if cfg.critic == 'central' else None
            acts = learner.act(obs, cs, deterministic=not train, store=train)
        else:
            acts = None
        obs, rew, done, info = env.step(acts, rule=rule, record=collector is not None)
        if train:
            learner.store_rewards(rew)
        if collector is not None:
            collector.add(info)
        tot['W'] += info['W']; tot['D'] += info['demand']; tot['inter'] += float(info['inter_profit'].sum())
        tot['pros'] += info['rel_pros']; tot['pb'] += info['rel_pb']
        tot['p1'] += info['p1_vol']; tot['p2'] += info['p2_vol']; tot['ecc'] += info['ecc_fallback_vol']
        tot['p1_rel'] += float((info['p1_price'] * info['p1_vol_g'] / specs.P_ES).sum())   # 成交价 / P^ES，按成交量加权
        tot['pros_sell'] += info['pros_sell_surplus']; tot['sellall_loss'] += info['sellall_ecc_loss']
        tot['reserve_rel'] += info['reserve_rel']; tot['steps'] += 1
        tot['p2_rel'] += float((info['p2_price'] * info['p2_vol_g'] / specs.P_ES).sum())
    if collector is not None:
        collector.end_day()
    return tot


class NetTracker:
    """训练中途验证时记录交易网络的形成：每天的一级/二级边数（散户–中介对）、活跃卖方比例、
    卖方/买方的度（每天交易的中介数）、零售流量在中介间的集中度 HHI。与 Collector 同接口，可传给 run_day。"""

    def __init__(self, env):
        self.env = env; self.rows = []

    def start_day(self, did):
        self.s1, self.s2 = {}, {}
        self.q2 = np.zeros(self.env.M)

    def add(self, info):
        for (m, i, g, q, pb, ps) in info['p1_log']:
            self.s1.setdefault(i, set()).add(m)
        for (i, m, g, q, pb, ps) in info['p2_log']:
            self.s2.setdefault(i, set()).add(m); self.q2[m] += q

    def end_day(self):
        env = self.env
        n_pr = max(int((env.is_pros & ~env.dropped).sum()), 1)
        sh = self.q2 / self.q2.sum() if self.q2.sum() > 0 else None
        self.rows.append(dict(edges_p1=sum(len(v) for v in self.s1.values()), edges_p2=sum(len(v) for v in self.s2.values()),
                              active_sellers=len(self.s1) / n_pr,
                              seller_degree=np.mean([len(v) for v in self.s1.values()]) if self.s1 else 0.0,
                              buyer_degree=np.mean([len(v) for v in self.s2.values()]) if self.s2 else 0.0,
                              hhi_retail=float((sh ** 2).sum()) if sh is not None else np.nan))

    def summary(self):
        d = pd.DataFrame(self.rows)
        return {f'val_net_{k}': float(d[k].mean()) for k in d.columns} if len(d) else {}


class Trainer:
    def __init__(self, cfg, ds, seed, out_dir, verbose=True):
        self.cfg, self.ds, self.seed = cfg, ds, seed
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        set_seed(seed)
        if cfg.torch_threads:
            torch.set_num_threads(cfg.torch_threads)
        self.device = pick_device(cfg)
        self.env = MarketEnv(cfg, ds, seed=seed)
        d, t, did = split_day(cfg, ds, 'train', 0)
        self.env.reset(d, t, did)
        self.learner = build_learner(cfg, self.env, self.device)
        self.val_env = MarketEnv(cfg, ds, seed=seed)

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def validate(self, all_days=False):
        cfg = self.cfg
        self.learner.eval()
        n = cfg.val_days if all_days else min(cfg.val_days_per_eval, cfg.val_days)
        idx = np.linspace(0, cfg.val_days - 1, n).astype(int)
        if not all_days:     # 训练中途的验证：产消者的价格预期与公共需求信号的历史取训练环境当前的状态（同一群人、同一策略），避免冷启动
            self.val_env.copy_state_from(self.env)
        tracker = NetTracker(self.val_env) if not all_days else None
        acc = dict(W=0.0, D=0.0, inter=0.0, pros=0.0, pb=0.0, p1=0.0, p1_rel=0.0, pros_sell=0.0, reserve_rel=0.0, steps=0)
        with torch.no_grad():
            for k in idx:
                r = run_day(self.val_env, cfg, self.ds, 'val', int(k), learner=self.learner, collector=tracker)
                for key in acc: acc[key] += r[key]
        self.learner.train()
        n = max(len(idx), 1)
        n_pb, n_pr, M = max(len(self.env.idx_pb), 1), max(len(self.env.idx_pros), 1), max(self.env.M, 1)
        # 确定性策略（无探索噪声）在验证日上的表现：收敛判断以它为准，训练曲线含探索噪声、会随退火"假性上升"
        return dict(val_unit_welfare=1e4 * acc['W'] / max(acc['D'], 1e-9), val_welfare=acc['W'] / n,
                    val_inter_profit_avg=acc['inter'] / n / M, val_buyer_savings_pc=acc['pb'] / n / n_pb,
                    val_prosumer_savings_pc=acc['pros'] / n / n_pr,
                    val_p1_price_rel=acc['p1_rel'] / max(acc['p1'], 1e-9), val_pros_sell_surplus_pc=acc['pros_sell'] / n / n_pr,
                    val_reserve_rel=acc['reserve_rel'] / max(acc['steps'], 1),
                    **(tracker.summary() if tracker is not None else {}))

    def train(self):
        cfg, L = self.cfg, self.learner
        n_ep = cfg.train_days
        rows, vrows = [], []
        n_pb, n_pr, M = max(len(self.env.idx_pb), 1), max(len(self.env.idx_pros), 1), max(self.env.M, 1)
        t0 = time.time()
        self.log(f'🚀 训练 {n_ep} 天（N={self.env.N}，|M|={self.env.M}，seed={self.seed}，device={self.device}）')
        for ep in range(n_ep):
            L.schedule(ep, n_ep)
            r = run_day(self.env, cfg, self.ds, 'train', ep, learner=L, train=True)
            L.update()
            rows.append(dict(episode=ep, welfare=r['W'], unit_welfare=1e4 * r['W'] / max(r['D'], 1e-9),
                             inter_profit_avg=r['inter'] / M, buyer_savings_pc=r['pb'] / n_pb,
                             prosumer_savings_pc=r['pros'] / n_pr, p1_vol=r['p1'], p2_vol=r['p2'], ecc_vol=r['ecc'],
                             p1_price_rel=r['p1_rel'] / max(r['p1'], 1e-9), p2_price_rel=r['p2_rel'] / max(r['p2'], 1e-9),
                             pros_sell_surplus_pc=r['pros_sell'] / n_pr, pros_sellall_loss_pc=r['sellall_loss'] / n_pr,
                             reserve_rel=r['reserve_rel'] / max(r['steps'], 1),
                             elapsed_min=(time.time() - t0) / 60))
            if (ep % cfg.val_interval == 0 and ep > 0) or ep == n_ep - 1:
                vrows.append(dict(episode=ep, **self.validate()))
                pd.DataFrame(rows).to_csv(self.out / 'train_log.partial.csv', index=False)   # 训练中途可查看进度
                pd.DataFrame(vrows).to_csv(self.out / 'val_log.partial.csv', index=False)
            if ep % max(cfg.val_interval, 1) == 0 or ep == n_ep - 1:
                vs = f" | 验证 {vrows[-1]['val_unit_welfare']:.2f}" if vrows else ''
                self.log(f"[Ep {ep:5d}] 单位福利 {rows[-1]['unit_welfare']:7.2f} | 中介均利 {rows[-1]['inter_profit_avg']:8.1f} | "
                         f"买方 {rows[-1]['buyer_savings_pc']:6.3f} | 产消者 {rows[-1]['prosumer_savings_pc']:6.3f}{vs} | "
                         f"{rows[-1]['elapsed_min']:.1f} min")
        pd.DataFrame(rows).to_csv(self.out / 'train_log.csv', index=False)
        pd.DataFrame(vrows).to_csv(self.out / 'val_log.csv', index=False)
        for f in ('train_log.partial.csv', 'val_log.partial.csv'):
            (self.out / f).unlink(missing_ok=True)
        torch.save(L.state_dict(), self.out / 'final.pt')
        # 最后一帧在全部验证日上的确定性表现：调参只用它选择（新建验证环境，避免受训练中验证顺序影响）
        self.val_env = MarketEnv(cfg, self.ds, seed=self.seed)
        self.learner.eval(); warm_env(self.val_env, cfg, self.ds, 'train', learner=self.learner); self.learner.train()
        val_final = self.validate(all_days=True)
        self.log(f"🏁 最后一帧验证（{cfg.val_days} 天）：单位福利 {val_final['val_unit_welfare']:.2f}")
        meta = dict(seed=self.seed, train_minutes=(time.time() - t0) / 60, device=self.device, val_final=val_final,
                    cfg=cfg.to_dict(), data=self.ds['path'],
                    gpu_mem_mb=(torch.cuda.max_memory_allocated() / 2 ** 20) if torch.cuda.is_available() else 0.0)
        with open(self.out / 'train_meta.json', 'w') as f:
            json.dump(meta, f, indent=1, default=str)
        return meta


def load_policy(cfg, ds, seed, ckpt, env=None, groups=None, trainable=None):
    device = pick_device(cfg)
    env = env or MarketEnv(cfg, ds, seed=seed)
    d, t, did = split_day(cfg, ds, 'val', 0)
    env.reset(d, t, did)
    L = build_learner(cfg, env, device, groups=groups, trainable=trainable)
    sd = torch.load(ckpt, map_location=device)
    L.load_state_dict(sd, strict_names=False)
    L.eval()
    return L


def evaluate(cfg, ds, seed, method, out_dir, learner=None, env_kwargs=None, verbose=True):
    """统一评估函数。method='marl' 时需传入 learner；否则为规则基准名。"""
    env_kwargs = env_kwargs or {}
    env = MarketEnv(cfg, ds, seed=seed, **env_kwargs)
    rule = None if method == 'marl' else method
    if learner is not None:
        learner.eval()
    t0 = time.time()
    with torch.no_grad():
        warm_env(env, cfg, ds, 'val', learner=learner, rule=rule)               # 预热：验证集最后若干天
        col = Collector(env, method)
        for k in range(cfg.test_days):
            run_day(env, cfg, ds, 'test', k, learner=learner, rule=rule, collector=col)
    summ = col.save(out_dir)
    summ['eval_minutes'] = (time.time() - t0) / 60
    with open(Path(out_dir) / 'summary.json', 'w') as f:
        json.dump(summ, f, indent=1, default=float)
    (Path(out_dir) / 'DONE').write_text('ok')
    if verbose:
        print(f"   [{method.upper():6s}] 单位福利 {summ['unit_welfare']:7.2f} | 日福利 {summ['welfare_per_day']:9.1f} | "
              f"中介均利 {summ['inter_profit_avg']:8.1f} | 内部流量占比 {summ['internal_flow_share']:.3f}", flush=True)
    return summ
