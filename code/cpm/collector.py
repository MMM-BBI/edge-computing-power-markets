"""
评估期数据采集：把逐时成交日志汇总为论文需要的全部表格（设计文档第八、十一节）。

产出（每个评估目录 eval_<方法>/）
  daily.csv          逐日：W、总需求、单位福利、各角色剩余、各阶段成交量、ECC 流量、诊断计数
  hourly.csv         逐时（按天平均）：成交量、价差、电价、双向节点占比
  hourly_gpu.csv     逐时 × 卡型：一级/二级成交量加权价、成交量、ECC 兜底量
  inter.csv          每家中介：收购、零售、清算、利润、吞吐率、漏损率、汇集入度
  subst_market.csv / subst_all.csv   层间替代网络（外部最优卡型 → 实际成交卡型）
  sankey.csv         三类主体 + ECC 的日均流量（按卡型分层）
  price_hist.csv     二级成交价 p2/P^ES 的直方图（按卡型）与触及出价上限的成交占比
  agents.csv         每个散户的累计相对剩余
  summary.json       汇总指标与网络度量
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import specs

BINS = np.linspace(0, 1, 51)


class Collector:
    def __init__(self, env, method):
        self.env, self.method = env, method
        G, M, N = specs.G, env.M, env.N
        self.daily = []
        self.day_acc = None
        self.h = defaultdict(lambda: defaultdict(float))
        self.hg = defaultdict(lambda: defaultdict(float))
        self.inter = {k: defaultdict(float) for k in range(M)}
        self.indeg = np.zeros((M,)); self.indeg_n = np.zeros((M,))
        self.subst_m = np.zeros((G, G)); self.subst_a = np.zeros((G, G))
        self.sankey = defaultdict(float)
        self.edge = defaultdict(float)
        self.hist = np.zeros((G, len(BINS) - 1)); self.hist_cap = np.zeros((G, len(BINS) - 1)); self.at_cap = np.zeros(G); self.n_p2 = np.zeros(G)
        self.agent_rel = np.zeros(N); self.agent_dem = np.zeros(N)
        # —— 交易网络（网络科学指标）：逐日的边集合、节点度、成交量 ——
        self.vol_sell = np.zeros(N); self.vol_buy = np.zeros(N)
        self.d_sell = defaultdict(set); self.d_buy = defaultdict(set)          # 当天：散户 → 交易过的中介
        self.d_in = defaultdict(set); self.d_out = defaultdict(set)            # 当天：中介 → 交易过的散户
        self.prev_e1 = self.prev_e2 = None
        self.net_rows = []
        self.deg_hist = defaultdict(float)
        self.n_days = 0

    def start_day(self, day_id):
        self.day_acc = defaultdict(float); self.day_acc['day'] = day_id
        self.day_inter = np.zeros(self.env.M)

    def add(self, info):
        env = self.env; a = self.day_acc; h = info['hour']; G = specs.G
        for k in ('W', 'demand', 'rel_pros', 'rel_pb', 'p1_vol', 'p2_vol', 'ecc_fallback_vol', 'ecc_revenue',
                  'ecc_buyback', 'salvage_inter_vol', 'salvage_pros_vol', 'self_vol', 'operator_surplus',
                  'n_fallback_self', 'n_unfilled_eligible', 'n_sellall_ecc', 'sellall_ecc_loss',
                  'n_buyers_active', 'n_p2_settled', 'idle_total', 'sold_total', 'reserve_rel'):
            a[k] += info[k]
        self.day_inter += info['inter_profit']
        # —— 逐时 ——
        hh = self.h[h]
        hh['n'] += 1; hh['demand'] += info['demand']; hh['W'] += info['W']
        hh['p1_vol'] += info['p1_vol']; hh['p2_vol'] += info['p2_vol']; hh['ecc_vol'] += info['ecc_fallback_vol']
        hh['self_vol'] += info['self_vol']; hh['salvage_vol'] += info['salvage_inter_vol'] + info['salvage_pros_vol']
        hh['elec_res'] += info['elec_res']; hh['elec_com'] += info['elec_com']
        p1v, p2v = info['p1_vol_g'], info['p2_vol_g']
        both = (p1v > 0) & (p2v > 0)
        hh['spread_num'] += float(((info['p2_price'] - info['p1_price']) * p2v)[both].sum())
        hh['spread_den'] += float(p2v[both].sum())
        hh['p1_pv'] += float((info['p1_price'] * p1v).sum()); hh['p2_pv'] += float((info['p2_price'] * p2v).sum())
        a['p1_rel_num'] += float((info['p1_price'] * p1v / specs.P_ES).sum()); a['n_steps'] += 1
        for g in range(G):
            x = self.hg[(h, g)]
            x['n'] += 1; x['p1_pv'] += info['p1_price'][g] * p1v[g]; x['p1_vol'] += p1v[g]
            x['p2_pv'] += info['p2_price'][g] * p2v[g]; x['p2_vol'] += p2v[g]; x['ecc_vol'] += info['ecc_vol_g'][g]
        # —— 成交日志 → 网络度量 ——
        rel = info['rel']; self.agent_rel += rel; self.agent_dem += info['d']
        bought = np.zeros(env.N, bool)
        sellers_by = defaultdict(set)
        pros_mask = env.is_pros
        for (m, i, g, q, pb, ps) in info['p1_log']:
            self.vol_sell[i] += q; self.d_sell[i].add(m); self.d_in[m].add(i)
            self.inter[m]['p1_in'] += q; self.inter[m]['p1_cost'] += q * pb
            sellers_by[(m, g)].add(i)
            self.sankey[('Prosumer', f'Intermediary', g)] += q
            self.edge['seller_margin'] += q * (ps - info['c'][i])
        for (m, g), s in sellers_by.items():
            self.indeg[m] += len(s); self.indeg_n[m] += 1
        cap = info['cap']
        for (i, m, g, q, pb, ps) in info['p2_log']:
            self.vol_buy[i] += q; self.d_buy[i].add(m); self.d_out[m].add(i)
            bought[i] = True
            self.inter[m]['p2_out'] += q; self.inter[m]['p2_rev'] += q * ps
            role = 'Prosumer' if pros_mask[i] else 'PureBuyer'
            self.sankey[('Intermediary', role, g)] += q
            self.subst_m[info['best_g'][i], g] += q; self.subst_a[info['best_g'][i], g] += q
            r = pb / specs.P_ES[g]
            self.hist[g, min(int(r * (len(BINS) - 1)), len(BINS) - 2)] += 1
            self.n_p2[g] += 1
            rc = pb / max(cap[i, g], 1e-12)                  # 成交价 / 该买方的出价上限
            self.hist_cap[g, min(int(rc * (len(BINS) - 1)), len(BINS) - 2)] += 1
            if pb >= 0.99 * cap[i, g]:                       # 成交价达到出价上限的 99% 视为触及上限
                self.at_cap[g] += 1
            self.edge['buyer_gain'] += q * (specs.P_ES[g] - pb)
            self.edge['p2_value'] += q * pb
        # 中介边增值：p2 − p1 − o（按卡型进货均价）
        p1_cost_g = defaultdict(float); p1_q_g = defaultdict(float)
        for (m, i, g, q, pb, ps) in info['p1_log']:
            p1_cost_g[(m, g)] += q * pb; p1_q_g[(m, g)] += q
        for (i, m, g, q, pb, ps) in info['p2_log']:
            avg = p1_cost_g[(m, g)] / p1_q_g[(m, g)] if p1_q_g[(m, g)] > 0 else 0.0
            self.edge['inter_margin'] += q * (ps - avg - info['o'][m, g])
        for m in range(env.M):
            sal = info['salv_inter'][m]
            self.inter[m]['salvage'] += float(sal.sum())
            self.inter[m]['profit'] += float(info['inter_profit'][m])
            for g in range(G):
                if sal[g] > 0:
                    self.sankey[('Intermediary', 'ECC', g)] += sal[g]
        ecc_buy = info['ecc_buy']
        for i in np.where(ecc_buy)[0]:
            g = info['ecc_g'][i]; q = info['ecc_qty'][i]
            self.subst_a[info['best_g'][i], g] += q
            self.sankey[('ECC', 'Prosumer' if pros_mask[i] else 'PureBuyer', g)] += q
            bought[i] = True
        self.sankey[('Prosumer', 'Self', -1)] += info['self_vol']
        if info['salvage_pros_vol'] > 0:
            self.sankey[('Prosumer', 'ECC', -1)] += info['salvage_pros_vol']
        sold = info['sold'] > 1e-9
        bidir = pros_mask & sold & bought
        part = pros_mask & (sold | bought)
        hh['bidir'] += float(bidir.sum()); hh['participants'] += float(part.sum())
        hh['forced_buy'] += float((pros_mask & bought & info['active'] & ~info['can_self']).sum())

    def end_day(self):
        a = self.day_acc; env = self.env
        # 交易网络当天的结构：边 = (散户, 中介) 当天至少成交一次
        e1 = {(i, m) for i, ms in self.d_sell.items() for m in ms}
        e2 = {(i, m) for i, ms in self.d_buy.items() for m in ms}
        jac = lambda x, y: len(x & y) / len(x | y) if (x | y) else np.nan
        npr_, nall = max(int((env.is_pros & ~env.dropped).sum()), 1), max(int((~env.dropped).sum()), 1)
        self.net_rows.append(dict(
            day=a['day'], edges_p1=len(e1), edges_p2=len(e2),
            persist_p1=jac(e1, self.prev_e1) if self.prev_e1 is not None else np.nan,
            persist_p2=jac(e2, self.prev_e2) if self.prev_e2 is not None else np.nan,
            active_sellers=len(self.d_sell) / npr_, active_buyers=len(self.d_buy) / nall,
            seller_degree=np.mean([len(v) for v in self.d_sell.values()]) if self.d_sell else 0.0,
            buyer_degree=np.mean([len(v) for v in self.d_buy.values()]) if self.d_buy else 0.0,
            inter_in_degree=np.mean([len(self.d_in[m]) for m in range(env.M)]) if env.M else 0.0,
            inter_out_degree=np.mean([len(self.d_out[m]) for m in range(env.M)]) if env.M else 0.0))
        for v in self.d_sell.values(): self.deg_hist[('seller', len(v))] += 1
        for v in self.d_buy.values(): self.deg_hist[('buyer', len(v))] += 1
        for m in range(env.M):
            self.deg_hist[('inter_in', len(self.d_in[m]))] += 1; self.deg_hist[('inter_out', len(self.d_out[m]))] += 1
        self.prev_e1, self.prev_e2 = e1, e2
        self.d_sell.clear(); self.d_buy.clear(); self.d_in.clear(); self.d_out.clear()
        a['unit_W'] = 1e4 * a['W'] / max(a['demand'], 1e-9)
        a['inter_profit'] = float(self.day_inter.sum())
        for k, name in enumerate(env.inter_names):
            a[f'profit_{name}'] = float(self.day_inter[k])
        npb = max(int((~env.is_pros & ~env.dropped).sum()), 1)        # 人均指标只计仍在市场中的散户（网络冲击中退出者不计）
        npr = max(int((env.is_pros & ~env.dropped).sum()), 1)
        a['buyer_savings_pc'] = a['rel_pb'] / npb
        a['prosumer_savings_pc'] = a['rel_pros'] / npr
        a['inter_profit_avg'] = a['inter_profit'] / max(env.M, 1)
        self.daily.append(dict(a)); self.n_days += 1

    # ───────────────────────────── 写盘 ─────────────────────────────
    def save(self, out: Path):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        env = self.env; G = specs.G; nd = max(self.n_days, 1)
        daily = pd.DataFrame(self.daily); daily.to_csv(out / 'daily.csv', index=False)
        rows = []
        for h in sorted(self.h):
            x = self.h[h]; n = max(x['n'], 1)
            rows.append(dict(hour=h, period=specs.PERIOD_NAMES[specs.PERIOD[h]], demand=x['demand'] / n, W=x['W'] / n,
                             p1_vol=x['p1_vol'] / n, p2_vol=x['p2_vol'] / n, ecc_vol=x['ecc_vol'] / n,
                             self_vol=x['self_vol'] / n, salvage_vol=x['salvage_vol'] / n,
                             elec_res=x['elec_res'] / n, elec_com=x['elec_com'] / n,
                             p1_price=x['p1_pv'] / x['p1_vol'] if x['p1_vol'] else np.nan,
                             p2_price=x['p2_pv'] / x['p2_vol'] if x['p2_vol'] else np.nan,
                             spread=x['spread_num'] / x['spread_den'] if x['spread_den'] else np.nan,
                             bidir_share=x['bidir'] / x['participants'] if x['participants'] else 0.0,
                             bidir_n=x['bidir'] / n, forced_buy_n=x['forced_buy'] / n))
        pd.DataFrame(rows).to_csv(out / 'hourly.csv', index=False)
        rows = []
        for (h, g) in sorted(self.hg):
            x = self.hg[(h, g)]; n = max(x['n'], 1)
            rows.append(dict(hour=h, gpu=specs.GPU_NAMES[g],
                             p1_price=x['p1_pv'] / x['p1_vol'] if x['p1_vol'] else np.nan,
                             p2_price=x['p2_pv'] / x['p2_vol'] if x['p2_vol'] else np.nan,
                             p1_vol=x['p1_vol'] / n, p2_vol=x['p2_vol'] / n, ecc_vol=x['ecc_vol'] / n))
        pd.DataFrame(rows).to_csv(out / 'hourly_gpu.csv', index=False)
        rows = []
        tot_p2 = sum(self.inter[m]['p2_out'] for m in range(env.M)) or 1.0
        tot_p1 = sum(self.inter[m]['p1_in'] for m in range(env.M)) or 1.0
        for m, name in enumerate(env.inter_names):
            x = self.inter[m]
            rows.append(dict(inter=name, theta=env.theta[m], rho=env.rho[m],
                             p1_in=x['p1_in'] / nd, p2_out=x['p2_out'] / nd, salvage=x['salvage'] / nd,
                             profit=x['profit'] / nd, throughput=x['p2_out'] / x['p1_in'] if x['p1_in'] else np.nan,
                             leakage=x['salvage'] / x['p1_in'] if x['p1_in'] else np.nan,
                             retail_share=x['p2_out'] / tot_p2, purchase_share=x['p1_in'] / tot_p1,
                             p1_avg_price=x['p1_cost'] / x['p1_in'] if x['p1_in'] else np.nan,
                             p2_avg_price=x['p2_rev'] / x['p2_out'] if x['p2_out'] else np.nan,
                             in_degree=self.indeg[m] / self.indeg_n[m] if self.indeg_n[m] else 0.0))
        inter = pd.DataFrame(rows); inter.to_csv(out / 'inter.csv', index=False)
        for name, mat in (('subst_market', self.subst_m), ('subst_all', self.subst_a)):
            pd.DataFrame(mat / nd, index=specs.GPU_NAMES, columns=specs.GPU_NAMES).to_csv(out / f'{name}.csv')
        rows = [dict(source=s, target=t, gpu=specs.GPU_NAMES[g] if g >= 0 else 'all', value=v / nd)
                for (s, t, g), v in self.sankey.items()]
        pd.DataFrame(rows).to_csv(out / 'sankey.csv', index=False)
        rows = []
        for g in range(G):
            for b in range(len(BINS) - 1):
                rows.append(dict(gpu=specs.GPU_NAMES[g], bin_lo=BINS[b], bin_hi=BINS[b + 1], count=self.hist[g, b],
                                 count_over_cap=self.hist_cap[g, b]))
        pd.DataFrame(rows).to_csv(out / 'price_hist.csv', index=False)
        ag = pd.DataFrame(dict(agent=np.arange(env.N), role=np.where(env.is_pros, 'Prosumer', 'PureBuyer'),
                               gpu=[specs.GPU_NAMES[g] if g >= 0 else '' for g in env.own], n_gpus=env.n_gpus,
                               tariff=np.where(env.is_pros, np.array(specs.TARIFF_CLASSES)[env.tclass], ''),
                               surplus_per_day=self.agent_rel / nd, demand_per_day=self.agent_dem / nd,
                               in_market=~env.dropped))
        ag.to_csv(out / 'agents.csv', index=False)
        # —— 汇总与网络度量（第十一节 11.2）——
        D = daily['demand'].sum()
        sh = inter['retail_share'].values if len(inter) else np.array([])
        hourly = pd.DataFrame([dict(h=h, b=self.h[h]['bidir'], p=self.h[h]['participants']) for h in self.h])
        summ = dict(
            method=self.method, n_days=self.n_days,
            welfare_per_day=float(daily['W'].mean()), unit_welfare=float(1e4 * daily['W'].sum() / max(D, 1e-9)),
            inter_profit_avg=float(daily['inter_profit_avg'].mean()),
            buyer_savings_pc=float(daily['buyer_savings_pc'].mean()),
            prosumer_savings_pc=float(daily['prosumer_savings_pc'].mean()),
            p1_vol=float(daily['p1_vol'].mean()), p2_vol=float(daily['p2_vol'].mean()),
            ecc_fallback_vol=float(daily['ecc_fallback_vol'].mean()),
            ecc_revenue=float(daily['ecc_revenue'].mean()), ecc_buyback=float(daily['ecc_buyback'].mean()),
            operator_surplus=float(daily['operator_surplus'].mean()),
            internal_flow_share=float(daily['p2_vol'].sum() / max(daily['p2_vol'].sum() + daily['ecc_fallback_vol'].sum(), 1e-9)),
            hhi_retail=float((sh ** 2).sum()) if (len(sh) and sum(self.inter[m]['p2_out'] for m in range(env.M)) > 0) else np.nan,
            throughput=float(inter['p2_out'].sum() / inter['p1_in'].sum()) if len(inter) and inter['p1_in'].sum() else np.nan,
            leakage=float(inter['salvage'].sum() / inter['p1_in'].sum()) if len(inter) and inter['p1_in'].sum() else np.nan,
            in_degree=float(inter['in_degree'].mean()) if len(inter) else np.nan,
            bidir_share=float(hourly['b'].sum() / hourly['p'].sum()) if len(hourly) and hourly['p'].sum() else 0.0,
            cross_layer_share=float(1 - np.trace(self.subst_m) / self.subst_m.sum()) if self.subst_m.sum() else np.nan,
            share_p2_at_cap=float(self.at_cap.sum() / self.n_p2.sum()) if self.n_p2.sum() else np.nan,
            edge_seller_margin=self.edge['seller_margin'] / nd, edge_inter_margin=self.edge['inter_margin'] / nd,
            edge_buyer_gain=self.edge['buyer_gain'] / nd,
            fallback_self_per_day=float(daily['n_fallback_self'].mean()),
            unfilled_eligible_per_day=float(daily['n_unfilled_eligible'].mean()),
            sellall_ecc_per_day=float(daily['n_sellall_ecc'].mean()),
            sellall_ecc_loss_per_day=float(daily['sellall_ecc_loss'].mean()),
            p2_fill_rate=float(daily['n_p2_settled'].sum() / max(daily['n_buyers_active'].sum(), 1)),
            # 产消者出售侧：一级市场成交价 / ECC 零售价（量加权）、闲置算力售出比例、平均保留价 / ECC 零售价
            p1_price_rel=float(daily['p1_rel_num'].sum() / max(daily['p1_vol'].sum(), 1e-9)),
            idle_sold_share=float(daily['sold_total'].sum() / max(daily['idle_total'].sum(), 1e-9)),
            reserve_rel=float(daily['reserve_rel'].sum() / max(daily['n_steps'].sum(), 1)),
        )
        # —— 交易网络指标 ——
        net = pd.DataFrame(self.net_rows); net.to_csv(out / 'network_daily.csv', index=False)
        pd.DataFrame([dict(kind=k, degree=d, count=c) for (k, d), c in sorted(self.deg_hist.items())]).to_csv(
            out / 'network_degrees.csv', index=False)

        def gini(x):
            x = np.sort(np.asarray(x, float)); n = len(x)
            return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() > 0 else np.nan
        vs = self.vol_sell[env.is_pros & ~env.dropped]; vb = self.vol_buy[~env.dropped]
        top = lambda x: float(np.sort(x)[::-1][:max(1, len(x) // 10)].sum() / x.sum()) if x.sum() > 0 else np.nan
        cv = lambda x: float(np.std(x) / np.mean(x)) if len(x) and np.mean(x) > 0 else np.nan
        summ.update(
            net_persist_p1=float(net['persist_p1'].mean()) if len(net) else np.nan,
            net_persist_p2=float(net['persist_p2'].mean()) if len(net) else np.nan,
            net_edges_p1=float(net['edges_p1'].mean()) if len(net) else 0.0,
            net_edges_p2=float(net['edges_p2'].mean()) if len(net) else 0.0,
            net_active_sellers=float(net['active_sellers'].mean()) if len(net) else 0.0,
            net_active_buyers=float(net['active_buyers'].mean()) if len(net) else 0.0,
            net_seller_degree=float(net['seller_degree'].mean()) if len(net) else 0.0,
            net_buyer_degree=float(net['buyer_degree'].mean()) if len(net) else 0.0,
            net_inter_in_degree=float(net['inter_in_degree'].mean()) if len(net) else 0.0,
            net_inter_out_degree=float(net['inter_out_degree'].mean()) if len(net) else 0.0,
            net_gini_sellers=gini(vs), net_gini_buyers=gini(vb), net_top10_sellers=top(vs), net_top10_buyers=top(vb),
            net_cv_p1_daily=cv(daily['p1_vol'].values), net_cv_p2_daily=cv(daily['p2_vol'].values))
        with open(out / 'summary.json', 'w') as f:
            json.dump(summ, f, indent=1, default=float)
        return summ
