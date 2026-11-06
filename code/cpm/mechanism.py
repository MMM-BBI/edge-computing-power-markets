"""
出清机制：统一价中点规则（本文机制）与 McAfee 交易缩减规则（机制对照，命题 3 之后的声明）。

两级市场共用同一套撮合：
  · 一级（产消者 → 中介）可分割：卖方逐张显卡可分，中介可从多个产消者处汇集同型号显卡；
  · 二级（中介 → 买方）不可分割：订单须由单一中介、单一卡型整单承接。
每笔成交返回 (buyer_id, seller_id, qty, pay_price, recv_price)：中点规则下两者相等（强预算平衡）；
McAfee 交易缩减规则下买方付 b_k、卖方收 s_k，差额归市场运营方（弱预算平衡），计入总剩余 W。
SDR/MMR 统一价规则的二级市场用挂牌价成交（rule='posted'）：成交价 = 中介挂牌价（规则买方价 + 整合成本）。
"""
EPS = 1e-9


def _walk(bids, asks):
    """按价格优先求可分割情形下的边际配对。bids/asks: [(price, qty, id)]，已排序。
    返回 (matched, i, j, rb, ra)：matched 为 [(bi, aj, q)]，i/j 为首个未完全成交的位置。"""
    rb = [b[1] for b in bids]; ra = [a[1] for a in asks]
    i = j = 0; matched = []
    while i < len(bids) and j < len(asks):
        if bids[i][0] < asks[j][0] - 1e-12:
            break
        q = min(rb[i], ra[j])
        if q > EPS:
            matched.append((i, j, q))
            rb[i] -= q; ra[j] -= q
        if rb[i] <= EPS: i += 1
        if ra[j] <= EPS: j += 1
    return matched, i, j, rb, ra


def _prices(bids, asks, matched, i, j, rule):
    """返回 (mode, pay, recv, p_ref)。mode='all'：全部边际配对成交；mode='reduce'：剔除最后一对。"""
    bk, sk = bids[matched[-1][0]][0], asks[matched[-1][1]][0]
    if rule == 'mid':
        c = (bk + sk) / 2.0
        return 'all', c, c, c
    # McAfee (1992)：p0 = (b_{k+1} + s_{k+1}) / 2；若 s_k ≤ p0 ≤ b_k 则全部按 p0 成交，否则剔除第 k 对
    b_next = bids[i][0] if i < len(bids) else None
    s_next = asks[j][0] if j < len(asks) else None
    if b_next is not None and s_next is not None:
        p0 = (b_next + s_next) / 2.0
        if sk - 1e-12 <= p0 <= bk + 1e-12:
            return 'all', p0, p0, p0
    return 'reduce', bk, sk, (bk + sk) / 2.0


def clear_divisible(bids, asks, rule='mid'):
    """一级市场。bids/asks: [(price, qty, id)]。返回 (p_ref, trades)。"""
    if not bids or not asks:
        return 0.0, []
    bids = sorted(bids, key=lambda x: -x[0]); asks = sorted(asks, key=lambda x: x[0])
    matched, i, j, _, _ = _walk(bids, asks)
    if not matched:
        return 0.0, []
    mode, pay, recv, pref = _prices(bids, asks, matched, i, j, rule)
    if mode == 'reduce':
        matched = matched[:-1]
    trades = [(bids[bi][2], asks[aj][2], q, pay, recv) for bi, aj, q in matched]
    return pref, trades


def clear_indivisible(bids, asks, rule='mid'):
    """二级市场：先按可分割情形定价，再按买方出价由高到低为其寻找库存足以整单承接的最廉中介。
    bids: [(price, qty, buyer_id)]；asks: [(price, qty, seller_id)]。
    返回 (p_ref, trades, eligible_unfilled)：后者为出价达标却因单一来源约束未成交的买方。"""
    if not bids or not asks:
        return 0.0, [], []
    bids = sorted(bids, key=lambda x: -x[0]); asks = sorted(asks, key=lambda x: x[0])
    if rule == 'posted':                    # 挂牌价成交（SDR/MMR 统一价规则）：买方按出价高低依次选库存足够、挂牌价不高于其出价的最廉中介
        avail = [a[1] for a in asks]
        trades, unfilled, val, vol = [], [], 0.0, 0.0
        for b in bids:
            pick = None; ok = False
            for a_k, a in enumerate(asks):
                if a[0] > b[0] + 1e-12:
                    break
                ok = True
                if avail[a_k] >= b[1] - 1e-9:
                    pick = a_k; break
            if pick is None:
                if ok:
                    unfilled.append(b[2])
                continue
            avail[pick] -= b[1]
            pr = asks[pick][0]
            trades.append((b[2], asks[pick][2], b[1], pr, pr)); val += pr * b[1]; vol += b[1]
        return (val / vol if vol else 0.0), trades, unfilled
    matched, i, j, _, _ = _walk(bids, asks)
    if not matched:
        return 0.0, [], []
    mode, pay, recv, pref = _prices(bids, asks, matched, i, j, rule)
    if mode == 'all':
        b_ok = lambda k: bids[k][0] >= pay - 1e-12
        a_ok = lambda k: asks[k][0] <= recv + 1e-12
    else:                                   # 交易缩减：边际买方与边际卖方出局
        kb, ka = matched[-1][0], matched[-1][1]
        b_ok = lambda k: k < kb
        a_ok = lambda k: k < ka
    avail = [a[1] for a in asks]
    trades, unfilled = [], []
    for k, b in enumerate(bids):
        if not b_ok(k):
            if mode == 'all':
                break
            continue
        need = b[1]
        pick = None
        for a_k, a in enumerate(asks):
            if not a_ok(a_k):
                break
            if avail[a_k] >= need - 1e-9:
                pick = a_k; break
        if pick is None:
            unfilled.append(b[2]); continue
        avail[pick] -= need
        trades.append((b[2], asks[pick][2], need, pay, recv))
    return pref, trades, unfilled
