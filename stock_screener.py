#!/usr/bin/env python3
"""
台股選股程式 —— 基本面 / 籌碼面 / 技術面 / 避雷賣出
收盤後批次執行：python stock_screener.py            (使用 WATCHLIST)
                python stock_screener.py 2330 2454  (指定代號)

安裝：pip install requests pandas yfinance
環境變數：FINMIND_TOKEN（選填，有 token 額度較高）
輸出：screener_result.csv、screener_report.html、cache.db（SQLite 快取）
"""
import os, sys, json, time, sqlite3, datetime as dt
import requests
import pandas as pd
import yfinance as yf

# ───────────────────────── 設定區（改這裡就好）─────────────────────────
WATCHLIST = ["2330", "2454", "3711", "2382", "3231", "6669", "2317", "2308"]
CYCLICAL = {"2603", "2609", "2615", "1301", "2002"}   # 景氣循環股：本益比邏輯反向，不用 PE 評分/否決

CFG = dict(
    weights=dict(fund=0.40, chip=0.30, tech=0.30),
    pass_score=65,          # 總分門檻
    roe_min=15,             # ROE %
    gm_min=30,              # 毛利率 %
    pe_cheap=12, pe_fair=15, pe_pricey=18, pe_sell=40,
    quality_min=0.5,        # 盈餘品質 = 營業現金流 / 稅後淨利
    trust_days=3,           # 投信連買天數
    big_holder_level="more than 1,000,001",  # 集保大戶級距（>1000張）
    bias_max=10,            # 月線乖離率上限 %
)
CACHE_TTL_H = dict(fund=24 * 7, chip=12, per=12)      # 快取時數：財報一週、籌碼半天
API = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.getenv("FINMIND_TOKEN", "")
DB = "cache.db"

# ───────────────────────── 資料層（FinMind + SQLite 快取）─────────────────────────
_con = sqlite3.connect(DB)
_con.execute("CREATE TABLE IF NOT EXISTS cache(k TEXT PRIMARY KEY, t REAL, v TEXT)")

def fm(dataset, sid, days, ttl_key):
    """呼叫 FinMind，結果快取，避免浪費免費額度"""
    key = f"{dataset}|{sid}|{days}"
    row = _con.execute("SELECT t, v FROM cache WHERE k=?", (key,)).fetchone()
    if row and time.time() - row[0] < CACHE_TTL_H[ttl_key] * 3600:
        return pd.DataFrame(json.loads(row[1]))
    start = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    p = dict(dataset=dataset, data_id=sid, start_date=start)
    if TOKEN: p["token"] = TOKEN
    r = requests.get(API, params=p, timeout=30).json()
    if r.get("status") != 200:
        print(f"  [FinMind] {dataset} {sid}: {r.get('msg')}")
        return pd.DataFrame()
    _con.execute("REPLACE INTO cache VALUES(?,?,?)", (key, time.time(), json.dumps(r["data"])))
    _con.commit()
    return pd.DataFrame(r["data"])

def pivot(df):
    if df.empty: return df
    return df.pivot_table(index="date", columns="type", values="value", aggfunc="last").sort_index()

def find_col(df, *keys):
    for c in df.columns:
        if all(k.lower() in c.lower() for k in keys): return c
    return None

# ───────────────────────── 一、基本面 ─────────────────────────
def fundamental(sid):
    s, flags, veto = {}, [], []
    inc = pivot(fm("TaiwanStockFinancialStatements", sid, 1000, "fund"))
    bs = pivot(fm("TaiwanStockBalanceSheet", sid, 1000, "fund"))
    cf = pivot(fm("TaiwanStockCashFlowsStatement", sid, 1000, "fund"))
    per = fm("TaiwanStockPER", sid, 30, "per")
    info = {}
    if len(inc) >= 8 and {"Revenue", "GrossProfit", "OperatingIncome", "IncomeAfterTaxes"} <= set(inc.columns):
        gm = inc.GrossProfit / inc.Revenue * 100
        om = inc.OperatingIncome / inc.Revenue * 100
        nm = inc.IncomeAfterTaxes / inc.Revenue * 100
        up = lambda x: x.tail(4).mean() > x.iloc[-8:-4].mean()   # 近4季均值 > 前4季（三率三升）
        info["毛利率"] = round(gm.tail(4).mean(), 1)
        s["毛利率≥30%"] = 20 if info["毛利率"] >= CFG["gm_min"] else 0
        s["營益率上升"] = 10 if up(om) else 0
        s["淨利率上升"] = 10 if up(nm) else 0
        if not up(gm) and gm.tail(4).mean() < gm.iloc[-8:-4].mean() - 2:
            flags.append("毛利率明顯下滑")
        ttm_ni = inc.IncomeAfterTaxes.tail(4).sum()
        if "EPS" in inc.columns: info["EPS(TTM)"] = round(inc.EPS.tail(4).sum(), 2)
        eq_col = "Equity" if "Equity" in bs.columns else find_col(bs, "equity")
        if eq_col and len(bs):
            roe = ttm_ni / bs[eq_col].tail(4).mean() * 100
            info["ROE"] = round(roe, 1)
            s["ROE≥15%"] = 25 if roe >= CFG["roe_min"] else 0
            if roe < CFG["roe_min"]: flags.append(f"ROE {roe:.1f}% < {CFG['roe_min']}%（壞了？）")
        # 現金流：取最近一個完整年度（台股現金流為累計值，取 12-31）
        oc, ic = find_col(cf, "operating"), find_col(cf, "investing")
        yr = cf[cf.index.str.endswith("12-31")]
        if oc and ic and len(yr):
            y = yr.index[-1][:4]
            ni_y = inc[inc.index.str.startswith(y)].IncomeAfterTaxes.sum()
            ocf, fcf = yr[oc].iloc[-1], yr[oc].iloc[-1] + yr[ic].iloc[-1]
            q = ocf / ni_y if ni_y > 0 else 0
            info["盈餘品質"] = round(q, 2)
            s["盈餘品質"] = 15 if q >= CFG["quality_min"] else 0
            s["自由現金流>0"] = 10 if fcf > 0 else 0
            if ocf < 0 and fcf < 0: veto.append("營業＆自由現金流皆為負")
    else:
        flags.append("財報資料不足")
    # 估值
    pe = float(per.PER.iloc[-1]) if len(per) and "PER" in per else None
    info["PE"] = pe
    if pe and sid not in CYCLICAL and pe > 0:
        s["估值"] = 10 if pe < CFG["pe_cheap"] else 6 if pe <= CFG["pe_fair"] else 2 if pe <= CFG["pe_pricey"] else 0
        if pe > CFG["pe_sell"]: veto.append(f"PE {pe:.0f} > 40（貴了）")
    elif sid in CYCLICAL:
        flags.append("景氣循環股：請人工判斷景氣位置")
    return sum(s.values()), s, flags, veto, info

# ───────────────────────── 二、籌碼面 ─────────────────────────
def chips(sid):
    s, flags, info = {}, [], {}
    ins = fm("TaiwanStockInstitutionalInvestorsBuySell", sid, 40, "chip")
    if len(ins):
        ins["net"] = ins.buy - ins.sell
        d = ins.pivot_table(index="date", columns="name", values="net", aggfunc="sum").sort_index()
        foreign = d[[c for c in d.columns if c.startswith("Foreign")]].sum(axis=1)
        trust = d["Investment_Trust"] if "Investment_Trust" in d else pd.Series(0, index=d.index)
        n = 0
        for v in trust.iloc[::-1]:
            if v > 0: n += 1
            else: break
        info["投信連買天數"], info["外資5日淨買(張)"], info["投信5日淨買(張)"] = n, int(foreign.tail(5).sum() / 1000), int(trust.tail(5).sum() / 1000)
        s["投信連買"] = 30 if n >= CFG["trust_days"] else 0
        s["外資5日買超"] = 20 if foreign.tail(5).sum() > 0 else 0
        s["土洋同向"] = 20 if foreign.tail(5).sum() > 0 and trust.tail(5).sum() > 0 else 0
        if foreign.tail(5).sum() > 0 > trust.tail(5).sum() or trust.tail(5).sum() > 0 > foreign.tail(5).sum():
            flags.append("土洋對作")
    hold = fm("TaiwanStockHoldingSharesPer", sid, 60, "chip")
    if len(hold):
        big = hold[hold.HoldingSharesLevel == CFG["big_holder_level"]].sort_values("date")
        if len(big) >= 2:
            chg = big.percent.iloc[-1] - big.percent.iloc[0]
            info["大戶持股%變化"] = round(chg, 2)
            s["大戶集中"] = 20 if chg > 0 else 0
    mg = fm("TaiwanStockMarginPurchaseShortSale", sid, 20, "chip")
    px = fm("TaiwanStockPrice", sid, 20, "chip")
    if len(mg) > 5 and len(px) > 5:
        m_chg = mg.MarginPurchaseTodayBalance.iloc[-1] / max(mg.MarginPurchaseTodayBalance.iloc[0], 1) - 1
        p_chg = px.close.iloc[-1] / px.close.iloc[0] - 1
        info["融資變化%"] = round(m_chg * 100, 1)
        if p_chg < 0 and m_chg > 0.1: flags.append("股價跌、融資增（籌碼凌亂）")
        else: s["融資健康"] = 10
    return sum(s.values()), s, flags, info

# ───────────────────────── 三、技術面 ─────────────────────────
def technical(sid):
    s, flags, info = {}, [], {}
    h = yf.Ticker(f"{sid}.TW").history(period="2y")
    if h.empty: h = yf.Ticker(f"{sid}.TWO").history(period="2y")
    if len(h) < 250: return 0, s, ["價格資料不足"], info
    c, v = h.Close, h.Volume
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 240)}
    last = c.iloc[-1]
    info["收盤"] = round(last, 2)
    s["站上季線"] = 25 if last > ma[60].iloc[-1] else 0
    s["站上年線"] = 20 if last > ma[240].iloc[-1] else 0
    cross = ((ma[5] > ma[20]) & (ma[5].shift(1) <= ma[20].shift(1))).tail(5).any()
    s["5/20黃金交叉"] = 20 if cross else 0
    near60 = abs(last / ma[60].iloc[-1] - 1) < 0.03 and last > ma[60].iloc[-1] * 0.97
    shrink = v.tail(3).mean() < v.tail(20).mean() * 0.7
    s["量縮回測季線"] = 20 if near60 and shrink else 0
    bias = (last / ma[20].iloc[-1] - 1) * 100
    info["月線乖離%"] = round(bias, 1)
    s["乖離不過大"] = 15 if bias < CFG["bias_max"] else 0
    if bias >= CFG["bias_max"]: flags.append("乖離過大，不追高")
    if v.iloc[-1] > v.tail(60).mean() * 3 and last < c.tail(60).max() * 0.92:
        flags.append("高檔爆量後轉弱")
    return sum(s.values()), s, flags, info

# ───────────────────────── 整合 ─────────────────────────
def screen(sid):
    print(f"分析 {sid} ...")
    f, fd, ff, veto, fi = fundamental(sid)
    c, cd, cf_, ci = chips(sid)
    t, td, tf, ti = technical(sid)
    w = CFG["weights"]
    total = round(f * w["fund"] + c * w["chip"] + t * w["tech"], 1)
    ok = total >= CFG["pass_score"] and not veto
    return dict(代號=sid, 總分=total, 基本面=f, 籌碼面=c, 技術面=t,
                結果="✅ 入選" if ok else ("⛔ 否決" if veto else "—"),
                否決原因="；".join(veto), 警示="；".join(ff + cf_ + tf),
                _detail=dict(基本面=fd, 籌碼面=cd, 技術面=td),
                **fi, **ci, **ti)

def to_html(df):
    css = "body{font-family:sans-serif;background:#111;color:#eee;padding:16px}table{border-collapse:collapse}" \
          "td,th{border:1px solid #444;padding:6px 10px;font-size:13px}th{background:#222}"
    return f"<meta charset=utf-8><style>{css}</style><h2>選股結果 {dt.date.today()}</h2>" + df.to_html(index=False, na_rep="")

if __name__ == "__main__":
    wl = open("watchlist.txt", encoding="utf-8").read().split() if os.path.exists("watchlist.txt") else WATCHLIST
    ids = sys.argv[1:] or wl
    rows = []
    for i in ids:
        try: rows.append(screen(i))
        except Exception as e: print(f"  {i} 失敗：{e}")
    detail = {r["代號"]: r.pop("_detail") for r in rows}
    df = pd.DataFrame(rows).sort_values("總分", ascending=False)
    payload = dict(updated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"), pass_score=CFG["pass_score"],
                   rows=json.loads(df.to_json(orient="records", force_ascii=False)), detail=detail)
    json.dump(payload, open("screener_result.json", "w", encoding="utf-8"), ensure_ascii=False)
    df.to_csv("screener_result.csv", index=False, encoding="utf-8-sig")
    open("screener_report.html", "w", encoding="utf-8").write(to_html(df))
    print(df[["代號", "總分", "基本面", "籌碼面", "技術面", "結果", "否決原因"]].to_string(index=False))
