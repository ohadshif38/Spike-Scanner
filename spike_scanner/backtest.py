"""
כיול משקלות: בודק לאחור מה היו ציוני הרכיבים של המניות שזינקו הכי הרבה,
ביום שלפני הזינוק, ומשווה אותם לכל שאר המניות באותו יום.

ההשוואה לשאר המניות חשובה: אם 90% מהמזנקות היו עם מעט מניות, אבל גם 90%
מכל המניות הקטנות הן כאלה, זה לא סיגנל. מה שקובע הוא כמה הרכיב מבדיל
בין מזנקות לבין השאר (AUC).
"""
from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import scoring as sc
import sources as src

# פתיחת המסחר בניו יורק (9:30) בשעון UTC בתקופת שעון קיץ אמריקאי; קירוב מספיק לצורך הניתוח
OPEN_UTC_HOUR, OPEN_UTC_MIN = 13, 30


def recent_trading_days(hist: dict[str, pd.DataFrame], n: int) -> list[pd.Timestamp]:
    counts = pd.Series(np.concatenate([df.index.values for df in hist.values()])).value_counts()
    common = sorted(counts[counts >= 0.5 * len(hist)].index)
    return [pd.Timestamp(d) for d in common[-n:]]


def build_panel(hist: dict[str, pd.DataFrame], meta: dict[str, dict], n_days: int,
                top_k: int) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    days = recent_trading_days(hist, n_days)
    recs = []
    for D in days:
        for t, df in hist.items():
            if D not in df.index:
                continue
            i = df.index.get_loc(D)
            if not isinstance(i, (int, np.integer)) or i < 31:
                continue
            prev = df.iloc[:i]  # כל הנתונים עד היום שלפני D כולל
            tech = sc.compute_technicals(prev)
            if not tech or tech["price"] <= 0:
                continue
            ts, _ = sc.score_technical(tech)
            shares = meta.get(t, {}).get("shares")
            ss, _ = sc.score_structure(None, None, tech["vol_today"], shares)
            row = df.iloc[i]
            recs.append({
                "date": D, "ticker": t,
                "ret": (row["Close"] / tech["price"] - 1) * 100,
                "high_ret": (row["High"] / tech["price"] - 1) * 100,
                "tech": ts, "structure": ss,
                "f_rvol2": tech["rvol"] >= 2,
                "f_accum": tech["accum_ratio"] > 1.3,
                "f_breakout": tech["breakout20"],
                "f_near_high": tech["dist_52w_pct"] > -10,
                "f_tight": tech["range15_pct"] < 20,
                "f_up_prev": tech["chg_pct"] >= 3,
                "f_small_shares": bool(shares and shares < 20e6),
            })
    panel = pd.DataFrame(recs)
    if panel.empty:
        return panel, days
    panel["rank"] = panel.groupby("date")["ret"].rank(ascending=False, method="first")
    panel["gainer"] = panel["rank"] <= top_k
    return panel, days


def add_catalysts(panel: pd.DataFrame, meta: dict[str, dict], email: str, control_per_day: int,
                  with_news: bool, progress=None) -> pd.DataFrame:
    """ציון קטליזטור לכל המזנקות ולמדגם ביקורת אקראי מכל יום.
    נספרים רק דיווחים עד היום שלפני, וחדשות שפורסמו לפני פתיחת המסחר ביום הזינוק."""
    rnd = random.Random(42)
    rows = []
    for D, g in panel.groupby("date"):
        rows += g[g["gainer"]].index.tolist()
        others = g[~g["gainer"]].index.tolist()
        rows += rnd.sample(others, min(control_per_day, len(others)))
    sample = panel.loc[rows].copy()

    cik_map = src.get_cik_map(email) if email else {}
    tickers = sorted(sample["ticker"].unique())
    filings: dict[str, list] = {}
    if cik_map:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for t, f in zip(tickers, ex.map(lambda t: src.get_filings(t, cik_map, email, days=75), tickers)):
                filings[t] = f

    def news_for(idx):
        r = sample.loc[idx]
        D = r["date"].date()
        return idx, src.get_news_google(r["ticker"], meta.get(r["ticker"], {}).get("name", ""),
                                        after=str(D - timedelta(days=7)), before=str(D + timedelta(days=1)))

    news: dict = {}
    if with_news:
        with ThreadPoolExecutor(max_workers=4) as ex:
            for n_done, (idx, items) in enumerate(ex.map(news_for, sample.index), 1):
                news[idx] = items
                if progress:
                    progress(n_done / len(sample))

    scores, has_news, has_8k, dil = [], [], [], []
    for idx, r in sample.iterrows():
        D = r["date"]
        ref = datetime(D.year, D.month, D.day, OPEN_UTC_HOUR, OPEN_UTC_MIN, tzinfo=timezone.utc)
        prev_day_end = datetime(D.year, D.month, D.day, tzinfo=timezone.utc)  # דיווחים עד D-1 בלבד
        fl = [f for f in filings.get(r["ticker"], []) if f.get("date") and f["date"] < prev_day_end]
        nw = [n for n in news.get(idx, []) if n.get("ts") and n["ts"] < ref]
        s, _, info = sc.score_catalyst([dict(n) for n in nw], fl, ref=ref)
        scores.append(s)
        has_news.append(any((ref - n["ts"]).total_seconds() < 24 * 3600 for n in nw))
        has_8k.append(any(f["form"] == "8-K" and (ref - f["date"]).days <= 5 for f in fl))
        dil.append(info["dilution_filing"])
    sample["catalyst"] = scores
    sample["f_news_24h"] = has_news
    sample["f_8k_5d"] = has_8k
    sample["f_dilution"] = dil
    return sample


def auc(pos: pd.Series, neg: pd.Series) -> float:
    """הסתברות שמזנקת אקראית קיבלה ציון גבוה יותר ממניה אקראית אחרת. 0.5 = אין מידע."""
    pos, neg = pos.dropna(), neg.dropna()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = pd.concat([pos, neg]).rank()
    return float((ranks.iloc[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


FEATURE_LABELS = {
    "f_rvol2": "נפח פי 2+ ביום שלפני",
    "f_accum": "נפח עולה כמה ימים (איסוף)",
    "f_breakout": "פריצה של שיא 20 יום",
    "f_near_high": "עד 10% משיא שנתי",
    "f_tight": "דשדוש צר (טווח < 20%)",
    "f_up_prev": "עלתה 3%+ ביום שלפני",
    "f_small_shares": "פחות מ-20M מניות",
    "f_news_24h": "חדשה ב-24 שעות לפני הפתיחה",
    "f_8k_5d": "דיווח 8-K בחמשת הימים שלפני",
    "f_dilution": "דיווח הנפקה/דילול לאחרונה",
}


def summarize(panel: pd.DataFrame, cat: pd.DataFrame | None, cur_weights: dict[str, int]):
    g, o = panel[panel["gainer"]], panel[~panel["gainer"]]
    comp = [
        {"key": "tech", "רכיב": "טכני", "מזנקות": g["tech"].mean(), "שאר המניות": o["tech"].mean(),
         "AUC": auc(g["tech"], o["tech"])},
        {"key": "structure", "רכיב": "מבנה", "מזנקות": g["structure"].mean(),
         "שאר המניות": o["structure"].mean(), "AUC": auc(g["structure"], o["structure"])},
    ]
    if cat is not None and "catalyst" in cat:
        cg, co = cat[cat["gainer"]], cat[~cat["gainer"]]
        comp.append({"key": "catalyst", "רכיב": "קטליזטור", "מזנקות": cg["catalyst"].mean(),
                     "שאר המניות": co["catalyst"].mean(), "AUC": auc(cg["catalyst"], co["catalyst"])})
    comp_df = pd.DataFrame(comp)

    feats = []
    for f, label in FEATURE_LABELS.items():
        src_df = cat if f in ("f_news_24h", "f_8k_5d", "f_dilution") else panel
        if src_df is None or f not in src_df:
            continue
        pg = src_df[src_df["gainer"]][f].mean() * 100
        po = src_df[~src_df["gainer"]][f].mean() * 100
        feats.append({"מאפיין": label, "% מהמזנקות": pg, "% משאר המניות": po,
                      "פי כמה נפוץ יותר": pg / po if po > 0 else np.nan})
    feat_df = pd.DataFrame(feats)

    # משקלות מוצעים: רכיב חברתי לא ניתן לשחזור לאחור, לכן נשאר כפי שהוא;
    # שאר המשקל מתחלק לפי כמה כל רכיב מבדיל (AUC מעל 0.5), עם רצפה קטנה נגד רעש
    social = cur_weights.get("social", 20)
    edges = {r["key"]: max((r["AUC"] if pd.notna(r["AUC"]) else 0.5) - 0.5, 0) + 0.02 for r in comp}
    if "catalyst" not in edges:
        edges["catalyst"] = None
    known = {k: v for k, v in edges.items() if v is not None}
    budget = 100 - social - (cur_weights["catalyst"] if edges["catalyst"] is None else 0)
    tot = sum(known.values())
    new = {k: round(budget * v / tot) for k, v in known.items()}
    if edges["catalyst"] is None:
        new["catalyst"] = cur_weights["catalyst"]
    new["social"] = social
    diff = 100 - sum(new.values())
    if diff:
        k = max(known, key=known.get)
        new[k] += diff

    # האם ציון גבוה ביום שלפני תרגם לתשואה טובה יותר?
    panel = panel.copy()
    panel["combo"] = (panel["tech"] * new["tech"] + panel["structure"] * new["structure"]) / max(
        new["tech"] + new["structure"], 1)
    top = panel.sort_values("combo", ascending=False).groupby("date").head(20)
    perf = {
        "top20_avg_ret": top["ret"].mean(), "all_avg_ret": panel["ret"].mean(),
        "top20_hit": top["gainer"].mean() * 100,
        "base_hit": panel["gainer"].mean() * 100,
        "top20_big": (top["ret"] >= 20).mean() * 100, "all_big": (panel["ret"] >= 20).mean() * 100,
    }
    return comp_df, feat_df, new, perf
