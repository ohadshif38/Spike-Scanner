"""
לוגיקת הניקוד של הסורק.
כל פונקציה מחזירה ציון 0-100 ורשימת "סיבות" קריאות, כדי שתמיד יהיה ברור
למה מניה קיבלה את הציון שלה.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# אינדיקטורים טכניים
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def compute_technicals(df: pd.DataFrame) -> dict | None:
    """df: היסטוריה יומית עם העמודות Open, High, Low, Close, Volume."""
    if df is None or df.empty:
        return None
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 25:
        return None

    close, high, low, vol, opn = df["Close"], df["High"], df["Low"], df["Volume"], df["Open"]
    last, prev = float(close.iloc[-1]), float(close.iloc[-2])
    vol_today = float(vol.iloc[-1])
    avg_vol20 = float(vol.iloc[-21:-1].mean())
    avg_vol5_prior = float(vol.iloc[-6:-1].mean())
    avg_vol_base = float(vol.iloc[-26:-6].mean()) if len(vol) >= 26 else avg_vol20

    high20_prior = float(high.iloc[-21:-1].max())
    low15 = float(low.iloc[-16:-1].min())
    high15 = float(high.iloc[-16:-1].max())
    high52 = float(high.max())

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr5, atr30 = float(tr.iloc[-6:-1].mean()), float(tr.iloc[-31:-1].mean())

    return {
        "price": last,
        "chg_pct": (last / prev - 1) * 100 if prev else 0.0,
        "gap_pct": (float(opn.iloc[-1]) / prev - 1) * 100 if prev else 0.0,
        "rvol": vol_today / avg_vol20 if avg_vol20 > 0 else 0.0,
        # האם נפח המסחר עלה כבר בימים שלפני היום (איסוף שקט)
        "accum_ratio": avg_vol5_prior / avg_vol_base if avg_vol_base > 0 else 1.0,
        "avg_vol20": avg_vol20,
        "avg_dollar_vol20": avg_vol20 * float(close.iloc[-21:-1].mean()),
        "vol_today": vol_today,
        "breakout20": last > high20_prior,
        "range15_pct": (high15 / low15 - 1) * 100 if low15 > 0 else 999.0,
        "atr_ratio": atr5 / atr30 if atr30 > 0 else 1.0,
        "dist_52w_pct": (last / high52 - 1) * 100 if high52 > 0 else -100.0,
        "above_sma20": last > float(close.iloc[-20:].mean()),
        "rsi": _rsi(close),
        "chg_5d_pct": (last / float(close.iloc[-6]) - 1) * 100 if len(close) > 6 else 0.0,
    }


def score_technical(t: dict) -> tuple[float, list[str]]:
    s, r = 0.0, []
    rvol = t["rvol"]
    s += min(rvol / 5, 1) * 35
    if rvol >= 2:
        r.append(f"נפח מסחר פי {rvol:.1f} מהממוצע")

    acc = t["accum_ratio"]
    if acc > 1.3:
        s += min((acc - 1) / 2, 1) * 15
        r.append(f"נפח עולה כבר כמה ימים (פי {acc:.1f})")

    if t["breakout20"]:
        s += 15
        r.append("פריצה מעל שיא 20 הימים")
        if t["range15_pct"] < 30:
            s += 10
            r.append(f"אחרי דשדוש צר ({t['range15_pct']:.0f}% טווח)")
    elif t["range15_pct"] < 20 and t["atr_ratio"] < 0.8:
        s += 8
        r.append("התכווצות תנודתיות (דחיסה לפני תנועה)")

    if t["dist_52w_pct"] > -10:
        s += 10
        r.append("קרוב לשיא שנתי")

    chg = t["chg_pct"]
    if 3 <= chg <= 40:
        s += 15
    elif 40 < chg <= 100:
        s += 8
    if t["gap_pct"] >= 5:
        r.append(f"פתיחה בפער של {t['gap_pct']:.0f}%")

    return min(s, 100.0), r


# ---------------------------------------------------------------------------
# מבנה: Float, שורט, רוטציה
# ---------------------------------------------------------------------------

def score_structure(float_shares, short_pct, vol_today, shares_out=None) -> tuple[float, list[str]]:
    s, r = 0.0, []
    fl = float_shares or shares_out
    if fl:
        m = fl / 1e6
        pts = 40 if m < 5 else 30 if m < 10 else 20 if m < 20 else 10 if m < 50 else 0
        s += pts
        if pts >= 20:
            r.append(f"Float נמוך ({m:.1f}M מניות)")
        if vol_today:
            rot = vol_today / fl
            s += 30 if rot > 1 else 20 if rot > 0.5 else 10 if rot > 0.2 else 0
            if rot > 0.5:
                r.append(f"רוטציית Float של {rot:.1f}x היום")
    if short_pct:
        sp = short_pct * 100 if short_pct < 1 else short_pct
        s += 30 if sp > 20 else 20 if sp > 10 else 10 if sp > 5 else 0
        if sp > 10:
            r.append(f"שורט {sp:.0f}% מה-Float")
    return min(s, 100.0), r


# ---------------------------------------------------------------------------
# קטליזטורים: חדשות ודיווחי SEC
# ---------------------------------------------------------------------------

KEYWORDS: dict[str, int] = {
    # קטליזטורים חזקים
    r"fda approv\w*": 30, r"\bapproves?\b": 20, r"breakthrough therapy": 25, r"fast track": 20,
    r"\bpdufa\b": 20, r"topline": 15, r"positive (?:results|data)": 20, r"primary endpoint": 20,
    r"to be acquired": 35, r"merger agreement": 25, r"definitive agreement": 20, r"tender offer": 30,
    r"\bbuyout\b": 25, r"\bacquisition\b": 12,
    # עסקיים
    r"\bcontract\b": 15, r"\bawarded?\b": 15, r"purchase order": 15, r"partnership": 15,
    r"collaboration": 12, r"licens\w+": 10, r"department of defense|\bdod\b|pentagon": 15,
    r"\bnasa\b": 12, r"\bpatent\b": 8, r"uplist\w*": 15, r"record revenue": 15,
    r"raises? guidance": 15, r"short squeeze": 10,
    # נושאים חמים (ספקולטיביים)
    r"artificial intelligence|\bai\b": 8, r"quantum": 10, r"bitcoin|crypto\w*|\bbtc\b": 8,
    r"\bnuclear\b": 8, r"\bdrones?\b": 8, r"rare earth": 10, r"robotic\w*": 6,
    # שליליים: דילול, מחיקה, בעיות
    r"public offering|registered direct|private placement|at-the-market|\batm\b": -30,
    r"\bpriced\b": -15, r"\bwarrants?\b": -10, r"reverse (?:stock )?split": -20,
    r"delist\w*|deficiency notice|minimum bid": -25, r"going concern": -30,
    r"bankruptcy|chapter 11": -40, r"class action|investigation": -15, r"\bhalt\w*": -5,
}
_KW = [(re.compile(k, re.I), w) for k, w in KEYWORDS.items()]

PROMO_PATTERNS = re.compile(
    r"could soar|next big|\d+x (?:potential|gain)|explod\w+|moon|skyrocket|massive upside|don'?t miss", re.I
)

DILUTION_FORMS = {"S-1", "S-1/A", "S-3", "S-3/A", "F-1", "F-3", "424B1", "424B2", "424B3",
                  "424B4", "424B5", "EFFECT"}
# פריטי 8-K מעניינים
ITEMS_8K = {
    "1.01": ("הסכם מהותי חדש", 15),
    "2.01": ("השלמת רכישה/מיזוג", 15),
    "2.02": ("דוח תוצאות", 5),
    "3.01": ("הודעה על אי עמידה בתנאי רישום", -20),
    "3.02": ("מכירת מניות לא רשומה (דילול)", -20),
    "5.02": ("שינוי בהנהלה", 0),
    "7.01": ("הודעה לשוק (Reg FD)", 5),
    "8.01": ("אירוע אחר", 5),
}


def score_headline(title: str) -> tuple[int, list[str]]:
    pts, hits = 0, []
    for rx, w in _KW:
        m = rx.search(title or "")
        if m:
            pts += w
            hits.append(m.group(0).lower())
    return pts, hits


def _age_hours(ts: datetime | None) -> float:
    if ts is None:
        return 999.0
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return max((datetime.now(timezone.utc) - ts).total_seconds() / 3600, 0)


def score_catalyst(news: list[dict], filings: list[dict]) -> tuple[float, list[str], dict]:
    """news: [{title, ts, ...}], filings: [{form, date, items, ...}]"""
    s, r = 0.0, []
    info = {"dilution_filing": False, "promo_news": 0, "neg_news": False, "fresh_news": 0}

    best_title, best_pts = None, 0
    for n in news:
        age = _age_hours(n.get("ts"))
        if age > 24 * 7:
            continue
        pts, hits = score_headline(n.get("title", ""))
        n["score"], n["hits"] = pts, hits
        decay = 1.0 if age < 24 else 0.6 if age < 72 else 0.25
        s += pts * decay
        if age < 48:
            info["fresh_news"] += 1
        if pts < -10 and age < 72:
            info["neg_news"] = True
        if PROMO_PATTERNS.search(n.get("title", "")):
            info["promo_news"] += 1
        if pts * decay > best_pts:
            best_pts, best_title = pts * decay, n.get("title")
    if best_title:
        r.append(f"כותרת: {best_title[:90]}")

    form4 = 0
    for f in filings:
        age_days = _age_hours(f.get("date")) / 24
        form = f.get("form", "")
        if form == "8-K" and age_days <= 5:
            for it in (f.get("items") or "").split(","):
                it = it.strip()
                if it in ITEMS_8K:
                    label, w = ITEMS_8K[it]
                    s += w
                    if w:
                        r.append(f"8-K: {label}")
        elif form in ("SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A") and age_days <= 14:
            s += 20
            r.append("דיווח 13D (משקיע מהותי/אקטיביסט)")
        elif form == "4" and age_days <= 14:
            form4 += 1
        elif form in DILUTION_FORMS and age_days <= 45:
            info["dilution_filing"] = True
    if form4:
        s += min(form4 * 4, 12)
        r.append(f"{form4} דיווחי בעלי עניין (Form 4) בשבועיים האחרונים")

    return float(np.clip(s, 0, 100)), r, info


# ---------------------------------------------------------------------------
# רשתות חברתיות
# ---------------------------------------------------------------------------

def score_social(ape: dict | None, twits: dict | None) -> tuple[float, list[str]]:
    s, r = 0.0, []
    if ape:
        m, m24 = ape.get("mentions", 0), ape.get("mentions_24h_ago", 0)
        accel = (m + 1) / (m24 + 1)
        if accel > 1:
            s += min(math.log2(accel) / 3, 1) * 50
        s += min(m / 50, 1) * 15
        rank, rank24 = ape.get("rank"), ape.get("rank_24h_ago")
        if rank and rank24 and rank24 > rank:
            s += min((rank24 - rank) / 50, 1) * 10
        if accel >= 2 and m >= 5:
            r.append(f"אזכורים ברדיט: {m} (פי {accel:.1f} מאתמול)")
        elif m >= 10:
            r.append(f"{m} אזכורים ברדיט ב-24 שעות")
    if twits:
        if twits.get("trending"):
            s += 15
            r.append("טרנדינג ב-StockTwits")
        mph = twits.get("msgs_per_hour", 0)
        s += min(mph / 10, 1) * 10
        bull = twits.get("bull_ratio")
        if bull is not None and twits.get("n_sentiment", 0) >= 5:
            r.append(f"StockTwits: {bull:.0%} שוריים, {mph:.1f} הודעות לשעה")
    return min(s, 100.0), r


# ---------------------------------------------------------------------------
# דגלים אדומים וציון משוקלל
# ---------------------------------------------------------------------------

@dataclass
class Flags:
    items: list[str] = field(default_factory=list)
    penalty: float = 0.0

    def add(self, text: str, pen: float):
        self.items.append(text)
        self.penalty += pen


def red_flags(t: dict, cat_info: dict, social: float, catalyst: float) -> Flags:
    f = Flags()
    if cat_info.get("dilution_filing"):
        f.add("דילול: דיווח הנפקה/רישום מניות לאחרונה", 15)
    if cat_info.get("neg_news"):
        f.add("חדשות שליליות (הנפקה/מחיקה/פיצול הפוך)", 10)
    if social >= 50 and catalyst < 10:
        f.add("באזז ברשת בלי קטליזטור אמיתי (חשד לקידום)", 15)
    if cat_info.get("promo_news"):
        f.add("שפה קידומית בכותרות", 10)
    if t["chg_pct"] > 100 or t["chg_5d_pct"] > 200:
        f.add("כבר עלתה חזק (מאוחר להיכנס?)", 10)
    if t["avg_dollar_vol20"] < 300_000:
        f.add("נזילות נמוכה", 8)
    if t["rsi"] > 85:
        f.add(f"RSI קיצוני ({t['rsi']:.0f})", 5)
    return f


def composite(scores: dict[str, float], weights: dict[str, float], penalty: float) -> float:
    tot_w = sum(weights.values()) or 1
    base = sum(scores[k] * weights[k] for k in weights) / tot_w
    return float(np.clip(base - penalty, 0, 100))
