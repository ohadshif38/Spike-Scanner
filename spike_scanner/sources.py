"""
משיכת נתונים ממקורות חינמיים.
כל פונקציה עמידה לכשלים: אם מקור לא זמין היא מחזירה ערך ריק ולא מפילה את הסריקה.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
MAJOR_EXCHANGES = ["NMS", "NYQ", "ASE", "NCM", "NGM", "NAS"]


def _get_json(url: str, headers: dict | None = None, params: dict | None = None, timeout: int = 15):
    try:
        resp = requests.get(url, headers=headers or {"User-Agent": BROWSER_UA}, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# יקום המניות (Yahoo screener דרך yfinance)
# ---------------------------------------------------------------------------

def get_universe(min_cap: float, max_cap: float, min_price: float, min_avg_vol: float,
                 majors_only: bool = True) -> tuple[pd.DataFrame, list[str]]:
    from yfinance import EquityQuery as EQ

    errors: list[str] = []
    base = [
        EQ("eq", ["region", "us"]),
        EQ("btwn", ["intradaymarketcap", min_cap, max_cap]),
        EQ("gte", ["intradayprice", min_price]),
        EQ("gte", ["avgdailyvol3m", min_avg_vol]),
    ]
    queries = []
    if majors_only:
        try:
            queries.append(EQ("and", base + [EQ("is-in", ["exchange", *MAJOR_EXCHANGES])]))
        except Exception as e:
            errors.append(f"סינון בורסות נכשל ({e}), ממשיך בלי")
    queries.append(EQ("and", base))

    rows: dict[str, dict] = {}
    for q in queries:
        for sort_field in ("dayvolume", "percentchange"):
            try:
                res = yf.screen(q, sortField=sort_field, sortAsc=False, size=250)
                for qt in res.get("quotes", []):
                    sym = qt.get("symbol")
                    if sym and sym not in rows:
                        rows[sym] = {
                            "ticker": sym,
                            "name": qt.get("shortName") or qt.get("longName") or "",
                            "exchange": qt.get("exchange", ""),
                            "mcap": qt.get("marketCap"),
                            "q_price": qt.get("regularMarketPrice"),
                            "q_chg": qt.get("regularMarketChangePercent"),
                            "q_vol": qt.get("regularMarketVolume"),
                        }
            except Exception as e:
                errors.append(f"סורק Yahoo ({sort_field}): {e}")
        if rows:
            break  # השאילתה הראשונה הצליחה

    return pd.DataFrame(rows.values()), errors


# ---------------------------------------------------------------------------
# היסטוריית מחירים
# ---------------------------------------------------------------------------

def get_history(tickers: list[str], period: str = "1y", chunk: int = 80) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        try:
            data = yf.download(batch, period=period, interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
        except Exception:
            continue
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = set(data.columns.get_level_values(0))
            for t in batch:
                if t in lvl0:
                    out[t] = data[t].dropna(how="all")
        elif len(batch) == 1:
            out[batch[0]] = data.dropna(how="all")
    return out


def get_intraday(ticker: str) -> pd.DataFrame:
    """נרות של 5 דקות כולל Pre/After market, לגרף בדף המניה."""
    try:
        return yf.Ticker(ticker).history(period="5d", interval="5m", prepost=True)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# פרטי חברה וחדשות
# ---------------------------------------------------------------------------

def get_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}
    keys = ["floatShares", "sharesOutstanding", "shortPercentOfFloat", "sharesShort",
            "sector", "industry", "longName", "website", "totalCash", "marketCap", "country"]
    return {k: info.get(k) for k in keys}


def _parse_ts(val) -> datetime | None:
    if val is None:
        return None
    try:
        if isinstance(val, (int, float)):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except Exception:
        return None


def get_news_yahoo(ticker: str) -> list[dict]:
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw:
        c = item.get("content", item) if isinstance(item, dict) else {}
        url = c.get("canonicalUrl") or c.get("clickThroughUrl")
        url = url.get("url") if isinstance(url, dict) else c.get("link")
        prov = c.get("provider")
        prov = prov.get("displayName") if isinstance(prov, dict) else c.get("publisher")
        out.append({
            "title": c.get("title", ""),
            "url": url,
            "source": prov or "Yahoo",
            "ts": _parse_ts(c.get("pubDate") or c.get("providerPublishTime")),
        })
    return out


def get_news_finnhub(ticker: str, api_key: str, days: int = 7) -> list[dict]:
    if not api_key:
        return []
    to = datetime.now(timezone.utc).date()
    data = _get_json("https://finnhub.io/api/v1/company-news", params={
        "symbol": ticker, "from": str(to - timedelta(days=days)), "to": str(to), "token": api_key})
    if not isinstance(data, list):
        return []
    return [{"title": d.get("headline", ""), "url": d.get("url"), "source": d.get("source", "Finnhub"),
             "ts": _parse_ts(d.get("datetime"))} for d in data[:30]]


def merge_news(*lists: list[dict]) -> list[dict]:
    seen, out = set(), []
    for lst in lists:
        for n in lst:
            key = (n.get("title") or "").strip().lower()[:80]
            if key and key not in seen:
                seen.add(key)
                out.append(n)
    out.sort(key=lambda n: n.get("ts") or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
    return out


# ---------------------------------------------------------------------------
# SEC EDGAR (חינמי, דורש User-Agent עם מייל)
# ---------------------------------------------------------------------------

def get_cik_map(email: str) -> dict[str, int]:
    data = _get_json("https://www.sec.gov/files/company_tickers.json",
                     headers={"User-Agent": f"SpikeScanner {email}"})
    if not data:
        return {}
    return {v["ticker"].upper(): int(v["cik_str"]) for v in data.values()}


def get_filings(ticker: str, cik_map: dict[str, int], email: str, days: int = 45) -> list[dict]:
    cik = cik_map.get(ticker.upper().replace("-", "."), cik_map.get(ticker.upper()))
    if not cik:
        return []
    data = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json",
                     headers={"User-Agent": f"SpikeScanner {email}"})
    time.sleep(0.12)  # SEC מגביל ל-10 בקשות בשנייה
    if not data:
        return []
    rec = data.get("filings", {}).get("recent", {})
    forms, dates = rec.get("form", []), rec.get("filingDate", [])
    accs, docs, items = rec.get("accessionNumber", []), rec.get("primaryDocument", []), rec.get("items", [])
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out = []
    for i, form in enumerate(forms):
        d = _parse_ts(dates[i] + "T00:00:00+00:00") if i < len(dates) else None
        if not d or d < cutoff:
            break  # הרשימה ממוינת מהחדש לישן
        acc = accs[i].replace("-", "") if i < len(accs) else ""
        doc = docs[i] if i < len(docs) else ""
        out.append({
            "form": form, "date": d, "items": items[i] if i < len(items) else "",
            "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}" if acc else None,
        })
    return out


# ---------------------------------------------------------------------------
# רשתות חברתיות
# ---------------------------------------------------------------------------

def get_apewisdom(pages: int = 5) -> dict[str, dict]:
    """אזכורי מניות ברדיט (ApeWisdom, חינמי ובלי מפתח)."""
    out: dict[str, dict] = {}
    for p in range(1, pages + 1):
        data = _get_json(f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/{p}")
        if not data or not data.get("results"):
            break
        for r in data["results"]:
            def num(k):
                try:
                    return int(float(r.get(k) or 0))
                except (TypeError, ValueError):
                    return 0
            out[str(r.get("ticker", "")).upper()] = {
                "mentions": num("mentions"), "mentions_24h_ago": num("mentions_24h_ago"),
                "rank": num("rank"), "rank_24h_ago": num("rank_24h_ago"), "upvotes": num("upvotes"),
            }
        if p >= int(data.get("pages", 1) or 1):
            break
    return out


def get_stocktwits_trending() -> set[str]:
    data = _get_json("https://api.stocktwits.com/api/2/trending/symbols.json")
    if not data:
        return set()
    return {s.get("symbol", "").upper() for s in data.get("symbols", [])}


def get_stocktwits_stream(ticker: str) -> dict | None:
    """קצב הודעות וסנטימנט. StockTwits חוסם לפעמים גישה אוטומטית; אם כך, מחזיר None."""
    data = _get_json(f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json")
    if not data or "messages" not in data:
        return None
    msgs = data["messages"]
    times = [t for t in (_parse_ts(m.get("created_at")) for m in msgs) if t]
    bull = bear = 0
    for m in msgs:
        sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        bull += sent == "Bullish"
        bear += sent == "Bearish"
    hours = max((max(times) - min(times)).total_seconds() / 3600, 0.25) if len(times) > 1 else 24
    newest_age = (datetime.now(timezone.utc) - max(times)).total_seconds() / 3600 if times else 99
    return {
        "msgs_per_hour": len(times) / hours if newest_age < 12 else 0.0,
        "bull_ratio": bull / (bull + bear) if (bull + bear) else None,
        "n_sentiment": bull + bear,
        "watchlist": (data.get("symbol") or {}).get("watchlist_count"),
    }
