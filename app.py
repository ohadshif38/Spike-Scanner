"""
סורק מומנטום למניות קטנות
הרצה:  streamlit run app.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import scoring as sc
import sources as src

st.set_page_config(page_title="סורק מניות קטנות", page_icon="📡", layout="wide")

st.markdown("""
<style>
[data-testid="stMarkdownContainer"], [data-testid="stHeading"], [data-testid="stCaptionContainer"],
[data-testid="stAlert"], [data-testid="stExpander"] summary, [data-testid="stWidgetLabel"],
[data-testid="stMetricLabel"] { direction: rtl; text-align: right; }
.flag { background:#FDECEA; color:#8A1C12; border-radius:4px; padding:2px 8px; margin:2px 0;
        display:inline-block; font-size:0.9rem; }
.reason { border-right:3px solid #0E7C86; padding:2px 10px; margin:4px 0; }
.news-pos { color:#0B6B3A; font-weight:600; } .news-neg { color:#A61B1B; font-weight:600; }
.small { color:#5B6573; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# גרסאות שמורות במטמון של משיכת הנתונים
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def c_universe(min_cap, max_cap, min_price, min_vol, majors):
    return src.get_universe(min_cap, max_cap, min_price, min_vol, majors)


@st.cache_data(ttl=900, show_spinner=False)
def c_history(tickers: tuple[str, ...]):
    return src.get_history(list(tickers))


@st.cache_data(ttl=600, show_spinner=False)
def c_apewisdom():
    return src.get_apewisdom()


@st.cache_data(ttl=600, show_spinner=False)
def c_trending():
    return src.get_stocktwits_trending()


@st.cache_data(ttl=86400, show_spinner=False)
def c_cik_map(email):
    return src.get_cik_map(email) if email else {}


@st.cache_data(ttl=600, show_spinner=False)
def c_press():
    return src.get_press_releases()


@st.cache_data(ttl=900, show_spinner=False)
def c_deep(ticker, name, email, finnhub_key, use_twits, cik_map_items):
    cik_map = dict(cik_map_items)
    news = src.merge_news(src.get_news_google(ticker, name), src.get_news_finnhub(ticker, finnhub_key),
                          src.get_news_yahoo_rss(ticker), src.get_news_yahoo(ticker))
    return {
        "info": src.get_info(ticker),
        "news": news,
        "filings": src.get_filings(ticker, cik_map, email) if email else [],
        "twits": src.get_stocktwits_stream(ticker) if use_twits else None,
    }


@st.cache_data(ttl=300, show_spinner=False)
def c_intraday(ticker):
    return src.get_intraday(ticker)


# ---------------------------------------------------------------------------
# סרגל צד: הגדרות
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("הגדרות סריקה")
    with st.form("settings"):
        cap_min, cap_max = st.slider("שווי שוק (מיליון $)", 1, 500, (5, 200))
        min_price = st.number_input("מחיר מינימלי ($)", 0.05, 50.0, 0.5, 0.05)
        min_vol = st.number_input("מחזור יומי ממוצע מינימלי (מניות)", 0, 5_000_000, 100_000, 50_000)
        majors = st.checkbox("רק NASDAQ / NYSE / AMEX (בלי OTC)", True)
        deep_n = st.slider("כמה מניות לנתח לעומק", 10, 80, 30,
                           help="המניות המובילות בשלב הראשון מקבלות ניתוח מלא: Float, שורט, חדשות, דיווחים")
        watchlist = st.text_input("רשימת מעקב (מופרדת בפסיקים)", "",
                                  help="מניות שתמיד ינותחו, גם אם לא עברו את הסינון")

        st.subheader("משקלות")
        w_tech = st.slider("טכני", 0, 100, 35)
        w_cat = st.slider("קטליזטורים (חדשות ודיווחים)", 0, 100, 30)
        w_soc = st.slider("רשתות חברתיות", 0, 100, 20)
        w_str = st.slider("מבנה (Float ושורט)", 0, 100, 15)

        st.subheader("מקורות")
        sec_email = st.text_input("מייל לזיהוי מול SEC", "",
                                  help="ה-SEC דורש כתובת מייל בכל בקשה. בלי זה לא ייבדקו דיווחים רשמיים.")
        finnhub_key = st.text_input("מפתח Finnhub (אופציונלי, חינמי)", "", type="password")
        use_twits = st.checkbox("לנסות StockTwits", True,
                                help="לפעמים StockTwits חוסם גישה אוטומטית. אם כך, הסריקה תמשיך בלעדיו.")
        run = st.form_submit_button("הרץ סריקה", type="primary", use_container_width=True)

    if st.button("נקה מטמון נתונים", use_container_width=True):
        st.cache_data.clear()
        st.success("המטמון נוקה. הסריקה הבאה תמשוך הכול מחדש.")

weights = {"tech": w_tech, "catalyst": w_cat, "social": w_soc, "structure": w_str}


# ---------------------------------------------------------------------------
# צינור הסריקה
# ---------------------------------------------------------------------------

def run_scan():
    log: list[str] = []
    prog = st.progress(0.0, "מאתר מניות שעומדות בסינון...")

    uni, errs = c_universe(cap_min * 1e6, cap_max * 1e6, min_price, min_vol, majors)
    log += errs
    wl = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
    tickers = list(dict.fromkeys((uni["ticker"].tolist() if not uni.empty else []) + wl))
    if not tickers:
        prog.empty()
        st.error("לא נמצאו מניות. אם Yahoo לא זמין כרגע, נסו להזין רשימת מעקב ולהריץ שוב.")
        return
    meta = uni.set_index("ticker").to_dict("index") if not uni.empty else {}

    prog.progress(0.15, f"מוריד היסטוריית מחירים ל-{len(tickers)} מניות...")
    hist = c_history(tuple(sorted(tickers)))

    prog.progress(0.45, "בודק אזכורים ברשתות...")
    ape = c_apewisdom()
    if not ape:
        log.append("ApeWisdom (אזכורים ברדיט) לא הגיב")
    trending = c_trending() if use_twits else set()
    press = c_press()
    if not press:
        log.append("פיד ההודעות לעיתונות לא הגיב")

    # שלב 1: ניקוד טכני + חברתי לכל המניות
    stage1 = []
    for t in tickers:
        tech = sc.compute_technicals(hist.get(t))
        if not tech:
            continue
        ts, tr = sc.score_technical(tech)
        ss, sr = sc.score_social(ape.get(t), {"trending": t in trending} if t in trending else None)
        prelim = (ts * w_tech + ss * w_soc) / max(w_tech + w_soc, 1)
        stage1.append({"ticker": t, "tech": tech, "ts": ts, "tr": tr, "ss": ss, "prelim": prelim})
    stage1.sort(key=lambda r: r["prelim"], reverse=True)

    have = {r["ticker"] for r in stage1}
    # מניות שהוציאו הודעה לעיתונות ב-36 השעות האחרונות נכנסות תמיד לניתוח עומק
    press_hits = [t for t, items in press.items() if t in have and
                  any(i.get("ts") and (datetime.now(timezone.utc) - i["ts"]).total_seconds() < 36 * 3600 for i in items)]
    deep_set = [r["ticker"] for r in stage1[:deep_n]] + [t for t in wl if t in have] + press_hits
    deep_set = list(dict.fromkeys(deep_set))

    # שלב 2: ניתוח עומק
    prog.progress(0.55, f"מנתח לעומק {len(deep_set)} מניות (חדשות, דיווחים, Float)...")
    cik_items = tuple(sorted(c_cik_map(sec_email).items())) if sec_email else tuple()
    if sec_email and not cik_items:
        log.append("SEC EDGAR לא הגיב; דיווחים רשמיים לא נבדקו")
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {t: ex.submit(c_deep, t, meta.get(t, {}).get("name") or "", sec_email, finnhub_key, use_twits, cik_items) for t in deep_set}
        deep = {}
        for i, (t, f) in enumerate(futures.items()):
            try:
                deep[t] = f.result()
            except Exception as e:
                log.append(f"{t}: {e}")
            prog.progress(0.55 + 0.4 * (i + 1) / len(futures), f"מנתח {t}...")

    by_t = {r["ticker"]: r for r in stage1}
    rows, details = [], {}
    for t in deep_set:
        r, d = by_t[t], dict(deep.get(t, {}))
        d["news"] = src.merge_news(press.get(t, []), d.get("news", []))
        fresh = [n["ts"] for n in d["news"] if n.get("ts")]
        news_age = (datetime.now(timezone.utc) - max(fresh)).total_seconds() / 3600 if fresh else None
        tech, info = r["tech"], d.get("info", {})
        twits = d.get("twits") or {}
        if t in trending:
            twits["trending"] = True
        st_s, st_r = sc.score_structure(info.get("floatShares"), info.get("shortPercentOfFloat"),
                                        tech["vol_today"], info.get("sharesOutstanding"))
        ca_s, ca_r, ca_info = sc.score_catalyst(d.get("news", []), d.get("filings", []))
        so_s, so_r = sc.score_social(ape.get(t), twits or None)
        flags = sc.red_flags(tech, ca_info, so_s, ca_s)
        scores = {"tech": r["ts"], "catalyst": ca_s, "social": so_s, "structure": st_s}
        total = sc.composite(scores, weights, flags.penalty)
        mcap = (meta.get(t, {}).get("mcap") or info.get("marketCap") or 0) / 1e6
        fl = info.get("floatShares") or info.get("sharesOutstanding")
        sp = info.get("shortPercentOfFloat")
        rows.append({
            "טיקר": t, "שם": meta.get(t, {}).get("name") or info.get("longName") or "",
            "ציון": round(total), "מחיר": tech["price"], "שינוי %": tech["chg_pct"],
            "RVOL": tech["rvol"], "שווי M$": mcap, "Float M": fl / 1e6 if fl else None,
            "שורט %": (sp * 100 if sp and sp < 1 else sp), "טכני": round(r["ts"]),
            "קטליזטור": round(ca_s), "חברתי": round(so_s), "מבנה": round(st_s),
            "חדשה אחרונה (שעות)": round(news_age) if news_age is not None else None,
            "דגלים": len(flags.items),
        })
        details[t] = {"scores": scores, "total": total, "tech": tech, "info": info,
                      "news": d.get("news", []), "filings": d.get("filings", []),
                      "reasons": r["tr"] + ca_r + so_r + st_r, "flags": flags.items,
                      "ape": ape.get(t), "twits": twits}

    others = pd.DataFrame([{
        "טיקר": r["ticker"], "מחיר": r["tech"]["price"], "שינוי %": r["tech"]["chg_pct"],
        "RVOL": r["tech"]["rvol"], "טכני": round(r["ts"]), "חברתי": round(r["ss"]),
    } for r in stage1 if r["ticker"] not in details])

    prog.empty()
    st.session_state.update({
        "results": pd.DataFrame(rows).sort_values("ציון", ascending=False) if rows else pd.DataFrame(),
        "details": details, "others": others, "log": log,
        "scanned_at": datetime.now(), "n_universe": len(tickers), "n_hist": len(stage1),
    })


if run:
    run_scan()


# ---------------------------------------------------------------------------
# עזרי תצוגה
# ---------------------------------------------------------------------------

def ago(ts):
    if not ts:
        return ""
    h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    if h < 1:
        return f"לפני {int(h * 60)} דק׳"
    if h < 48:
        return f"לפני {int(h)} שעות"
    return f"לפני {int(h / 24)} ימים"


def daily_chart(df: pd.DataFrame, ticker: str):
    df = df.tail(130)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                 name=ticker, increasing_line_color="#0B8A5A", decreasing_line_color="#C0392B"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="ממוצע 20",
                             line=dict(color="#0E7C86", width=1.4)), row=1, col=1)
    colors = ["#0B8A5A" if c >= o else "#C0392B" for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="נפח"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume"].rolling(20).mean(), name="נפח ממוצע",
                             line=dict(color="#5B6573", width=1, dash="dot")), row=2, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis_rangeslider_visible=False, plot_bgcolor="white")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], gridcolor="#EEF1F5")
    fig.update_yaxes(gridcolor="#EEF1F5")
    return fig


def intraday_chart(df: pd.DataFrame, ticker: str):
    fig = go.Figure(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                                   increasing_line_color="#0B8A5A", decreasing_line_color="#C0392B"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                      plot_bgcolor="white")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], gridcolor="#EEF1F5")
    return fig


# ---------------------------------------------------------------------------
# מסך ראשי
# ---------------------------------------------------------------------------

st.title("סורק מומנטום למניות קטנות")

tab_scan, tab_model = st.tabs(["סריקה", "איך הציון מחושב"])

with tab_scan:
    if "results" not in st.session_state:
        st.info("הגדירו את הסינון בסרגל הצד ולחצו על **הרץ סריקה**. סריקה ראשונה לוקחת בערך דקה עד שתיים; "
                "הבאות מהירות יותר בזכות המטמון. מומלץ להזין מייל כדי לקבל גם דיווחי SEC.")
    else:
        res: pd.DataFrame = st.session_state["results"]
        det: dict = st.session_state["details"]
        st.caption(f"נסרק ב-{st.session_state['scanned_at']:%d/%m %H:%M} · "
                   f"{st.session_state['n_universe']} מניות עברו סינון, "
                   f"{st.session_state['n_hist']} עם נתוני מחיר, {len(det)} נותחו לעומק")
        for m in st.session_state.get("log", []):
            st.warning(m)

        if res.empty:
            st.warning("אף מניה לא נותחה לעומק. נסו להרחיב את הסינון.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("ציון 60 ומעלה", int((res["ציון"] >= 60).sum()))
            c2.metric("RVOL מעל 3", int((res["RVOL"] >= 3).sum()))
            c3.metric("עם קטליזטור", int((res["קטליזטור"] >= 20).sum()))
            c4.metric("עם דגלים אדומים", int((res["דגלים"] > 0).sum()))

            only_clean = st.toggle("הסתר מניות עם דגלים אדומים", False)
            view = res[res["דגלים"] == 0] if only_clean else res

            st.dataframe(
                view, hide_index=True, use_container_width=True, height=min(38 * len(view) + 40, 640),
                column_config={
                    "ציון": st.column_config.ProgressColumn("ציון", min_value=0, max_value=100, format="%d"),
                    "מחיר": st.column_config.NumberColumn(format="$%.2f"),
                    "שינוי %": st.column_config.NumberColumn(format="%+.1f%%"),
                    "RVOL": st.column_config.NumberColumn(format="%.1fx"),
                    "שווי M$": st.column_config.NumberColumn(format="%.0f"),
                    "Float M": st.column_config.NumberColumn(format="%.1f"),
                    "שורט %": st.column_config.NumberColumn(format="%.1f%%"),
                    "דגלים": st.column_config.NumberColumn(format="%d ⚠️"),
                    "חדשה אחרונה (שעות)": st.column_config.NumberColumn(format="%d"),
                },
            )
            st.download_button("הורד כ-CSV", res.to_csv(index=False).encode("utf-8-sig"),
                               f"scan_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")

            st.divider()
            pick = st.selectbox("בחרו מניה לניתוח", view["טיקר"].tolist(),
                                format_func=lambda t: f"{t}  ·  ציון {int(res.set_index('טיקר').loc[t, 'ציון'])}")
            if pick:
                d = det[pick]
                info, tech = d["info"], d["tech"]
                st.subheader(f"{pick} · {info.get('longName') or ''}")
                st.caption(" · ".join(x for x in [info.get("sector"), info.get("industry"), info.get("country")] if x))
                st.markdown(
                    f"[Yahoo](https://finance.yahoo.com/quote/{pick}) · "
                    f"[Finviz](https://finviz.com/quote.ashx?t={pick}) · "
                    f"[StockTwits](https://stocktwits.com/symbol/{pick}) · "
                    f"[SEC](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={pick}&type=&dateb=&owner=include&count=40)")

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("ציון", f"{d['total']:.0f}")
                m2.metric("מחיר", f"${tech['price']:.2f}", f"{tech['chg_pct']:+.1f}%")
                m3.metric("RVOL", f"{tech['rvol']:.1f}x")
                fl = info.get("floatShares") or info.get("sharesOutstanding")
                m4.metric("Float", f"{fl / 1e6:.1f}M" if fl else "לא ידוע")
                m5.metric("RSI", f"{tech['rsi']:.0f}")

                left, right = st.columns([3, 2])
                with left:
                    ch1, ch2 = st.tabs(["גרף יומי", "תוך-יומי (כולל Pre-market)"])
                    with ch1:
                        h = c_history((pick,)).get(pick)
                        if h is not None and not h.empty:
                            st.plotly_chart(daily_chart(h, pick), use_container_width=True)
                    with ch2:
                        intr = c_intraday(pick)
                        if intr is not None and not intr.empty:
                            st.plotly_chart(intraday_chart(intr, pick), use_container_width=True)
                        else:
                            st.caption("אין נתונים תוך-יומיים זמינים כרגע.")
                with right:
                    s = d["scores"]
                    st.markdown("**פירוק הציון**")
                    for k, lbl in [("tech", "טכני"), ("catalyst", "קטליזטור"), ("social", "חברתי"), ("structure", "מבנה")]:
                        st.progress(int(s[k]) / 100, f"{lbl}: {s[k]:.0f}")
                    st.markdown("**למה היא ברשימה**")
                    for r in d["reasons"] or ["אין סיגנל בולט מעבר לציון הטכני"]:
                        st.markdown(f"<div class='reason'>{r}</div>", unsafe_allow_html=True)
                    if d["flags"]:
                        st.markdown("**דגלים אדומים**")
                        for f in d["flags"]:
                            st.markdown(f"<span class='flag'>⚠️ {f}</span>", unsafe_allow_html=True)

                n_col, f_col = st.columns([3, 2])
                with n_col:
                    st.markdown("**חדשות אחרונות**")
                    if not d["news"]:
                        st.caption("לא נמצאו חדשות. אפשר להוסיף מפתח Finnhub לכיסוי רחב יותר.")
                    for n in d["news"][:15]:
                        age_h = ((datetime.now(timezone.utc) - n["ts"]).total_seconds() / 3600) if n.get("ts") else 999
                        old = age_h > 24 * 7
                        pts = n.get("score", 0)
                        badge = (f"<span class='news-pos'>+{pts}</span>" if pts > 0 else
                                 f"<span class='news-neg'>{pts}</span>" if pts < 0 else "")
                        title = n["title"].replace("[", "(").replace("]", ")")
                        link = f"<a href='{n['url']}' target='_blank'>{title}</a>" if n.get("url") else title
                        pr = "📣 " if n.get("press") else ""
                        style = "text-align:left;opacity:0.45" if old else "text-align:left"
                        note = " · ישנה, לא נספרת בציון" if old else ""
                        st.markdown(f"<div dir='ltr' style='{style}'>{pr}{badge} {link}<br>"
                                    f"<span class='small'>{n.get('source', '')} · {ago(n.get('ts'))}{note}</span></div>",
                                    unsafe_allow_html=True)
                with f_col:
                    st.markdown("**דיווחי SEC (45 יום)**")
                    if not d["filings"]:
                        st.caption("אין דיווחים, או שלא הוזן מייל ל-SEC.")
                    for f in d["filings"][:15]:
                        dil = " ⚠️" if f["form"] in sc.DILUTION_FORMS else ""
                        items = f" (פריטים {f['items']})" if f.get("items") else ""
                        link = f"[{f['form']}]({f['url']})" if f.get("url") else f["form"]
                        st.markdown(f"{f['date']:%d/%m} · {link}{items}{dil}")
                    st.markdown("**רשתות**")
                    a, tw = d.get("ape"), d.get("twits") or {}
                    st.caption(f"רדיט: {a['mentions']} אזכורים (אתמול {a['mentions_24h_ago']})" if a
                               else "רדיט: לא ברשימת המוזכרות")
                    if tw.get("msgs_per_hour") is not None and "msgs_per_hour" in tw:
                        br = f", {tw['bull_ratio']:.0%} שוריים" if tw.get("bull_ratio") is not None else ""
                        st.caption(f"StockTwits: {tw['msgs_per_hour']:.1f} הודעות לשעה{br}")

        others = st.session_state.get("others")
        if others is not None and not others.empty:
            with st.expander(f"שאר המניות שנסרקו ({len(others)}), ניקוד ראשוני בלבד"):
                st.dataframe(others.sort_values("טכני", ascending=False), hide_index=True,
                             use_container_width=True)

with tab_model:
    st.markdown("""
הסורק עובד בשני שלבים. בשלב הראשון כל המניות שעברו את סינון שווי השוק מקבלות ציון טכני וחברתי,
שמבוססים על נתונים שאפשר למשוך במהירות לכולן. בשלב השני המניות המובילות מקבלות ניתוח מלא.

**טכני (0-100).** המרכיב הכבד ביותר הוא RVOL, כלומר נפח המסחר היום ביחס לממוצע 20 הימים; RVOL של 5 ומעלה נותן את מלוא 35 הנקודות.
נקודות נוספות ניתנות על עלייה בנפח כבר בימים שלפני היום (איסוף), פריצה מעל שיא 20 ימים, במיוחד אחרי דשדוש צר,
קרבה לשיא שנתי, ועלייה יומית סבירה של 3% עד 40%. מניה שכבר עלתה מעל 100% מקבלת פחות ודגל.

**קטליזטור (0-100).** חדשות נאספות מ-Google News, מ-Yahoo ומ-Finnhub (אם הוזן מפתח), ובנוסף מפיד ההודעות לעיתונות
של GlobeNewswire ו-PR Newswire. מניה שהוציאה הודעה לעיתונות ב-36 השעות האחרונות נכנסת תמיד לניתוח עומק.
כותרות מהשבוע האחרון נסרקות למילות מפתח עם משקלות: אישורי FDA, תוצאות ניסויים, חוזים,
שותפויות, רכישות ונושאים חמים מקבלים נקודות; הנפקות, פיצול הפוך, מחיקה מהמסחר ופשיטת רגל מורידים.
כותרות מהיממה האחרונה שוות יותר. בנוסף נבדקים דיווחי SEC: פריטים מסוימים ב-8-K, דיווחי 13D ודיווחי Form 4.

**חברתי (0-100).** העיקר הוא קצב השינוי באזכורים ברדיט לעומת אתמול, ולא הכמות עצמה.
נוספים לזה טרנדינג וקצב הודעות ב-StockTwits.

**מבנה (0-100).** Float נמוך, אחוז שורט גבוה, ונפח יומי שמתקרב לגודל ה-Float או עובר אותו.

**דגלים אדומים** מורידים נקודות מהציון הסופי: דיווח הנפקה או רישום מניות ב-45 הימים האחרונים,
באזז ברשתות בלי שום קטליזטור (דפוס נפוץ של קידום בתשלום), שפה קידומית בכותרות, נזילות נמוכה ו-RSI קיצוני.

המשקלות בסרגל הצד קובעים את התרומה של כל רכיב. מילות המפתח והמשקלות שלהן נמצאים בקובץ `scoring.py`
ואפשר לשנות אותם לפי הניסיון שלכם.
""")
    st.caption("הכלי הזה מזהה סיגנלים, לא ממליץ על קנייה או מכירה. במניות בגודל הזה רוב הסיגנלים לא מבשילים, "
               "ותנועות חדות לשני הכיוונים הן חלק מהשגרה.")
