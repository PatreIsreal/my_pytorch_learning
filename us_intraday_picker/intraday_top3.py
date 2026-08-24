#!/usr/bin/env python3
"""Produce a latest-public-snapshot U.S. activity Top 3 ranking.

This is an intraday/premarket monitoring task, separate from the audited close-to-
close daily selector. It combines:

* Nasdaq's public stock-screener snapshot for current price, percentage change,
  and cumulative share volume (the public surface can be delayed); and
* the same SEC point-in-time ``dei:EntityPublicFloat`` methodology used by the
  audited daily backtest to infer free-float shares.

Activity is estimated as cumulative dollar volume / current free-float market
cap. Because the public screener does not expose consolidated VWAP/dollar volume,
cumulative dollar volume is approximated by last sale * cumulative volume; the
price cancels algebraically, so the ranking is equivalent to cumulative share
volume / inferred free-float shares.

The program generates research rankings only. It does not place orders.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import exchange_calendars as xcals
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import us_daily_picker.picker as daily

BJT = ZoneInfo("Asia/Shanghai")
ET = ZoneInfo("America/New_York")
NASDAQ_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=10000&offset=0&download=true"
)

# Preserve the audited point-in-time float assumptions.
MAX_PUBLIC_FLOAT_AGE_DAYS = daily.MAX_PUBLIC_FLOAT_AGE_DAYS
MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS = daily.MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS
BAR_HISTORY_BUFFER_DAYS = daily.BAR_HISTORY_BUFFER_DAYS
MIN_PRICE = 1.0
MIN_FLOAT_MARKET_CAP = 20_000_000.0
DROP_THRESHOLD_PCT = -8.0
RANK_LIMIT = 3

# The SEC bulk endpoint accepted this descriptive identity during the audited run.
daily.SEC_USER_AGENT = "OpenAI point-in-time market research support@openai.com"


def scalar(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=scalar, allow_nan=False)


def append_output(**values: Any) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={'' if value is None else value}\n")


def append_summary(markdown: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n")


def browser_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Origin": "https://www.nasdaq.com",
            "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"--", "N/A", "NA", "null", "None"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()$% ").replace(",", "")
    multiplier = 1.0
    if text and text[-1].upper() in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
        text = text[:-1]
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return -number if negative else number


def extract_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    candidates = [
        data.get("rows"),
        (data.get("table") or {}).get("rows") if isinstance(data.get("table"), dict) else None,
    ]
    for candidate in candidates:
        if isinstance(candidate, list):
            return [item for item in candidate if isinstance(item, dict)]
    return []


def download_nasdaq_snapshot(now_utc: datetime) -> tuple[pd.DataFrame, dict[str, Any]]:
    session = browser_session()
    response = session.get(NASDAQ_SCREENER_URL, timeout=(20, 90))
    response.raise_for_status()
    payload = response.json()
    rows = extract_rows(payload)
    if len(rows) < 500:
        raise RuntimeError(f"Nasdaq screener returned only {len(rows)} rows")

    normalized: list[dict[str, Any]] = []
    for item in rows:
        symbol = str(item.get("symbol") or "").strip().upper().replace(".", "-")
        price = parse_number(item.get("lastsale") or item.get("lastSale"))
        volume = parse_number(item.get("volume"))
        change_pct = parse_number(item.get("pctchange") or item.get("percentChange"))
        if not symbol or price is None or price <= 0 or volume is None or volume < 0:
            continue
        normalized.append(
            {
                "symbol": symbol,
                "nasdaq_name": str(item.get("name") or item.get("companyName") or ""),
                "last_price": float(price),
                "cumulative_volume": float(volume),
                "change_pct": float(change_pct) if change_pct is not None else None,
                "nasdaq_market_cap": parse_number(item.get("marketCap") or item.get("marketcap")),
                "country": str(item.get("country") or ""),
                "sector": str(item.get("sector") or ""),
                "industry": str(item.get("industry") or ""),
            }
        )
    frame = pd.DataFrame(normalized).drop_duplicates("symbol", keep="first")
    if len(frame) < 500:
        raise RuntimeError(f"Only {len(frame)} usable Nasdaq snapshot rows remained")
    meta = {
        "fetched_at_utc": now_utc.isoformat(),
        "http_date": response.headers.get("Date"),
        "source_url": NASDAQ_SCREENER_URL,
        "raw_row_count": len(rows),
        "usable_row_count": len(frame),
        "content_length": len(response.content),
    }
    return frame, meta


def latest_completed_session(now_et: datetime) -> tuple[date, bool]:
    calendar = xcals.get_calendar("XNYS")
    sessions = daily.session_dates(
        calendar,
        now_et.date() - timedelta(days=15),
        now_et.date() + timedelta(days=5),
    )
    today_is_session = now_et.date() in sessions
    completed = [item for item in sessions if item < now_et.date()]
    if today_is_session:
        close_stamp = calendar.session_close(pd.Timestamp(now_et.date()))
        if close_stamp.tzinfo is None:
            close_stamp = close_stamp.tz_localize("UTC")
        close_et = close_stamp.tz_convert("America/New_York").to_pydatetime()
        if now_et >= close_et:
            return now_et.date(), True
    if not completed:
        raise RuntimeError("Unable to resolve the latest completed XNYS session")
    return completed[-1], today_is_session


def build_float_reference(
    *,
    kline_path: Path,
    sec_facts: pd.DataFrame,
    reference_date: date,
    snapshot_date: date,
) -> pd.DataFrame:
    history_start = reference_date - timedelta(
        days=MAX_PUBLIC_FLOAT_AGE_DAYS
        + MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS
        + BAR_HISTORY_BUFFER_DAYS
    )
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='6GB'")
    connection.execute(
        f"CREATE VIEW kline_raw AS SELECT * FROM read_parquet('{daily.qpath(kline_path)}')"
    )
    connection.register("sec_float_facts_raw", sec_facts)

    connection.execute(
        """
        CREATE TEMP TABLE bars_calc AS
        WITH prepared AS (
            SELECT
                UPPER(REPLACE(CAST(symbol AS VARCHAR), '.', '-')) AS symbol,
                TRY_CAST(bar_time AS DATE) AS trade_date,
                TRY_CAST(close AS DOUBLE) AS adjusted_close,
                CASE
                    WHEN TRY_CAST(adj_factor_cum AS DOUBLE) > 0
                    THEN TRY_CAST(adj_factor_cum AS DOUBLE)
                    ELSE 1.0
                END AS adjustment_factor,
                CASE
                    WHEN TRY_CAST(splits AS DOUBLE) > 0
                    THEN TRY_CAST(splits AS DOUBLE)
                    ELSE 1.0
                END AS split_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        UPPER(REPLACE(CAST(symbol AS VARCHAR), '.', '-')),
                        TRY_CAST(bar_time AS DATE)
                    ORDER BY TRY_CAST(bar_time AS TIMESTAMP) DESC NULLS LAST
                ) AS row_rank
            FROM kline_raw
            WHERE UPPER(COALESCE(CAST(kline_t AS VARCHAR), '1D')) = '1D'
              AND TRY_CAST(bar_time AS DATE) BETWEEN ? AND ?
        ), dedup AS (
            SELECT * EXCLUDE (row_rank)
            FROM prepared
            WHERE row_rank = 1 AND adjusted_close > 0
        )
        SELECT
            *,
            adjusted_close / adjustment_factor AS raw_close,
            EXP(
                SUM(LN(split_ratio)) OVER (
                    PARTITION BY symbol
                    ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
            ) AS cumulative_split_factor
        FROM dedup
        """,
        [history_start, reference_date],
    )

    connection.execute(
        """
        CREATE TEMP TABLE priced_facts AS
        SELECT * EXCLUDE (price_rank)
        FROM (
            SELECT
                facts.*,
                bars.trade_date AS measurement_price_date,
                bars.raw_close AS measurement_raw_close,
                bars.cumulative_split_factor AS measurement_split_factor,
                facts.public_float_value_usd / bars.raw_close AS measurement_float_shares,
                ROW_NUMBER() OVER (
                    PARTITION BY facts.ticker, facts.measurement_date,
                                 facts.filed_date, facts.accession
                    ORDER BY bars.trade_date DESC
                ) AS price_rank
            FROM sec_float_facts_raw facts
            INNER JOIN bars_calc bars
                ON bars.symbol = facts.ticker
               AND bars.trade_date <= facts.measurement_date
               AND bars.trade_date >= facts.measurement_date - CAST(? AS INTEGER)
               AND bars.raw_close > 0
        )
        WHERE price_rank = 1 AND measurement_float_shares > 0
        """,
        [MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS],
    )

    reference = connection.execute(
        """
        WITH reference_bars AS (
            SELECT * EXCLUDE (reference_rank)
            FROM (
                SELECT
                    symbol,
                    trade_date AS reference_price_date,
                    cumulative_split_factor AS reference_split_factor,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol ORDER BY trade_date DESC
                    ) AS reference_rank
                FROM bars_calc
                WHERE trade_date <= ?
                  AND trade_date >= ? - CAST(10 AS INTEGER)
            )
            WHERE reference_rank = 1
        ), candidates AS (
            SELECT
                facts.*,
                ref.reference_price_date,
                facts.measurement_float_shares
                    * (ref.reference_split_factor / facts.measurement_split_factor)
                    AS inferred_float_shares,
                DATE_DIFF('day', facts.measurement_date, ?) AS public_float_age_days,
                ROW_NUMBER() OVER (
                    PARTITION BY facts.ticker
                    ORDER BY facts.measurement_date DESC,
                             facts.filed_date DESC,
                             facts.accession DESC
                ) AS fact_rank
            FROM priced_facts facts
            INNER JOIN reference_bars ref ON ref.symbol = facts.ticker
            WHERE facts.filed_date < ?
              AND facts.measurement_date <= ?
              AND DATE_DIFF('day', facts.measurement_date, ?) BETWEEN 0 AND ?
        )
        SELECT
            ticker AS symbol,
            sec_entity_name,
            inferred_float_shares,
            public_float_value_usd,
            measurement_date,
            filed_date,
            form,
            accession,
            sec_source_url,
            reference_price_date,
            public_float_age_days
        FROM candidates
        WHERE fact_rank = 1 AND inferred_float_shares > 0
        """,
        [
            reference_date,
            reference_date,
            snapshot_date,
            snapshot_date,
            snapshot_date,
            snapshot_date,
            MAX_PUBLIC_FLOAT_AGE_DAYS,
        ],
    ).df()
    for column in ("measurement_date", "filed_date", "reference_price_date"):
        reference[column] = pd.to_datetime(reference[column], errors="coerce").dt.date
    return reference


def compact_money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:.2f}M"
    if abs(value) >= 1e3:
        return f"${value / 1e3:.1f}K"
    return f"${value:.2f}"


def build_markdown(
    ranking: pd.DataFrame,
    *,
    now_bjt: datetime,
    now_et: datetime,
    reference_date: date,
    snapshot_meta: dict[str, Any],
    diagnostics: dict[str, Any],
) -> str:
    lines = [
        "| 排名 | 代码 | 公司 | 最新价 | 当前涨跌幅 | 活跃度 | 累计成交量 | 估算成交额 | 自由流通市值 | 跌幅≤-8% |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranking.itertuples(index=False):
        company = str(row.company_name).replace("|", "/")
        change = "—" if pd.isna(row.change_pct) else f"{row.change_pct:+.2f}%"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.activity_rank)),
                    f"**{row.symbol}**",
                    company,
                    f"${row.last_price:.2f}",
                    change,
                    f"{row.activity_ratio:.3f}x",
                    f"{int(row.cumulative_volume):,}",
                    compact_money(row.estimated_dollar_volume),
                    compact_money(row.current_float_market_cap),
                    "**通过**" if row.drop_filter_pass else "未通过",
                ]
            )
            + " |"
        )
    table = "\n".join(lines)
    passed = ranking[ranking["drop_filter_pass"]]
    passed_text = (
        "、".join(f"**{item}**" for item in passed["symbol"].tolist())
        if len(passed)
        else "无"
    )
    return f"""# 美股盘中/盘前活跃度前三｜最新公开快照

> 北京时间：**{now_bjt.strftime('%Y-%m-%d %H:%M:%S')}**  
> 美东时间：**{now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}**  
> 自由流通股拆分调整基准日：**{reference_date}**

## 当前前三

{table}

## 跌幅筛选

当前前三中，涨跌幅≤-8%的标的：{passed_text}。

## 口径

- 活跃度 = 当前累计成交金额 ÷ 当前自由流通市值。
- Nasdaq公开筛选器未提供全市场综合VWAP，因此成交金额以“最新价 × 累计成交量”估算；排序等价于“累计成交量 ÷ 推算自由流通股数”。
- 自由流通股数继续使用信号时点前已公开的SEC `EntityPublicFloat`，并仅按股票拆分机械调整。
- 当前涨跌幅来自Nasdaq公开筛选器；公开网页数据可能延迟，不能视为交易所直连实时行情。
- 北京时间16:30和21:00均属于美股盘前阶段；成交量较薄时排名会比开盘后更不稳定。

## 数据诊断

- Nasdaq原始行数：{snapshot_meta['raw_row_count']:,}
- Nasdaq可用行数：{snapshot_meta['usable_row_count']:,}
- SEC流通值参考数：{diagnostics['float_reference_count']:,}
- 成功匹配数：{diagnostics['matched_count']:,}
- 参与排名数：{diagnostics['rankable_count']:,}
- Nasdaq响应Date：{snapshot_meta.get('http_date') or '—'}

> 本任务只生成研究排名，不回测，也不连接券商下单。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/us_intraday_picker/sec")
    parser.add_argument("--work-dir", default=".work/us_intraday_picker")
    parser.add_argument("--output-dir", default="intraday_picks_top3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-sec", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    now_bjt = now_utc.astimezone(BJT)
    now_et = now_utc.astimezone(ET)
    reference_date, today_is_session = latest_completed_session(now_et)
    if not today_is_session:
        result = {
            "status": "skipped_non_session",
            "generated_at_bjt": now_bjt,
            "generated_at_et": now_et,
        }
        append_output(status=result["status"], notify="false")
        print(json.dumps(result, default=scalar, ensure_ascii=False, indent=2))
        return 0

    cache_dir = Path(args.cache_dir)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    info_path, sec_facts, reference_manifest = daily.ensure_weekly_reference_cache(
        cache_dir,
        reference_date,
        force_refresh=args.refresh_sec,
    )
    del info_path  # The strict SEC ticker mapping was already built in the cache.
    kline_path, kline_size = daily.download_fresh_kline(work_dir)
    snapshot, snapshot_meta = download_nasdaq_snapshot(now_utc)
    float_reference = build_float_reference(
        kline_path=kline_path,
        sec_facts=sec_facts,
        reference_date=reference_date,
        snapshot_date=now_et.date(),
    )

    merged = snapshot.merge(float_reference, on="symbol", how="inner")
    merged["estimated_dollar_volume"] = (
        merged["last_price"] * merged["cumulative_volume"]
    )
    merged["current_float_market_cap"] = (
        merged["last_price"] * merged["inferred_float_shares"]
    )
    merged["activity_ratio"] = (
        merged["estimated_dollar_volume"] / merged["current_float_market_cap"]
    )
    merged["company_name"] = merged["nasdaq_name"].where(
        merged["nasdaq_name"].astype(str).str.len() > 0,
        merged["sec_entity_name"],
    )
    rankable = merged[
        (merged["last_price"] >= MIN_PRICE)
        & (merged["current_float_market_cap"] >= MIN_FLOAT_MARKET_CAP)
        & (merged["cumulative_volume"] > 0)
        & merged["activity_ratio"].notna()
        & (merged["activity_ratio"] > 0)
    ].copy()
    rankable = rankable.sort_values(
        ["activity_ratio", "estimated_dollar_volume", "symbol"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if len(rankable) < RANK_LIMIT:
        raise RuntimeError(f"Only {len(rankable)} stocks were rankable")
    rankable["activity_rank"] = range(1, len(rankable) + 1)
    ranking = rankable.head(RANK_LIMIT).copy()
    ranking["drop_filter_pass"] = ranking["change_pct"].fillna(math.inf) <= DROP_THRESHOLD_PCT

    diagnostics = {
        "float_reference_count": len(float_reference),
        "matched_count": len(merged),
        "rankable_count": len(rankable),
        "kline_size_bytes": kline_size,
        "today_is_xnys_session": today_is_session,
        "reference_manifest": reference_manifest,
    }
    markdown = build_markdown(
        ranking,
        now_bjt=now_bjt,
        now_et=now_et,
        reference_date=reference_date,
        snapshot_meta=snapshot_meta,
        diagnostics=diagnostics,
    )

    run_key = now_bjt.strftime("%Y-%m-%d_%H%M%S_BJT")
    result_dir = output_dir / run_key
    result_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = result_dir / "ranking.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    (output_dir / "latest.md").write_text(markdown, encoding="utf-8")
    ranking.to_csv(result_dir / "top3.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "latest.csv", index=False, encoding="utf-8-sig")

    tickers = ranking["symbol"].tolist()
    issue_title = (
        f"[US INTRADAY TOP3] {now_bjt.strftime('%Y-%m-%d %H:%M BJT')} | "
        + ", ".join(tickers)
    )
    payload = {
        "status": "ready",
        "generated_at_utc": now_utc,
        "generated_at_bjt": now_bjt,
        "generated_at_et": now_et,
        "reference_date": reference_date,
        "issue_title": issue_title,
        "ranking": ranking.to_dict("records"),
        "snapshot_meta": snapshot_meta,
        "diagnostics": diagnostics,
        "methodology": {
            "activity": "estimated cumulative dollar volume / current inferred free-float market cap",
            "estimated_dollar_volume": "last sale * cumulative share volume",
            "drop_threshold_pct": DROP_THRESHOLD_PCT,
            "rank_limit": RANK_LIMIT,
        },
    }
    write_json(result_dir / "ranking.json", payload)
    write_json(output_dir / "latest.json", payload)
    append_output(
        status="ready",
        notify="true",
        issue_title=issue_title,
        markdown_path=str(markdown_path),
        run_key=run_key,
        top3=",".join(tickers),
    )
    append_summary(markdown)
    print(json.dumps(payload, default=scalar, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        append_output(status="failed", notify="false")
        print(f"INTRADAY TOP3 FAILED: {error}", file=sys.stderr, flush=True)
        raise
