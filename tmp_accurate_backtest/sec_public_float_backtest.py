#!/usr/bin/env python3
"""Six-month U.S. activity backtest using SEC point-in-time public float.

Strict data policy
------------------
* Free-float market value comes only from SEC XBRL ``dei:EntityPublicFloat``.
* A filing is usable only on sessions strictly after its SEC filing date because
  companyfacts provides a date but not a reliable public release timestamp.
* The latest disclosed fact known on each signal date is carried forward for at
  most 550 calendar days from its measurement date.
* Free-float shares are inferred at the SEC measurement date using unadjusted
  market price and then adjusted only for disclosed stock splits.
* Signal-day ranking is completed before entry/exit availability is checked.
* Same-day exits do not require a following-session bar.
* Companies with multiple SEC tickers are excluded because EntityPublicFloat is
  company-wide and cannot be allocated reliably across share classes.

This is the strictest reproducible public-data approximation.  SEC public float is
periodic, not a daily cap-table feed, so the workbook reports coverage and age for
every selected fact instead of claiming unobservable daily perfection.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import orjson
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ALPHA_SOURCES = {
    "info": "https://huggingface.co/datasets/AlphaDojo/dojo_stock_info/resolve/main/data.parquet?download=true",
    "kline": "https://huggingface.co/datasets/AlphaDojo/dojo_stock_kline/resolve/main/data.parquet?download=true",
}
SEC_SOURCES = {
    "companyfacts": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "submissions": "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip",
}
SEC_USER_AGENT = "OpenAI point-in-time market research support@openai.com"
LOOKBACK_CALENDAR_DAYS = 183
BAR_HISTORY_START = date(2025, 1, 2)
BIG_DROP_THRESHOLDS = (-5.0, -8.0, -10.0, -15.0)
PRIMARY_BIG_DROP_PCT = -8.0
ONE_WAY_COST = 0.0025
MAX_PUBLIC_FLOAT_AGE_DAYS = 550
MAX_PRICE_LOOKBACK_DAYS = 10
MIN_DAILY_MARKET_BARS = 1000
TRADABLE_MIN_RAW_CLOSE = 1.0
TRADABLE_MIN_DOLLAR_VOLUME = 1_000_000.0
TRADABLE_MIN_FLOAT_MCAP = 20_000_000.0
AVG_PRICE_TOLERANCE = 0.20
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260822
ALLOWED_PUBLIC_FLOAT_FORMS = {
    "10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A",
}
CIK_NAME_RE = re.compile(r"CIK(\d{10})\.json$")


def http_session(sec: bool = False) -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": SEC_USER_AGENT if sec else "sec-public-float-backtest/2026-08-22"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download(session: requests.Session, url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000:
        return destination.stat().st_size
    temp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with session.get(url, stream=True, timeout=(30, 2400), allow_redirects=True) as response:
                response.raise_for_status()
                with temp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temp.stat().st_size <= 1_000:
                raise RuntimeError(f"download unexpectedly small: {temp.stat().st_size} bytes")
            temp.replace(destination)
            return destination.stat().st_size
        except Exception:
            if temp.exists():
                temp.unlink()
            if attempt == 3:
                raise
            time.sleep(attempt * 8)
    raise AssertionError("unreachable")


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def scalar(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def finite_series(values: Iterable[Any]) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def max_drawdown(returns: pd.Series) -> float:
    values = finite_series(returns)
    if values.empty:
        return np.nan
    equity = (1.0 + values).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def bootstrap_mean_ci(returns: pd.Series) -> tuple[float, float]:
    values = finite_series(returns).to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED + len(values))
    means = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    chunk = 500
    for start in range(0, BOOTSTRAP_DRAWS, chunk):
        stop = min(start + chunk, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def metric_block(values: pd.Series) -> dict[str, Any]:
    clean = finite_series(values)
    wins = clean[clean > 0]
    losses = clean[clean < 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) and avg_loss != 0 else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if len(wins) and len(losses) and losses.sum() != 0 else np.nan
    break_even = abs(avg_loss) / (avg_win + abs(avg_loss)) if len(wins) and len(losses) and avg_win + abs(avg_loss) > 0 else np.nan
    ci_low, ci_high = bootstrap_mean_ci(clean)
    return {
        "observation_count": int(len(clean)),
        "win_rate": float((clean > 0).mean()) if len(clean) else np.nan,
        "avg_return": float(clean.mean()) if len(clean) else np.nan,
        "median_return": float(clean.median()) if len(clean) else np.nan,
        "avg_win": scalar(avg_win),
        "avg_loss": scalar(avg_loss),
        "payoff_ratio": scalar(payoff),
        "profit_factor": scalar(profit_factor),
        "break_even_win_rate": scalar(break_even),
        "best_return": float(clean.max()) if len(clean) else np.nan,
        "worst_return": float(clean.min()) if len(clean) else np.nan,
        "mean_ci_95_low": scalar(ci_low),
        "mean_ci_95_high": scalar(ci_high),
    }


def normalize_ticker(value: str) -> str:
    ticker = value.strip().upper()
    return ticker.replace(".", "-")


def parse_submission_tickers(path: Path, alpha_tickers: set[str]) -> tuple[dict[int, str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if CIK_NAME_RE.search(Path(name).name)]
        total = len(names)
        for index, name in enumerate(names, start=1):
            match = CIK_NAME_RE.search(Path(name).name)
            if not match:
                continue
            try:
                payload = orjson.loads(archive.read(name))
            except Exception:
                continue
            cik = int(payload.get("cik") or match.group(1))
            tickers = [normalize_ticker(str(item)) for item in (payload.get("tickers") or []) if str(item).strip()]
            exchanges = [str(item or "") for item in (payload.get("exchanges") or [])]
            unique = sorted(set(tickers))
            for position, ticker in enumerate(tickers):
                rows.append({
                    "cik": cik,
                    "sec_entity_name": payload.get("name"),
                    "ticker": ticker,
                    "sec_exchange": exchanges[position] if position < len(exchanges) else "",
                    "ticker_count_for_cik": len(unique),
                    "in_alpha_master": ticker in alpha_tickers,
                })
            if index % 2500 == 0:
                print(f"parsed SEC submissions {index}/{total}", flush=True)
    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise RuntimeError("SEC submissions archive yielded no ticker mapping")
    strict = mapping[(mapping["ticker_count_for_cik"] == 1) & mapping["in_alpha_master"]].copy()
    strict = strict.drop_duplicates(["cik", "ticker"])
    cik_to_ticker = dict(zip(strict["cik"].astype(int), strict["ticker"]))
    return cik_to_ticker, mapping


def filing_index_url(cik: int, accession: str) -> str:
    no_dashes = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{no_dashes}/{accession}-index.html"


def parse_public_float_facts(path: Path, cik_to_ticker: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_ciks = set(cik_to_ticker)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        total = len(names)
        matched = 0
        for index, name in enumerate(names, start=1):
            match = CIK_NAME_RE.search(Path(name).name)
            if not match:
                continue
            cik = int(match.group(1))
            if cik not in target_ciks:
                continue
            matched += 1
            try:
                payload = orjson.loads(archive.read(name))
            except Exception:
                continue
            fact = (((payload.get("facts") or {}).get("dei") or {}).get("EntityPublicFloat") or {})
            units = fact.get("units") or {}
            observations = units.get("USD") or []
            for item in observations:
                form = str(item.get("form") or "")
                accession = str(item.get("accn") or "")
                try:
                    value = float(item.get("val"))
                    end = date.fromisoformat(str(item.get("end")))
                    filed = date.fromisoformat(str(item.get("filed")))
                except Exception:
                    continue
                if value <= 0 or form not in ALLOWED_PUBLIC_FLOAT_FORMS:
                    continue
                if end < BAR_HISTORY_START or filed < BAR_HISTORY_START:
                    continue
                rows.append({
                    "cik": cik,
                    "ticker": cik_to_ticker[cik],
                    "sec_entity_name": payload.get("entityName"),
                    "taxonomy_label": fact.get("label"),
                    "public_float_value_usd": value,
                    "measurement_date": end,
                    "filed_date": filed,
                    "form": form,
                    "accession": accession,
                    "fiscal_year": item.get("fy"),
                    "fiscal_period": item.get("fp"),
                    "frame": item.get("frame"),
                    "sec_source_url": filing_index_url(cik, accession) if accession else "",
                })
            if matched % 1000 == 0:
                print(f"parsed companyfacts for {matched} targeted CIKs ({index}/{total} archive entries)", flush=True)
    facts = pd.DataFrame(rows)
    if facts.empty:
        raise RuntimeError("no SEC EntityPublicFloat facts matched strict ticker mappings")
    facts = facts.sort_values(
        ["ticker", "measurement_date", "filed_date", "accession", "public_float_value_usd"],
        kind="mergesort",
    )
    facts = facts.drop_duplicates(
        ["ticker", "measurement_date", "filed_date", "accession"], keep="last"
    )
    return facts.reset_index(drop=True)


def summarize_returns(
    trades: pd.DataFrame,
    scenario: str,
    threshold: float,
    exit_key: str,
    return_basis: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    selected = trades[(trades["scenario"] == scenario) & (trades["big_drop_threshold_pct"] == threshold)].copy()
    column = f"{return_basis}_{exit_key}"
    available = selected[selected[column].notna()].copy()
    daily = (
        available.groupby("entry_date", as_index=False)[column]
        .mean()
        .rename(columns={column: "portfolio_return"})
        .sort_values("entry_date")
    )
    daily["scenario"] = scenario
    daily["big_drop_threshold_pct"] = threshold
    daily["exit_key"] = exit_key
    daily["return_basis"] = return_basis
    daily["portfolio_win"] = daily["portfolio_return"] > 0
    daily["diagnostic_cumulative"] = (1.0 + daily["portfolio_return"]).cumprod() - 1.0
    common = {
        "scenario": scenario,
        "big_drop_threshold_pct": threshold,
        "exit_key": exit_key,
        "return_basis": return_basis,
        "selected_signal_count": int(len(selected)),
        "selected_signal_days": int(selected["signal_date"].nunique()) if len(selected) else 0,
        "entry_fill_count": int(selected["entry_open"].notna().sum()) if len(selected) else 0,
        "return_available_count": int(len(available)),
        "return_coverage_rate": float(len(available) / len(selected)) if len(selected) else np.nan,
    }
    trade_summary = {
        **common,
        "statistics_level": "individual_trades",
        **metric_block(available[column] if len(available) else pd.Series(dtype=float)),
        "diagnostic_compounded": np.nan,
        "diagnostic_max_drawdown": np.nan,
    }
    daily_summary = {
        **common,
        "statistics_level": "daily_equal_weight",
        **metric_block(daily["portfolio_return"] if len(daily) else pd.Series(dtype=float)),
        "diagnostic_compounded": float((1.0 + daily["portfolio_return"]).prod() - 1.0) if len(daily) else np.nan,
        "diagnostic_max_drawdown": max_drawdown(daily["portfolio_return"]) if len(daily) else np.nan,
    }
    return trade_summary, daily_summary, daily


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".tmp_sec_float_data")
    parser.add_argument("--out-dir", default="sec_float_backtest_output")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha_session = http_session(False)
    sec_session = http_session(True)
    files: dict[str, Path] = {}
    manifest: dict[str, Any] = {}
    for name, url in ALPHA_SOURCES.items():
        path = data_dir / f"{name}.parquet"
        size = download(alpha_session, url, path)
        files[name] = path
        manifest[name] = {"url": url, "size_bytes": size, "source_class": "market_data"}
        print(f"downloaded {name}: {size / 1_000_000:.1f} MB", flush=True)
    for name, url in SEC_SOURCES.items():
        path = data_dir / f"{name}.zip"
        size = download(sec_session, url, path)
        files[name] = path
        manifest[name] = {"url": url, "size_bytes": size, "source_class": "official_sec"}
        print(f"downloaded SEC {name}: {size / 1_000_000:.1f} MB", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute(f"CREATE VIEW info_raw AS SELECT * FROM read_parquet('{qpath(files['info'])}')")
    con.execute(f"CREATE VIEW kline_raw AS SELECT * FROM read_parquet('{qpath(files['kline'])}')")

    con.execute(r"""
        CREATE TEMP TABLE info_intervals AS
        SELECT
            UPPER(REPLACE(NULLIF(CAST(ticker AS VARCHAR), ''), '.', '-')) AS ticker,
            COALESCE(CAST(short_name AS VARCHAR), CAST(long_name AS VARCHAR), CAST(ticker AS VARCHAR)) AS company_name,
            COALESCE(CAST(full_exchange_name AS VARCHAR), '') AS exchange,
            TRY_CAST(first_traded_date AS DATE) AS first_traded_date,
            TRY_CAST(delisted_at AS DATE) AS delisted_at,
            COALESCE(TRY_CAST(is_delisted AS BOOLEAN), FALSE) AS is_delisted,
            COALESCE(CAST(country AS VARCHAR), '') AS country
        FROM info_raw
        WHERE LOWER(COALESCE(CAST(market AS VARCHAR), '')) = 'us'
          AND UPPER(COALESCE(CAST(quote_type AS VARCHAR), '')) = 'EQUITY'
          AND NULLIF(CAST(ticker AS VARCHAR), '') IS NOT NULL
          AND REGEXP_MATCHES(
              UPPER(COALESCE(CAST(full_exchange_name AS VARCHAR), '')),
              'NASDAQ|NYSE|AMEX|ARCA|CBOE|BATS'
          )
          AND NOT REGEXP_MATCHES(
              UPPER(COALESCE(CAST(short_name AS VARCHAR), '') || ' ' ||
                    COALESCE(CAST(long_name AS VARCHAR), '') || ' ' ||
                    COALESCE(CAST(type_disp AS VARCHAR), '')),
              '(^|[^A-Z])(ETF|ETN|WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PREF|CLOSED.END FUND|MUTUAL FUND)([^A-Z]|$)'
          )
    """)
    alpha_tickers = set(row[0] for row in con.execute("SELECT DISTINCT ticker FROM info_intervals").fetchall())
    cik_to_ticker, sec_mapping = parse_submission_tickers(files["submissions"], alpha_tickers)
    print(f"strict single-ticker SEC mappings matched Alpha master: {len(cik_to_ticker)}", flush=True)
    sec_facts = parse_public_float_facts(files["companyfacts"], cik_to_ticker)
    print(f"SEC public-float fact observations: {len(sec_facts)}", flush=True)

    con.register("sec_float_facts_raw", sec_facts)
    con.register("sec_mapping_raw", sec_mapping)

    latest_date = con.execute("""
        SELECT MAX(TRY_CAST(k.bar_time AS DATE))
        FROM kline_raw k
        WHERE UPPER(COALESCE(CAST(k.kline_t AS VARCHAR), '1D')) = '1D'
          AND TRY_CAST(k.close AS DOUBLE) > 0
          AND EXISTS (
              SELECT 1 FROM info_intervals i
              WHERE i.ticker = UPPER(REPLACE(CAST(k.symbol AS VARCHAR), '.', '-'))
          )
    """).fetchone()[0]
    if latest_date is None:
        raise RuntimeError("no U.S. market date found")
    if isinstance(latest_date, datetime):
        latest_date = latest_date.date()
    analysis_start = latest_date - timedelta(days=LOOKBACK_CALENDAR_DAYS)

    con.execute("""
        CREATE TEMP TABLE bars_source AS
        WITH prepared AS (
            SELECT
                UPPER(REPLACE(CAST(k.symbol AS VARCHAR), '.', '-')) AS symbol,
                TRY_CAST(k.bar_time AS DATE) AS trade_date,
                TRY_CAST(k.open AS DOUBLE) AS adjusted_open,
                TRY_CAST(k.high AS DOUBLE) AS adjusted_high,
                TRY_CAST(k.low AS DOUBLE) AS adjusted_low,
                TRY_CAST(k.close AS DOUBLE) AS adjusted_close,
                TRY_CAST(k.vol AS DOUBLE) AS volume,
                TRY_CAST(k.amount AS DOUBLE) AS amount,
                TRY_CAST(k.change_p AS DOUBLE) AS vendor_change_pct,
                CASE
                    WHEN TRY_CAST(k.adj_factor_cum AS DOUBLE) > 0 THEN TRY_CAST(k.adj_factor_cum AS DOUBLE)
                    ELSE 1.0
                END AS adjustment_factor,
                CASE
                    WHEN TRY_CAST(k.splits AS DOUBLE) > 0 THEN TRY_CAST(k.splits AS DOUBLE)
                    ELSE 1.0
                END AS split_ratio,
                COALESCE(TRY_CAST(k.dividends AS DOUBLE), 0.0) AS dividends,
                ROW_NUMBER() OVER (
                    PARTITION BY UPPER(REPLACE(CAST(k.symbol AS VARCHAR), '.', '-')), TRY_CAST(k.bar_time AS DATE)
                    ORDER BY TRY_CAST(k.bar_time AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM kline_raw k
            WHERE UPPER(COALESCE(CAST(k.kline_t AS VARCHAR), '1D')) = '1D'
              AND TRY_CAST(k.bar_time AS DATE) BETWEEN ? AND ?
        )
        SELECT * EXCLUDE (rn)
        FROM prepared
        WHERE rn = 1
          AND trade_date IS NOT NULL
          AND adjusted_open > 0 AND adjusted_high > 0 AND adjusted_low > 0 AND adjusted_close > 0
          AND volume >= 0
    """, [BAR_HISTORY_START, latest_date])

    con.execute("""
        CREATE TEMP TABLE bars_listed AS
        SELECT * EXCLUDE (interval_rank)
        FROM (
            SELECT
                b.*,
                i.company_name,
                i.exchange,
                i.first_traded_date,
                i.delisted_at,
                i.is_delisted,
                i.country,
                ROW_NUMBER() OVER (
                    PARTITION BY b.symbol, b.trade_date
                    ORDER BY i.first_traded_date DESC NULLS LAST, i.delisted_at DESC NULLS LAST
                ) AS interval_rank
            FROM bars_source b
            INNER JOIN info_intervals i
                ON i.ticker = b.symbol
               AND (i.first_traded_date IS NULL OR b.trade_date >= i.first_traded_date)
               AND (i.delisted_at IS NULL OR b.trade_date <= i.delisted_at)
        )
        WHERE interval_rank = 1
    """)

    con.execute("""
        CREATE TEMP TABLE bars_calc AS
        WITH raw AS (
            SELECT
                *,
                adjusted_open / adjustment_factor AS raw_open,
                adjusted_high / adjustment_factor AS raw_high,
                adjusted_low / adjustment_factor AS raw_low,
                adjusted_close / adjustment_factor AS raw_close,
                CASE WHEN amount > 0 AND volume > 0 THEN amount / volume ELSE NULL END AS avg_trade_price,
                EXP(SUM(LN(split_ratio)) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )) AS cumulative_split_factor,
                LAG(adjusted_close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prior_adjusted_close
            FROM bars_listed
        )
        SELECT
            *,
            CASE
                WHEN prior_adjusted_close > 0 THEN 100.0 * (adjusted_close / prior_adjusted_close - 1.0)
                ELSE NULL
            END AS drop_pct,
            CASE
                WHEN avg_trade_price > 0 AND raw_low > 0 AND raw_high > 0
                 AND avg_trade_price BETWEEN raw_low * ? AND raw_high * ?
                THEN TRUE ELSE FALSE
            END AS avg_price_inside_raw_ohlc,
            CASE
                WHEN prior_adjusted_close > 0 AND vendor_change_pct IS NOT NULL
                THEN ABS(100.0 * (adjusted_close / prior_adjusted_close - 1.0) - vendor_change_pct)
                ELSE NULL
            END AS change_pct_abs_error
        FROM raw
    """, [1.0 - AVG_PRICE_TOLERANCE, 1.0 + AVG_PRICE_TOLERANCE])

    con.execute("""
        CREATE TEMP TABLE market_dates AS
        SELECT trade_date, COUNT(*) AS listed_bar_count
        FROM bars_calc
        GROUP BY trade_date
        HAVING COUNT(*) >= ?
        ORDER BY trade_date
    """, [MIN_DAILY_MARKET_BARS])
    market_date_count = int(con.execute("SELECT COUNT(*) FROM market_dates").fetchone()[0])
    if market_date_count < 80:
        raise RuntimeError(f"too few reliable market dates: {market_date_count}")

    con.execute("""
        CREATE TEMP TABLE calendar_map AS
        SELECT
            trade_date AS signal_date,
            LEAD(trade_date, 1) OVER (ORDER BY trade_date) AS entry_date,
            LEAD(trade_date, 2) OVER (ORDER BY trade_date) AS next_date
        FROM market_dates
    """)

    # Price each SEC public-float observation using the latest prior raw close.
    con.execute("""
        CREATE TEMP TABLE sec_float_facts_priced AS
        SELECT * EXCLUDE (price_rank)
        FROM (
            SELECT
                f.*,
                b.trade_date AS measurement_price_date,
                b.raw_close AS measurement_raw_close,
                b.cumulative_split_factor AS measurement_split_factor,
                f.public_float_value_usd / b.raw_close AS measurement_float_shares,
                DATE_DIFF('day', b.trade_date, f.measurement_date) AS measurement_price_lag_days,
                ROW_NUMBER() OVER (
                    PARTITION BY f.ticker, f.measurement_date, f.filed_date, f.accession
                    ORDER BY b.trade_date DESC
                ) AS price_rank
            FROM sec_float_facts_raw f
            INNER JOIN bars_calc b
                ON b.symbol = f.ticker
               AND b.trade_date <= f.measurement_date
               AND b.trade_date >= f.measurement_date - INTERVAL ? DAY
               AND b.raw_close > 0
        )
        WHERE price_rank = 1
          AND measurement_float_shares > 0
    """, [MAX_PRICE_LOOKBACK_DAYS])

    con.execute("""
        CREATE TEMP TABLE signal_base AS
        SELECT b.*
        FROM bars_calc b
        INNER JOIN market_dates m USING (trade_date)
        WHERE b.trade_date >= ?
          AND b.trade_date <= ?
          AND b.amount > 0
          AND b.volume > 0
          AND b.raw_close > 0
          AND b.avg_price_inside_raw_ohlc
          AND b.drop_pct IS NOT NULL
    """, [analysis_start, latest_date])

    # Match only facts that were publicly filed before the signal date.  We use
    # strict '<', not '<=', because companyfacts does not expose a filing time.
    con.execute("""
        CREATE TEMP TABLE covered_signal_rows AS
        SELECT * EXCLUDE (fact_rank)
        FROM (
            SELECT
                b.*,
                f.cik,
                f.sec_entity_name,
                f.public_float_value_usd,
                f.measurement_date,
                f.measurement_price_date,
                f.measurement_raw_close,
                f.measurement_split_factor,
                f.measurement_float_shares,
                f.measurement_price_lag_days,
                f.filed_date,
                f.form,
                f.accession,
                f.sec_source_url,
                DATE_DIFF('day', f.measurement_date, b.trade_date) AS public_float_age_days,
                DATE_DIFF('day', f.filed_date, b.trade_date) AS filing_known_days,
                f.measurement_float_shares
                  * (b.cumulative_split_factor / f.measurement_split_factor) AS signal_float_shares,
                b.raw_close * f.measurement_float_shares
                  * (b.cumulative_split_factor / f.measurement_split_factor) AS signal_float_market_cap,
                b.amount / (
                    b.raw_close * f.measurement_float_shares
                    * (b.cumulative_split_factor / f.measurement_split_factor)
                ) AS activity_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY b.symbol, b.trade_date
                    ORDER BY f.measurement_date DESC, f.filed_date DESC, f.accession DESC
                ) AS fact_rank
            FROM signal_base b
            INNER JOIN sec_float_facts_priced f
                ON f.ticker = b.symbol
               AND f.filed_date < b.trade_date
               AND f.measurement_date <= b.trade_date
               AND DATE_DIFF('day', f.measurement_date, b.trade_date) BETWEEN 0 AND ?
        )
        WHERE fact_rank = 1
          AND signal_float_shares > 0
          AND signal_float_market_cap > 0
          AND activity_ratio > 0
    """, [MAX_PUBLIC_FLOAT_AGE_DAYS])

    coverage_by_date = con.execute("""
        WITH all_rows AS (
            SELECT trade_date, COUNT(*) AS quality_eligible_rows
            FROM signal_base
            GROUP BY trade_date
        ), covered AS (
            SELECT trade_date, COUNT(*) AS sec_float_covered_rows
            FROM covered_signal_rows
            GROUP BY trade_date
        ), tradable AS (
            SELECT trade_date, COUNT(*) AS sec_float_tradable_rows
            FROM covered_signal_rows
            WHERE raw_close >= ?
              AND amount >= ?
              AND signal_float_market_cap >= ?
            GROUP BY trade_date
        )
        SELECT
            a.trade_date,
            a.quality_eligible_rows,
            COALESCE(c.sec_float_covered_rows, 0) AS sec_float_covered_rows,
            COALESCE(t.sec_float_tradable_rows, 0) AS sec_float_tradable_rows,
            COALESCE(c.sec_float_covered_rows, 0) * 1.0 / a.quality_eligible_rows AS coverage_rate,
            COALESCE(t.sec_float_tradable_rows, 0) * 1.0 / a.quality_eligible_rows AS tradable_coverage_rate
        FROM all_rows a
        LEFT JOIN covered c USING (trade_date)
        LEFT JOIN tradable t USING (trade_date)
        ORDER BY a.trade_date
    """, [TRADABLE_MIN_RAW_CLOSE, TRADABLE_MIN_DOLLAR_VOLUME, TRADABLE_MIN_FLOAT_MCAP]).df()

    covered = con.execute("SELECT * FROM covered_signal_rows").df()
    if covered.empty:
        raise RuntimeError("SEC public-float coverage produced no signal rows")

    date_cols = ["trade_date", "measurement_date", "measurement_price_date", "filed_date", "first_traded_date", "delisted_at"]
    for column in date_cols:
        if column in covered:
            covered[column] = pd.to_datetime(covered[column], errors="coerce").dt.date
    coverage_by_date["trade_date"] = pd.to_datetime(coverage_by_date["trade_date"]).dt.date

    scenario_frames: list[pd.DataFrame] = []
    for scenario in ("SEC披露覆盖", "SEC披露覆盖｜实盘过滤"):
        frame = covered.copy()
        if scenario.endswith("实盘过滤"):
            frame = frame[
                (frame["raw_close"] >= TRADABLE_MIN_RAW_CLOSE)
                & (frame["amount"] >= TRADABLE_MIN_DOLLAR_VOLUME)
                & (frame["signal_float_market_cap"] >= TRADABLE_MIN_FLOAT_MCAP)
            ].copy()
        frame = frame.sort_values(
            ["trade_date", "activity_ratio", "amount", "symbol"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        frame["activity_rank"] = frame.groupby("trade_date").cumcount() + 1
        frame = frame[frame["activity_rank"] <= 3].copy()
        frame["scenario"] = scenario
        frame = frame.rename(columns={"trade_date": "signal_date"})
        scenario_frames.append(frame)
    daily_top3 = pd.concat(scenario_frames, ignore_index=True)

    selected_frames: list[pd.DataFrame] = []
    for threshold in BIG_DROP_THRESHOLDS:
        chosen = daily_top3[daily_top3["drop_pct"] <= threshold].copy()
        chosen["big_drop_threshold_pct"] = threshold
        selected_frames.append(chosen)
    selected = pd.concat(selected_frames, ignore_index=True)

    calendar = con.execute("SELECT * FROM calendar_map").df()
    prices = con.execute("""
        SELECT symbol, trade_date, adjusted_open, adjusted_close
        FROM bars_calc
        INNER JOIN market_dates USING (trade_date)
        WHERE adjusted_open > 0 AND adjusted_close > 0
    """).df()
    for column in ["signal_date", "entry_date", "next_date"]:
        calendar[column] = pd.to_datetime(calendar[column]).dt.date
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date

    selected = selected.merge(calendar, on="signal_date", how="left", validate="many_to_one")
    entry_prices = prices.rename(
        columns={"trade_date": "entry_date", "adjusted_open": "entry_open", "adjusted_close": "entry_close"}
    )
    next_prices = prices.rename(
        columns={"trade_date": "next_date", "adjusted_open": "next_open", "adjusted_close": "next_close"}
    )
    selected = selected.merge(entry_prices, on=["symbol", "entry_date"], how="left", validate="many_to_one")
    selected = selected.merge(next_prices, on=["symbol", "next_date"], how="left", validate="many_to_one")

    selected["entry_filled"] = selected["entry_open"].notna() & (selected["entry_open"] > 0)
    selected["t_close_available"] = selected["entry_filled"] & selected["entry_close"].notna() & (selected["entry_close"] > 0)
    selected["t1_open_available"] = selected["entry_filled"] & selected["next_open"].notna() & (selected["next_open"] > 0)
    selected["t1_close_available"] = selected["entry_filled"] & selected["next_close"].notna() & (selected["next_close"] > 0)
    selected["gross_t_close"] = np.where(
        selected["t_close_available"], selected["entry_close"] / selected["entry_open"] - 1.0, np.nan
    )
    selected["gross_t1_open"] = np.where(
        selected["t1_open_available"], selected["next_open"] / selected["entry_open"] - 1.0, np.nan
    )
    selected["gross_t1_close"] = np.where(
        selected["t1_close_available"], selected["next_close"] / selected["entry_open"] - 1.0, np.nan
    )
    cost_factor = (1.0 - ONE_WAY_COST) / (1.0 + ONE_WAY_COST)
    for suffix in ("t_close", "t1_open", "t1_close"):
        selected[f"net_{suffix}"] = np.where(
            selected[f"gross_{suffix}"].notna(),
            (1.0 + selected[f"gross_{suffix}"]) * cost_factor - 1.0,
            np.nan,
        )
        selected[f"stress_net_{suffix}"] = np.where(
            selected["entry_filled"],
            np.where(selected[f"net_{suffix}"].notna(), selected[f"net_{suffix}"], -1.0),
            np.nan,
        )

    exit_labels = {"t_close": "当日收盘", "t1_open": "次日开盘", "t1_close": "次日收盘"}
    summary_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    for scenario in ("SEC披露覆盖", "SEC披露覆盖｜实盘过滤"):
        for threshold in BIG_DROP_THRESHOLDS:
            for exit_key, exit_label in exit_labels.items():
                for basis in ("gross", "net", "stress_net"):
                    trade_summary, daily_summary, daily = summarize_returns(
                        selected, scenario, threshold, exit_key, basis
                    )
                    trade_summary["exit_rule"] = exit_label
                    daily_summary["exit_rule"] = exit_label
                    summary_rows.extend([trade_summary, daily_summary])
                    if len(daily):
                        daily["exit_rule"] = exit_label
                        daily_frames.append(daily)
    summary = pd.DataFrame(summary_rows)
    daily_portfolios = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()

    primary_trades = selected[selected["big_drop_threshold_pct"] == PRIMARY_BIG_DROP_PCT].copy()
    top_symbols = (
        primary_trades.groupby(["scenario", "symbol", "company_name"], as_index=False)
        .agg(
            selection_count=("symbol", "size"),
            avg_activity_ratio=("activity_ratio", "mean"),
            avg_drop_pct=("drop_pct", "mean"),
            median_float_market_cap=("signal_float_market_cap", "median"),
            median_public_float_age_days=("public_float_age_days", "median"),
        )
        .sort_values(["scenario", "selection_count", "avg_activity_ratio"], ascending=[True, False, False])
    )

    # Data-quality and reproducibility audit.
    audit_rows = [
        {"metric": "latest_market_date", "value": latest_date.isoformat()},
        {"metric": "analysis_start_calendar_date", "value": analysis_start.isoformat()},
        {"metric": "market_date_count", "value": market_date_count},
        {"metric": "alpha_historical_ticker_count", "value": len(alpha_tickers)},
        {"metric": "sec_submission_mapping_rows", "value": int(len(sec_mapping))},
        {"metric": "strict_single_ticker_sec_cik_count", "value": len(cik_to_ticker)},
        {"metric": "sec_public_float_observations_raw", "value": int(len(sec_facts))},
        {"metric": "sec_public_float_observations_priced", "value": int(con.execute("SELECT COUNT(*) FROM sec_float_facts_priced").fetchone()[0])},
        {"metric": "quality_eligible_symbol_dates", "value": int(con.execute("SELECT COUNT(*) FROM signal_base").fetchone()[0])},
        {"metric": "sec_float_covered_symbol_dates", "value": int(len(covered))},
        {"metric": "median_daily_sec_coverage_rate", "value": float(coverage_by_date["coverage_rate"].median())},
        {"metric": "minimum_daily_sec_coverage_rate", "value": float(coverage_by_date["coverage_rate"].min())},
        {"metric": "maximum_daily_sec_coverage_rate", "value": float(coverage_by_date["coverage_rate"].max())},
        {"metric": "median_public_float_age_days", "value": float(covered["public_float_age_days"].median())},
        {"metric": "p90_public_float_age_days", "value": float(covered["public_float_age_days"].quantile(0.90))},
        {"metric": "currently_delisted_symbols_in_covered_rows", "value": int(covered.loc[covered["is_delisted"], "symbol"].nunique())},
        {"metric": "primary_minus8_selected_signals", "value": int(len(primary_trades))},
        {"metric": "primary_minus8_signal_days", "value": int(primary_trades["signal_date"].nunique())},
    ]
    audit_summary = pd.DataFrame(audit_rows)

    # Export facts used in the actual six-month window, not the full SEC archive.
    used_fact_keys = covered[["ticker" if "ticker" in covered.columns else "symbol", "accession"]].copy()
    used_accessions = set(covered["accession"].dropna().astype(str))
    facts_used = con.execute("SELECT * FROM sec_float_facts_priced").df()
    facts_used = facts_used[facts_used["accession"].astype(str).isin(used_accessions)].copy()

    primary_columns = [
        "scenario", "big_drop_threshold_pct", "signal_date", "entry_date", "next_date", "activity_rank",
        "symbol", "company_name", "exchange", "country", "drop_pct", "activity_ratio", "raw_close", "amount", "volume",
        "signal_float_shares", "signal_float_market_cap", "public_float_value_usd", "measurement_date",
        "measurement_price_date", "measurement_raw_close", "measurement_price_lag_days", "filed_date", "filing_known_days",
        "public_float_age_days", "form", "accession", "sec_source_url", "first_traded_date", "delisted_at", "is_delisted",
        "entry_open", "entry_close", "next_open", "next_close", "entry_filled", "t_close_available",
        "t1_open_available", "t1_close_available", "gross_t_close", "net_t_close", "stress_net_t_close",
        "gross_t1_open", "net_t1_open", "stress_net_t1_open", "gross_t1_close", "net_t1_close", "stress_net_t1_close",
    ]
    primary_trades = primary_trades[primary_columns].sort_values(["scenario", "entry_date", "activity_rank"])

    top3_columns = [
        "scenario", "signal_date", "activity_rank", "symbol", "company_name", "exchange", "drop_pct", "activity_ratio",
        "raw_close", "amount", "volume", "signal_float_shares", "signal_float_market_cap", "measurement_date", "filed_date",
        "public_float_age_days", "form", "accession", "sec_source_url", "is_delisted",
    ]
    daily_top3 = daily_top3[top3_columns].sort_values(["scenario", "signal_date", "activity_rank"])

    primary_trades.to_csv(out_dir / "trades_primary_minus8.csv", index=False, encoding="utf-8-sig")
    daily_top3.to_csv(out_dir / "daily_top3_rankings.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary_all_thresholds.csv", index=False, encoding="utf-8-sig")
    daily_portfolios.to_csv(out_dir / "daily_portfolios.csv", index=False, encoding="utf-8-sig")
    top_symbols.to_csv(out_dir / "top_symbols_primary.csv", index=False, encoding="utf-8-sig")
    coverage_by_date.to_csv(out_dir / "coverage_by_date.csv", index=False, encoding="utf-8-sig")
    facts_used.to_csv(out_dir / "sec_public_float_facts_used.csv", index=False, encoding="utf-8-sig")
    audit_summary.to_csv(out_dir / "audit_summary.csv", index=False, encoding="utf-8-sig")
    sec_mapping.to_csv(out_dir / "sec_ticker_mapping_audit.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_date.isoformat(),
        "analysis_start_calendar_date": analysis_start.isoformat(),
        "lookback_calendar_days": LOOKBACK_CALENDAR_DAYS,
        "primary_big_drop_threshold_pct": PRIMARY_BIG_DROP_PCT,
        "sensitivity_thresholds_pct": list(BIG_DROP_THRESHOLDS),
        "selection_rule": "Within the SEC public-float-covered historical universe, rank prior-session dollar volume / point-in-time carried-forward free-float market cap; keep top three; then require prior-session adjusted close-to-close drop <= threshold.",
        "free_float_source": "SEC XBRL dei:EntityPublicFloat in companyfacts bulk data",
        "availability_rule": "SEC filed_date must be strictly earlier than signal date",
        "carry_forward_rule": f"latest known public-float fact, measurement age <= {MAX_PUBLIC_FLOAT_AGE_DAYS} days",
        "share_inference_rule": "EntityPublicFloat USD / unadjusted close on latest market date on or before SEC measurement date",
        "split_adjustment_rule": "inferred float shares multiplied by cumulative product of daily split ratios between measurement and signal dates",
        "historical_universe_rule": "signal date must fall within AlphaDojo first_traded_date through delisted_at; currently delisted securities retained when historically listed",
        "multi_ticker_policy": "exclude SEC CIKs with more than one ticker because company-wide public float cannot be allocated reliably across classes",
        "entry_rule": "next reliable U.S. market session open, checked only after signal ranking",
        "exit_rules": exit_labels,
        "one_way_cost": ONE_WAY_COST,
        "tradable_filters": {
            "min_unadjusted_signal_close_usd": TRADABLE_MIN_RAW_CLOSE,
            "min_signal_dollar_volume_usd": TRADABLE_MIN_DOLLAR_VOLUME,
            "min_signal_free_float_market_cap_usd": TRADABLE_MIN_FLOAT_MCAP,
        },
        "limitations": [
            "SEC EntityPublicFloat is periodic and may be months old; it is the latest official disclosure known at the signal date, not an unobservable daily cap table.",
            "Issuances, insider sales, lock-up expirations, repurchases, and affiliate-status changes between SEC public-float disclosures are not inferred; only stock splits are adjusted mechanically.",
            "The primary ranking covers only single-ticker companies with usable SEC public-float disclosures and valid market data. Coverage is reported by date, so results must not be described as an unrestricted full-market backtest.",
            "Daily bars cannot model opening-auction queue position, volatility halts, bid-ask spread, partial fills, or market impact.",
            "Net returns assume 25 bps one-way cost; real microcap execution can be materially worse.",
            "Missing exits are excluded from standard returns and separately assigned -100% in stress_net summaries after an entry fill.",
            "Daily equal-weight compounding for next-day exits is diagnostic because positions initiated on adjacent sessions can overlap.",
        ],
        "sources": manifest,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=scalar, allow_nan=False)

    print("\nAUDIT SUMMARY")
    print(audit_summary.to_string(index=False))
    print("\nCOVERAGE")
    print(coverage_by_date.describe(include="all").to_string())
    print("\nPRIMARY DAILY SUMMARY")
    primary_summary = summary[
        (summary["big_drop_threshold_pct"] == PRIMARY_BIG_DROP_PCT)
        & (summary["statistics_level"] == "daily_equal_weight")
    ]
    print(primary_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"SEC PUBLIC FLOAT BACKTEST FAILED: {exc}", file=sys.stderr, flush=True)
        raise
