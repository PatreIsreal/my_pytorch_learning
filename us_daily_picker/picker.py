#!/usr/bin/env python3
"""Daily U.S. stock picker using the audited SEC point-in-time methodology.

The selector deliberately mirrors the validated backtest entry signal:

1. Use only SEC ``dei:EntityPublicFloat`` facts filed strictly before the signal date.
2. Build the historical listed universe from first-traded and delisting intervals.
3. Estimate signal-date free-float shares from the latest usable SEC public-float
   disclosure and adjust only for stock splits.
4. Apply the production filters: raw close >= $1, dollar volume >= $1 million,
   inferred free-float market cap >= $20 million.
5. Rank the covered tradable universe by dollar volume / free-float market cap.
6. Keep the market-wide top three, then select only stocks whose adjusted
   close-to-close return on the signal date is <= -8%.

The output is a candidate list for the next regular-session open.  The validated
exit plan is +10% take-profit, -15% stop-loss, and mandatory exit no later than
D+3 regular-session close.  This program does not place brokerage orders.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import duckdb
import exchange_calendars as xcals
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
SEC_USER_AGENT = "PatreIsreal daily US stock picker research@users.noreply.github.com"
ET = ZoneInfo("America/New_York")

DROP_THRESHOLD_PCT = -8.0
MAX_PUBLIC_FLOAT_AGE_DAYS = 550
MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS = 10
BAR_HISTORY_BUFFER_DAYS = 35
MIN_DAILY_MARKET_BARS = 1_000
MIN_RAW_CLOSE = 1.0
MIN_DOLLAR_VOLUME = 1_000_000.0
MIN_FLOAT_MARKET_CAP = 20_000_000.0
AVG_PRICE_TOLERANCE = 0.20

TAKE_PROFIT_PCT = 0.10
STOP_LOSS_PCT = 0.15
MAX_HOLD_SESSIONS = 3

ALLOWED_PUBLIC_FLOAT_FORMS = {
    "10-K",
    "10-K/A",
    "20-F",
    "20-F/A",
    "40-F",
    "40-F/A",
}
CIK_NAME_RE = re.compile(r"CIK(\d{10})\.json$")
EXCHANGE_RE = r"NASDAQ|NYSE|AMEX|ARCA|CBOE|BATS"
EXCLUDED_INSTRUMENT_RE = (
    r"(^|[^A-Z])(ETF|ETN|WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PREF|"
    r"CLOSED.END FUND|MUTUAL FUND)([^A-Z]|$)"
)


class DataNotReady(RuntimeError):
    """Raised when the upstream daily file has not published the expected session."""


@dataclass(frozen=True)
class RunContext:
    phase: str
    now_et: datetime
    signal_date: date
    entry_date: date


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


def append_github_output(**values: Any) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return
    with open(output_path, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            text = "" if value is None else str(value)
            if "\n" in text:
                marker = f"EOF_{key}_{int(time.time() * 1000)}"
                handle.write(f"{key}<<{marker}\n{text}\n{marker}\n")
            else:
                handle.write(f"{key}={text}\n")


def append_step_summary(markdown: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n")


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
    session.headers.update(
        {"User-Agent": SEC_USER_AGENT if sec else "PatreIsreal-us-daily-picker/2026"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download(
    session: requests.Session,
    url: str,
    destination: Path,
    *,
    force: bool = False,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not force and destination.exists() and destination.stat().st_size > 1_000:
        return destination.stat().st_size

    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with session.get(
                url,
                stream=True,
                timeout=(30, 2_400),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if temporary.stat().st_size <= 1_000:
                raise RuntimeError(
                    f"Downloaded file is unexpectedly small: {temporary.stat().st_size} bytes"
                )
            temporary.replace(destination)
            return destination.stat().st_size
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(attempt * 8)
    raise AssertionError("unreachable")


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def normalize_ticker(value: str) -> str:
    return value.strip().upper().replace(".", "-")


def session_dates(calendar: Any, start: date, end: date) -> list[date]:
    sessions = calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [stamp.date() for stamp in sessions]


def determine_phase(now_et: datetime, requested: str) -> str:
    if requested != "auto":
        return requested
    current = now_et.timetz().replace(tzinfo=None)
    if dt_time(16, 45) <= current <= dt_time(20, 15):
        return "after_hours"
    if dt_time(8, 0) <= current <= dt_time(9, 20):
        return "premarket"
    return "skip"


def build_run_context(
    *,
    requested_phase: str,
    as_of_date: date | None,
    now_et: datetime | None = None,
) -> RunContext | None:
    now = now_et or datetime.now(ET)
    phase = determine_phase(now, requested_phase)
    if phase == "skip":
        return None

    calendar = xcals.get_calendar("XNYS")
    nearby = session_dates(calendar, now.date() - timedelta(days=15), now.date() + timedelta(days=15))
    if not nearby:
        raise RuntimeError("NYSE calendar returned no nearby sessions")

    if as_of_date is not None:
        if as_of_date not in nearby:
            extended = session_dates(
                calendar,
                as_of_date - timedelta(days=7),
                as_of_date + timedelta(days=7),
            )
            if as_of_date not in extended:
                raise ValueError(f"Requested as-of date is not an XNYS session: {as_of_date}")
            nearby = sorted(set(nearby + extended))
        signal_date = as_of_date
    else:
        prior_or_today = [item for item in nearby if item <= now.date()]
        if not prior_or_today:
            raise RuntimeError("Unable to resolve a completed NYSE session")

        if phase == "after_hours" and now.date() in prior_or_today:
            session = pd.Timestamp(now.date())
            close_stamp = calendar.session_close(session)
            if close_stamp.tzinfo is None:
                close_stamp = close_stamp.tz_localize("UTC")
            close_et = close_stamp.tz_convert("America/New_York").to_pydatetime()
            if now >= close_et + timedelta(minutes=30):
                signal_date = now.date()
            else:
                signal_date = prior_or_today[-2]
        else:
            strictly_prior = [item for item in prior_or_today if item < now.date()]
            signal_date = strictly_prior[-1] if strictly_prior else prior_or_today[-1]

    all_sessions = session_dates(
        calendar,
        signal_date - timedelta(days=7),
        signal_date + timedelta(days=14),
    )
    index = all_sessions.index(signal_date)
    entry_date = all_sessions[index + 1]
    return RunContext(phase=phase, now_et=now, signal_date=signal_date, entry_date=entry_date)


def eligible_alpha_tickers(info_path: Path) -> set[str]:
    connection = duckdb.connect()
    rows = connection.execute(
        f"""
        SELECT DISTINCT UPPER(REPLACE(NULLIF(CAST(ticker AS VARCHAR), ''), '.', '-')) AS ticker
        FROM read_parquet('{qpath(info_path)}')
        WHERE LOWER(COALESCE(CAST(market AS VARCHAR), '')) = 'us'
          AND UPPER(COALESCE(CAST(quote_type AS VARCHAR), '')) = 'EQUITY'
          AND NULLIF(CAST(ticker AS VARCHAR), '') IS NOT NULL
          AND REGEXP_MATCHES(
                UPPER(COALESCE(CAST(full_exchange_name AS VARCHAR), '')),
                '{EXCHANGE_RE}'
              )
          AND NOT REGEXP_MATCHES(
                UPPER(COALESCE(CAST(short_name AS VARCHAR), '') || ' ' ||
                      COALESCE(CAST(long_name AS VARCHAR), '') || ' ' ||
                      COALESCE(CAST(type_disp AS VARCHAR), '')),
                '{EXCLUDED_INSTRUMENT_RE}'
              )
        """
    ).fetchall()
    return {row[0] for row in rows if row[0]}


def parse_submission_tickers(
    archive_path: Path,
    alpha_tickers: set[str],
) -> tuple[dict[int, str], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if CIK_NAME_RE.search(Path(name).name)
        ]
        for name in names:
            match = CIK_NAME_RE.search(Path(name).name)
            if not match:
                continue
            try:
                payload = orjson.loads(archive.read(name))
            except Exception:
                continue
            cik = int(payload.get("cik") or match.group(1))
            tickers = [
                normalize_ticker(str(item))
                for item in (payload.get("tickers") or [])
                if str(item).strip()
            ]
            exchanges = [str(item or "") for item in (payload.get("exchanges") or [])]
            unique_tickers = sorted(set(tickers))
            for position, ticker in enumerate(tickers):
                rows.append(
                    {
                        "cik": cik,
                        "sec_entity_name": payload.get("name"),
                        "ticker": ticker,
                        "sec_exchange": exchanges[position]
                        if position < len(exchanges)
                        else "",
                        "ticker_count_for_cik": len(unique_tickers),
                        "in_alpha_master": ticker in alpha_tickers,
                    }
                )

    mapping = pd.DataFrame(rows)
    if mapping.empty:
        raise RuntimeError("SEC submissions archive yielded no ticker mapping")
    strict = mapping[
        (mapping["ticker_count_for_cik"] == 1) & mapping["in_alpha_master"]
    ].drop_duplicates(["cik", "ticker"])
    cik_to_ticker = dict(zip(strict["cik"].astype(int), strict["ticker"]))
    return cik_to_ticker, mapping


def filing_index_url(cik: int, accession: str) -> str:
    accession_no_dashes = accession.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/"
        f"{accession_no_dashes}/{accession}-index.html"
    )


def parse_public_float_facts(
    archive_path: Path,
    cik_to_ticker: dict[int, str],
    minimum_measurement_date: date,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_ciks = set(cik_to_ticker)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            match = CIK_NAME_RE.search(Path(name).name)
            if not match:
                continue
            cik = int(match.group(1))
            if cik not in target_ciks:
                continue
            try:
                payload = orjson.loads(archive.read(name))
            except Exception:
                continue
            fact = (
                (((payload.get("facts") or {}).get("dei") or {}).get("EntityPublicFloat"))
                or {}
            )
            observations = (fact.get("units") or {}).get("USD") or []
            for item in observations:
                form = str(item.get("form") or "")
                accession = str(item.get("accn") or "")
                try:
                    value = float(item.get("val"))
                    measurement_date = date.fromisoformat(str(item.get("end")))
                    filed_date = date.fromisoformat(str(item.get("filed")))
                except Exception:
                    continue
                if (
                    value <= 0
                    or form not in ALLOWED_PUBLIC_FLOAT_FORMS
                    or measurement_date < minimum_measurement_date
                ):
                    continue
                rows.append(
                    {
                        "cik": cik,
                        "ticker": cik_to_ticker[cik],
                        "sec_entity_name": payload.get("entityName"),
                        "public_float_value_usd": value,
                        "measurement_date": measurement_date,
                        "filed_date": filed_date,
                        "form": form,
                        "accession": accession,
                        "sec_source_url": filing_index_url(cik, accession)
                        if accession
                        else "",
                    }
                )

    facts = pd.DataFrame(rows)
    if facts.empty:
        raise RuntimeError("No SEC EntityPublicFloat facts matched the strict ticker mapping")
    facts = facts.sort_values(
        [
            "ticker",
            "measurement_date",
            "filed_date",
            "accession",
            "public_float_value_usd",
        ],
        kind="mergesort",
    ).drop_duplicates(
        ["ticker", "measurement_date", "filed_date", "accession"],
        keep="last",
    )
    return facts.reset_index(drop=True)


def ensure_weekly_reference_cache(
    cache_directory: Path,
    signal_date: date,
    *,
    force_refresh: bool,
) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    info_path = cache_directory / "stock_info.parquet"
    mapping_path = cache_directory / "sec_ticker_mapping.parquet"
    facts_path = cache_directory / "sec_public_float_facts.parquet"
    manifest_path = cache_directory / "manifest.json"

    alpha_session = http_session(False)
    sec_session = http_session(True)
    info_size = download(
        alpha_session,
        ALPHA_SOURCES["info"],
        info_path,
        force=force_refresh,
    )

    if (
        not force_refresh
        and mapping_path.exists()
        and facts_path.exists()
        and mapping_path.stat().st_size > 1_000
        and facts_path.stat().st_size > 1_000
    ):
        facts = pd.read_parquet(facts_path)
        for column in ("measurement_date", "filed_date"):
            facts[column] = pd.to_datetime(facts[column], errors="coerce").dt.date
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        manifest["cache_reused"] = True
        return info_path, facts, manifest

    submissions_path = cache_directory / "submissions.zip"
    companyfacts_path = cache_directory / "companyfacts.zip"
    submissions_size = download(
        sec_session,
        SEC_SOURCES["submissions"],
        submissions_path,
        force=force_refresh,
    )
    companyfacts_size = download(
        sec_session,
        SEC_SOURCES["companyfacts"],
        companyfacts_path,
        force=force_refresh,
    )

    alpha_tickers = eligible_alpha_tickers(info_path)
    cik_to_ticker, mapping = parse_submission_tickers(submissions_path, alpha_tickers)
    minimum_measurement_date = signal_date - timedelta(
        days=MAX_PUBLIC_FLOAT_AGE_DAYS + BAR_HISTORY_BUFFER_DAYS
    )
    facts = parse_public_float_facts(
        companyfacts_path,
        cik_to_ticker,
        minimum_measurement_date,
    )
    mapping.to_parquet(mapping_path, index=False)
    facts.to_parquet(facts_path, index=False)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "info_size_bytes": info_size,
        "submissions_size_bytes": submissions_size,
        "companyfacts_size_bytes": companyfacts_size,
        "strict_single_ticker_cik_count": len(cik_to_ticker),
        "public_float_fact_count": len(facts),
        "minimum_measurement_date": minimum_measurement_date.isoformat(),
        "cache_reused": False,
        "sources": {**ALPHA_SOURCES, **SEC_SOURCES},
    }
    write_json(manifest_path, manifest)
    return info_path, facts, manifest


def download_fresh_kline(work_directory: Path) -> tuple[Path, int]:
    work_directory.mkdir(parents=True, exist_ok=True)
    destination = work_directory / "stock_kline.parquet"
    size = download(http_session(False), ALPHA_SOURCES["kline"], destination, force=True)
    return destination, size


def select_daily_candidates(
    *,
    info_path: Path,
    kline_path: Path,
    sec_facts: pd.DataFrame,
    signal_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    history_start = signal_date - timedelta(
        days=MAX_PUBLIC_FLOAT_AGE_DAYS
        + MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS
        + BAR_HISTORY_BUFFER_DAYS
    )

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='6GB'")
    connection.execute(
        f"CREATE VIEW info_raw AS SELECT * FROM read_parquet('{qpath(info_path)}')"
    )
    connection.execute(
        f"CREATE VIEW kline_raw AS SELECT * FROM read_parquet('{qpath(kline_path)}')"
    )
    connection.register("sec_float_facts_raw", sec_facts)

    latest_market_date = connection.execute(
        """
        SELECT MAX(TRY_CAST(bar_time AS DATE))
        FROM kline_raw
        WHERE UPPER(COALESCE(CAST(kline_t AS VARCHAR), '1D')) = '1D'
          AND TRY_CAST(close AS DOUBLE) > 0
        """
    ).fetchone()[0]
    if isinstance(latest_market_date, datetime):
        latest_market_date = latest_market_date.date()
    if latest_market_date is None or latest_market_date < signal_date:
        raise DataNotReady(
            f"Daily source latest session is {latest_market_date}; expected {signal_date}"
        )

    connection.execute(
        f"""
        CREATE TEMP TABLE info_intervals AS
        SELECT
            UPPER(REPLACE(NULLIF(CAST(ticker AS VARCHAR), ''), '.', '-')) AS ticker,
            COALESCE(
                CAST(short_name AS VARCHAR),
                CAST(long_name AS VARCHAR),
                CAST(ticker AS VARCHAR)
            ) AS company_name,
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
                '{EXCHANGE_RE}'
              )
          AND NOT REGEXP_MATCHES(
                UPPER(COALESCE(CAST(short_name AS VARCHAR), '') || ' ' ||
                      COALESCE(CAST(long_name AS VARCHAR), '') || ' ' ||
                      COALESCE(CAST(type_disp AS VARCHAR), '')),
                '{EXCLUDED_INSTRUMENT_RE}'
              )
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE bars_source AS
        WITH prepared AS (
            SELECT
                UPPER(REPLACE(CAST(symbol AS VARCHAR), '.', '-')) AS symbol,
                TRY_CAST(bar_time AS DATE) AS trade_date,
                TRY_CAST(open AS DOUBLE) AS adjusted_open,
                TRY_CAST(high AS DOUBLE) AS adjusted_high,
                TRY_CAST(low AS DOUBLE) AS adjusted_low,
                TRY_CAST(close AS DOUBLE) AS adjusted_close,
                TRY_CAST(vol AS DOUBLE) AS volume,
                TRY_CAST(amount AS DOUBLE) AS amount,
                TRY_CAST(change_p AS DOUBLE) AS vendor_change_pct,
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
        )
        SELECT * EXCLUDE (row_rank)
        FROM prepared
        WHERE row_rank = 1
          AND trade_date IS NOT NULL
          AND adjusted_open > 0
          AND adjusted_high > 0
          AND adjusted_low > 0
          AND adjusted_close > 0
          AND volume >= 0
        """,
        [history_start, signal_date],
    )

    connection.execute(
        """
        CREATE TEMP TABLE bars_listed AS
        SELECT * EXCLUDE (interval_rank)
        FROM (
            SELECT
                bars.*,
                info.company_name,
                info.exchange,
                info.first_traded_date,
                info.delisted_at,
                info.is_delisted,
                info.country,
                ROW_NUMBER() OVER (
                    PARTITION BY bars.symbol, bars.trade_date
                    ORDER BY
                        info.first_traded_date DESC NULLS LAST,
                        info.delisted_at DESC NULLS LAST
                ) AS interval_rank
            FROM bars_source bars
            INNER JOIN info_intervals info
                ON info.ticker = bars.symbol
               AND (
                    info.first_traded_date IS NULL
                    OR bars.trade_date >= info.first_traded_date
               )
               AND (
                    info.delisted_at IS NULL
                    OR bars.trade_date <= info.delisted_at
               )
        )
        WHERE interval_rank = 1
        """
    )

    connection.execute(
        """
        CREATE TEMP TABLE bars_calc AS
        WITH raw AS (
            SELECT
                *,
                adjusted_open / adjustment_factor AS raw_open,
                adjusted_high / adjustment_factor AS raw_high,
                adjusted_low / adjustment_factor AS raw_low,
                adjusted_close / adjustment_factor AS raw_close,
                CASE
                    WHEN amount > 0 AND volume > 0 THEN amount / volume
                    ELSE NULL
                END AS average_trade_price,
                EXP(
                    SUM(LN(split_ratio)) OVER (
                        PARTITION BY symbol
                        ORDER BY trade_date
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    )
                ) AS cumulative_split_factor,
                LAG(adjusted_close) OVER (
                    PARTITION BY symbol ORDER BY trade_date
                ) AS prior_adjusted_close
            FROM bars_listed
        )
        SELECT
            *,
            CASE
                WHEN prior_adjusted_close > 0
                THEN 100.0 * (adjusted_close / prior_adjusted_close - 1.0)
                ELSE NULL
            END AS drop_pct,
            CASE
                WHEN average_trade_price > 0
                 AND raw_low > 0
                 AND raw_high > 0
                 AND average_trade_price BETWEEN raw_low * ? AND raw_high * ?
                THEN TRUE
                ELSE FALSE
            END AS average_price_inside_raw_ohlc
        FROM raw
        """,
        [1.0 - AVG_PRICE_TOLERANCE, 1.0 + AVG_PRICE_TOLERANCE],
    )

    market_dates = connection.execute(
        """
        SELECT trade_date, COUNT(*) AS listed_bar_count
        FROM bars_calc
        GROUP BY trade_date
        HAVING COUNT(*) >= ?
        ORDER BY trade_date
        """,
        [MIN_DAILY_MARKET_BARS],
    ).df()
    market_dates["trade_date"] = pd.to_datetime(market_dates["trade_date"]).dt.date
    if signal_date not in set(market_dates["trade_date"]):
        raise DataNotReady(
            f"Expected signal date {signal_date} does not have a complete listed market cross-section"
        )

    connection.execute(
        """
        CREATE TEMP TABLE priced_sec_facts AS
        SELECT * EXCLUDE (price_rank)
        FROM (
            SELECT
                facts.*,
                bars.trade_date AS measurement_price_date,
                bars.raw_close AS measurement_raw_close,
                bars.cumulative_split_factor AS measurement_split_factor,
                facts.public_float_value_usd / bars.raw_close
                    AS measurement_float_shares,
                DATE_DIFF('day', bars.trade_date, facts.measurement_date)
                    AS measurement_price_lag_days,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        facts.ticker,
                        facts.measurement_date,
                        facts.filed_date,
                        facts.accession
                    ORDER BY bars.trade_date DESC
                ) AS price_rank
            FROM sec_float_facts_raw facts
            INNER JOIN bars_calc bars
                ON bars.symbol = facts.ticker
               AND bars.trade_date <= facts.measurement_date
               AND bars.trade_date >= facts.measurement_date - INTERVAL ? DAY
               AND bars.raw_close > 0
        )
        WHERE price_rank = 1
          AND measurement_float_shares > 0
        """,
        [MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS],
    )

    connection.execute(
        """
        CREATE TEMP TABLE signal_base AS
        SELECT bars.*
        FROM bars_calc bars
        WHERE bars.trade_date = ?
          AND bars.amount > 0
          AND bars.volume > 0
          AND bars.raw_close > 0
          AND bars.average_price_inside_raw_ohlc
          AND bars.drop_pct IS NOT NULL
        """,
        [signal_date],
    )

    connection.execute(
        """
        CREATE TEMP TABLE covered_signal_rows AS
        SELECT * EXCLUDE (fact_rank)
        FROM (
            SELECT
                bars.*,
                facts.cik,
                facts.sec_entity_name,
                facts.public_float_value_usd,
                facts.measurement_date,
                facts.measurement_price_date,
                facts.measurement_raw_close,
                facts.measurement_split_factor,
                facts.measurement_float_shares,
                facts.measurement_price_lag_days,
                facts.filed_date,
                facts.form,
                facts.accession,
                facts.sec_source_url,
                DATE_DIFF('day', facts.measurement_date, bars.trade_date)
                    AS public_float_age_days,
                DATE_DIFF('day', facts.filed_date, bars.trade_date)
                    AS filing_known_days,
                facts.measurement_float_shares
                    * (bars.cumulative_split_factor / facts.measurement_split_factor)
                    AS signal_float_shares,
                bars.raw_close
                    * facts.measurement_float_shares
                    * (bars.cumulative_split_factor / facts.measurement_split_factor)
                    AS signal_float_market_cap,
                bars.amount / (
                    bars.raw_close
                    * facts.measurement_float_shares
                    * (bars.cumulative_split_factor / facts.measurement_split_factor)
                ) AS activity_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY bars.symbol, bars.trade_date
                    ORDER BY
                        facts.measurement_date DESC,
                        facts.filed_date DESC,
                        facts.accession DESC
                ) AS fact_rank
            FROM signal_base bars
            INNER JOIN priced_sec_facts facts
                ON facts.ticker = bars.symbol
               AND facts.filed_date < bars.trade_date
               AND facts.measurement_date <= bars.trade_date
               AND DATE_DIFF(
                    'day', facts.measurement_date, bars.trade_date
               ) BETWEEN 0 AND ?
        )
        WHERE fact_rank = 1
          AND signal_float_shares > 0
          AND signal_float_market_cap > 0
          AND activity_ratio > 0
        """,
        [MAX_PUBLIC_FLOAT_AGE_DAYS],
    )

    quality_count = connection.execute(
        "SELECT COUNT(*) FROM signal_base"
    ).fetchone()[0]
    covered_count = connection.execute(
        "SELECT COUNT(*) FROM covered_signal_rows"
    ).fetchone()[0]

    tradable = connection.execute(
        """
        SELECT *
        FROM covered_signal_rows
        WHERE raw_close >= ?
          AND amount >= ?
          AND signal_float_market_cap >= ?
        ORDER BY activity_ratio DESC, amount DESC, symbol ASC
        """,
        [MIN_RAW_CLOSE, MIN_DOLLAR_VOLUME, MIN_FLOAT_MARKET_CAP],
    ).df()
    if tradable.empty:
        raise RuntimeError("The SEC-covered tradable universe is empty")

    tradable["activity_rank"] = range(1, len(tradable) + 1)
    top_three = tradable.head(3).copy()
    selected = top_three[top_three["drop_pct"] <= DROP_THRESHOLD_PCT].copy()

    for frame in (top_three, selected):
        for column in (
            "trade_date",
            "measurement_date",
            "measurement_price_date",
            "filed_date",
            "first_traded_date",
            "delisted_at",
        ):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date

    diagnostics = {
        "source_latest_market_date": latest_market_date,
        "signal_date": signal_date,
        "quality_eligible_count": int(quality_count),
        "sec_public_float_covered_count": int(covered_count),
        "tradable_covered_count": int(len(tradable)),
        "sec_coverage_rate": float(covered_count / quality_count)
        if quality_count
        else None,
        "tradable_coverage_rate": float(len(tradable) / quality_count)
        if quality_count
        else None,
        "drop_threshold_pct": DROP_THRESHOLD_PCT,
        "minimum_raw_close": MIN_RAW_CLOSE,
        "minimum_dollar_volume": MIN_DOLLAR_VOLUME,
        "minimum_float_market_cap": MIN_FLOAT_MARKET_CAP,
    }
    return top_three, selected, diagnostics


def money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"${number / 1_000:.1f}K"
    return f"${number:.2f}"


def percentage(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}%"


def ratio(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.3f}x"


def markdown_table(frame: pd.DataFrame, *, selected_only: bool) -> str:
    if frame.empty:
        return "无符合条件的标的。"
    headers = [
        "排名",
        "代码",
        "公司",
        "前日跌幅",
        "活跃度",
        "成交额",
        "自由流通市值",
        "SEC披露年龄",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in frame.itertuples(index=False):
        company = str(row.company_name).replace("|", "/")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.activity_rank)),
                    f"**{row.symbol}**" if selected_only else str(row.symbol),
                    company,
                    percentage(row.drop_pct),
                    ratio(row.activity_ratio),
                    money(row.amount),
                    money(row.signal_float_market_cap),
                    f"{int(row.public_float_age_days)}天",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_markdown(
    *,
    context: RunContext,
    top_three: pd.DataFrame,
    selected: pd.DataFrame,
    diagnostics: dict[str, Any],
    reference_manifest: dict[str, Any],
    kline_size: int,
) -> tuple[str, str, str]:
    tickers = [str(item) for item in selected.get("symbol", pd.Series(dtype=str)).tolist()]
    status = "ready" if tickers else "no_signal"
    suffix = ", ".join(tickers) if tickers else "NO TRADE"
    issue_title = (
        f"[US PICKS] {context.signal_date.isoformat()} → "
        f"{context.entry_date.isoformat()} | {suffix}"
    )

    if tickers:
        decision = (
            f"本交易日产生 **{len(tickers)} 只可执行候选**："
            + "、".join(f"**{ticker}**" for ticker in tickers)
            + f"。计划在 **{context.entry_date.isoformat()} 美股常规时段开盘**介入。"
        )
    else:
        decision = (
            "当日全市场活跃度前三中，没有股票同时满足前一交易日跌幅≤-8%，"
            "因此下一交易日 **不交易**。"
        )

    generated = context.now_et.strftime("%Y-%m-%d %H:%M:%S %Z")
    selected_section = markdown_table(selected, selected_only=True)
    top_three_section = markdown_table(top_three, selected_only=False)
    coverage = diagnostics.get("sec_coverage_rate")
    tradable_coverage = diagnostics.get("tradable_coverage_rate")

    markdown = f"""# 美股每日选票｜{context.signal_date.isoformat()} ET 收盘信号

> 自动生成时间：{generated}  
> 运行阶段：{context.phase}  
> 下一计划入场日：**{context.entry_date.isoformat()}**

## 今日结论

{decision}

## 入选标的

{selected_section}

## 全市场活跃度前三（审计表）

{top_three_section}

## 统一执行规则

- **入场：** 下一正常交易日常规时段开盘；候选股等权。
- **止盈：** 实际成交价上方 **+10%**。
- **止损：** 实际成交价下方 **-15%**。
- **时间退出：** 若止盈、止损均未触发，最迟在入场后的第 **3** 个交易日收盘卖出。
- **同一根日线双触发：** 按止损先发生处理，保持回测的保守口径。
- **资金纪律：** 上一批持仓仍未全部退出时，新信号只作观察，不重复满仓开新批次。

## 数据质量与口径

- 质量合格股票数：{diagnostics['quality_eligible_count']:,}
- 可获得点时SEC流通值：{diagnostics['sec_public_float_covered_count']:,}
- 通过实盘过滤的覆盖股票数：{diagnostics['tradable_covered_count']:,}
- SEC覆盖率：{percentage(coverage * 100 if coverage is not None else None)}
- 实盘过滤覆盖率：{percentage(tradable_coverage * 100 if tradable_coverage is not None else None)}
- 行情源最新交易日：{diagnostics['source_latest_market_date']}
- 本次下载日线文件：{kline_size / 1_000_000:.1f} MB
- SEC缓存：{'复用本周缓存' if reference_manifest.get('cache_reused') else '本周重新构建'}

## 重要提示

本名单严格复用已审计回测的入场条件，但SEC `EntityPublicFloat` 是周期性披露，
不是每日股东名册；列表仅覆盖能够用点时官方披露核验的股票池。日线数据无法模拟
开盘竞价排队、停牌、买卖价差、部分成交和市场冲击。本输出是研究型候选名单，
不构成投资建议，也不会自动连接券商下单。
"""
    return status, issue_title, markdown


def write_frame_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def write_success_outputs(
    *,
    output_directory: Path,
    context: RunContext,
    top_three: pd.DataFrame,
    selected: pd.DataFrame,
    diagnostics: dict[str, Any],
    reference_manifest: dict[str, Any],
    kline_size: int,
) -> dict[str, Any]:
    result_directory = output_directory / context.signal_date.isoformat()
    result_directory.mkdir(parents=True, exist_ok=True)

    status, issue_title, markdown = build_markdown(
        context=context,
        top_three=top_three,
        selected=selected,
        diagnostics=diagnostics,
        reference_manifest=reference_manifest,
        kline_size=kline_size,
    )
    markdown_path = result_directory / "picks.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    (output_directory / "latest.md").write_text(markdown, encoding="utf-8")
    write_frame_csv(result_directory / "top3.csv", top_three)
    write_frame_csv(result_directory / "selected.csv", selected)

    payload = {
        "status": status,
        "phase": context.phase,
        "generated_at_et": context.now_et.isoformat(),
        "signal_date": context.signal_date,
        "entry_date": context.entry_date,
        "issue_title": issue_title,
        "selected_tickers": selected.get("symbol", pd.Series(dtype=str)).tolist(),
        "diagnostics": diagnostics,
        "reference_manifest": reference_manifest,
        "top_three": top_three.to_dict("records"),
        "selected": selected.to_dict("records"),
        "exit_rule": {
            "take_profit_pct": TAKE_PROFIT_PCT,
            "stop_loss_pct": STOP_LOSS_PCT,
            "maximum_holding_sessions": MAX_HOLD_SESSIONS,
        },
    }
    write_json(result_directory / "picks.json", payload)
    write_json(output_directory / "latest.json", payload)
    return {
        **payload,
        "markdown_path": str(markdown_path),
        "notify": True,
    }


def write_data_not_ready_status(
    *,
    work_directory: Path,
    context: RunContext,
    message: str,
) -> dict[str, Any]:
    status_path = work_directory / "data_not_ready.md"
    markdown = f"""# 美股每日选票数据尚未就绪

- 美东运行时间：{context.now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}
- 目标信号日：{context.signal_date.isoformat()}
- 计划入场日：{context.entry_date.isoformat()}
- 原因：{message}

夜盘阶段的数据未完成发布时，系统会在下一交易日盘前自动再次运行。
"""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(markdown, encoding="utf-8")
    notify = context.phase == "premarket"
    return {
        "status": "data_not_ready",
        "phase": context.phase,
        "signal_date": context.signal_date,
        "entry_date": context.entry_date,
        "issue_title": (
            f"[US PICKS DATA] {context.signal_date.isoformat()} | DATA NOT READY"
        ),
        "markdown_path": str(status_path),
        "notify": notify,
    }


def emit_result(result: dict[str, Any]) -> None:
    append_github_output(
        status=result.get("status"),
        phase=result.get("phase"),
        signal_date=scalar(result.get("signal_date")),
        entry_date=scalar(result.get("entry_date")),
        issue_title=result.get("issue_title"),
        markdown_path=result.get("markdown_path"),
        notify=str(bool(result.get("notify"))).lower(),
        selected_tickers=",".join(result.get("selected_tickers") or []),
    )
    markdown_path = result.get("markdown_path")
    if markdown_path and Path(markdown_path).exists():
        append_step_summary(Path(markdown_path).read_text(encoding="utf-8"))
    else:
        append_step_summary(f"Picker status: **{result.get('status')}**")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=scalar))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("auto", "after_hours", "premarket", "manual"),
        default="auto",
    )
    parser.add_argument("--as-of-date", type=date.fromisoformat)
    parser.add_argument("--cache-dir", default=".cache/us_daily_picker/sec")
    parser.add_argument("--work-dir", default=".work/us_daily_picker")
    parser.add_argument("--output-dir", default="daily_picks")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-sec", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    context = build_run_context(
        requested_phase=args.phase,
        as_of_date=args.as_of_date,
    )
    if context is None:
        result = {
            "status": "skipped",
            "phase": "outside_window",
            "notify": False,
        }
        emit_result(result)
        return 0

    output_directory = Path(args.output_dir)
    existing_result = output_directory / context.signal_date.isoformat() / "picks.json"
    if existing_result.exists() and not args.force:
        payload = json.loads(existing_result.read_text(encoding="utf-8"))
        result = {
            "status": "already_done",
            "phase": context.phase,
            "signal_date": context.signal_date,
            "entry_date": context.entry_date,
            "issue_title": payload.get("issue_title"),
            "selected_tickers": payload.get("selected_tickers") or [],
            "markdown_path": str(existing_result.with_name("picks.md")),
            "notify": False,
        }
        emit_result(result)
        return 0

    cache_directory = Path(args.cache_dir)
    work_directory = Path(args.work_dir)
    if args.force and work_directory.exists():
        shutil.rmtree(work_directory)
    work_directory.mkdir(parents=True, exist_ok=True)

    try:
        info_path, sec_facts, reference_manifest = ensure_weekly_reference_cache(
            cache_directory,
            context.signal_date,
            force_refresh=args.refresh_sec,
        )
        kline_path, kline_size = download_fresh_kline(work_directory)
        top_three, selected, diagnostics = select_daily_candidates(
            info_path=info_path,
            kline_path=kline_path,
            sec_facts=sec_facts,
            signal_date=context.signal_date,
        )
        result = write_success_outputs(
            output_directory=output_directory,
            context=context,
            top_three=top_three,
            selected=selected,
            diagnostics=diagnostics,
            reference_manifest=reference_manifest,
            kline_size=kline_size,
        )
    except DataNotReady as error:
        result = write_data_not_ready_status(
            work_directory=work_directory,
            context=context,
            message=str(error),
        )
    emit_result(result)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        append_github_output(status="failed", notify="false")
        print(f"DAILY PICKER FAILED: {error}", file=sys.stderr, flush=True)
        raise
