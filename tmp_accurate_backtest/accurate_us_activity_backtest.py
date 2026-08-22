#!/usr/bin/env python3
"""Point-in-time six-month U.S. free-float activity backtest.

This version deliberately does NOT use a current free-float snapshot for historical
ranking.  It uses the daily bar field ``tr`` (turnover rate, percent of free-float
shares traded) and reconstructs the requested dollar-volume / free-float-market-cap
ratio as:

    activity = (tr / 100) * (daily average trade price / unadjusted close)

where daily average trade price = amount / volume and unadjusted close is recovered
from adjusted close / cumulative adjustment factor.  Algebraically this is:

    amount / (unadjusted close * point-in-time implied free-float shares)

with implied free-float shares = volume / (tr / 100).

The historical security universe is formed date by date from listing and delisting
intervals.  Signal-day ranking is completed before any entry/exit availability is
checked, so future missing bars cannot change the top-three ranking.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SOURCES = {
    "info": "https://huggingface.co/datasets/AlphaDojo/dojo_stock_info/resolve/main/data.parquet?download=true",
    "quote": "https://huggingface.co/datasets/AlphaDojo/dojo_quote/resolve/main/data.parquet?download=true",
    "kline": "https://huggingface.co/datasets/AlphaDojo/dojo_stock_kline/resolve/main/data.parquet?download=true",
}
LOOKBACK_CALENDAR_DAYS = 183
BIG_DROP_THRESHOLDS = (-5.0, -8.0, -10.0, -15.0)
PRIMARY_BIG_DROP_PCT = -8.0
ONE_WAY_COST = 0.0025
MIN_DAILY_MARKET_BARS = 1000
TRADABLE_MIN_RAW_CLOSE = 1.0
TRADABLE_MIN_DOLLAR_VOLUME = 1_000_000.0
TRADABLE_MIN_FLOAT_MCAP = 20_000_000.0
AVG_PRICE_TOLERANCE = 0.20
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260822


@dataclass(frozen=True)
class Scenario:
    name: str
    tradable: bool


SCENARIOS = (
    Scenario("历史全量可验证", False),
    Scenario("历史实盘过滤", True),
)


def http_session() -> requests.Session:
    retry = Retry(
        total=7,
        connect=7,
        read=7,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update({"User-Agent": "point-in-time-us-float-backtest/2026-08-22"})
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def download(session: requests.Session, url: str, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1_000:
        return destination.stat().st_size
    temp = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with session.get(url, stream=True, timeout=(30, 1200), allow_redirects=True) as response:
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
            time.sleep(attempt * 5)
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
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


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


def summarize_returns(
    trades: pd.DataFrame,
    scenario: str,
    threshold: float,
    exit_key: str,
    return_basis: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    selected = trades[(trades["scenario"] == scenario) & (trades["big_drop_threshold_pct"] == threshold)].copy()
    return_column = f"{return_basis}_{exit_key}"
    available = selected[selected[return_column].notna()].copy()
    daily = (
        available.groupby("entry_date", as_index=False)[return_column]
        .mean()
        .rename(columns={return_column: "portfolio_return"})
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
        **metric_block(available[return_column] if len(available) else pd.Series(dtype=float)),
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
    parser.add_argument("--data-dir", default=".tmp_accurate_backtest_data")
    parser.add_argument("--out-dir", default="accurate_backtest_output")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = http_session()
    files: dict[str, Path] = {}
    manifest: dict[str, Any] = {}
    for name, url in SOURCES.items():
        path = data_dir / f"{name}.parquet"
        size = download(session, url, path)
        files[name] = path
        manifest[name] = {"url": url, "size_bytes": size}
        print(f"downloaded {name}: {size / 1_000_000:.1f} MB", flush=True)

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute(f"CREATE VIEW info_raw AS SELECT * FROM read_parquet('{qpath(files['info'])}')")
    con.execute(f"CREATE VIEW quote_raw AS SELECT * FROM read_parquet('{qpath(files['quote'])}')")
    con.execute(f"CREATE VIEW kline_raw AS SELECT * FROM read_parquet('{qpath(files['kline'])}')")

    con.execute(r"""
        CREATE TEMP TABLE info_intervals AS
        SELECT
            NULLIF(CAST(ticker AS VARCHAR), '') AS ticker,
            COALESCE(CAST(short_name AS VARCHAR), CAST(long_name AS VARCHAR), CAST(ticker AS VARCHAR)) AS company_name,
            COALESCE(CAST(full_exchange_name AS VARCHAR), '') AS exchange,
            TRY_CAST(first_traded_date AS DATE) AS first_traded_date,
            TRY_CAST(delisted_at AS DATE) AS delisted_at,
            COALESCE(TRY_CAST(is_delisted AS BOOLEAN), FALSE) AS is_delisted,
            UPPER(COALESCE(CAST(quote_type AS VARCHAR), '')) AS quote_type,
            UPPER(COALESCE(CAST(type_disp AS VARCHAR), '')) AS type_disp
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

    latest_date = con.execute("""
        SELECT MAX(TRY_CAST(k.bar_time AS DATE))
        FROM kline_raw k
        WHERE UPPER(COALESCE(CAST(k.kline_t AS VARCHAR), '1D')) = '1D'
          AND TRY_CAST(k.close AS DOUBLE) > 0
          AND EXISTS (SELECT 1 FROM info_intervals i WHERE i.ticker = CAST(k.symbol AS VARCHAR))
    """).fetchone()[0]
    if latest_date is None:
        raise RuntimeError("no U.S. daily bars matched the historical security master")
    if isinstance(latest_date, datetime):
        latest_date = latest_date.date()
    analysis_start = latest_date - timedelta(days=LOOKBACK_CALENDAR_DAYS)
    load_start = analysis_start - timedelta(days=20)

    con.execute("""
        CREATE TEMP TABLE bars_source AS
        WITH prepared AS (
            SELECT
                CAST(k.symbol AS VARCHAR) AS symbol,
                TRY_CAST(k.bar_time AS DATE) AS trade_date,
                TRY_CAST(k.open AS DOUBLE) AS open,
                TRY_CAST(k.high AS DOUBLE) AS high,
                TRY_CAST(k.low AS DOUBLE) AS low,
                TRY_CAST(k.close AS DOUBLE) AS close,
                TRY_CAST(k.vol AS DOUBLE) AS volume,
                TRY_CAST(k.amount AS DOUBLE) AS amount,
                TRY_CAST(k.change_p AS DOUBLE) AS vendor_change_pct,
                TRY_CAST(k.tr AS DOUBLE) AS turnover_rate_pct,
                CASE
                    WHEN TRY_CAST(k.adj_factor_cum AS DOUBLE) > 0 THEN TRY_CAST(k.adj_factor_cum AS DOUBLE)
                    ELSE 1.0
                END AS adjustment_factor,
                COALESCE(TRY_CAST(k.dividends AS DOUBLE), 0.0) AS dividends,
                COALESCE(TRY_CAST(k.splits AS DOUBLE), 0.0) AS splits,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(k.symbol AS VARCHAR), TRY_CAST(k.bar_time AS DATE)
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
          AND open > 0 AND high > 0 AND low > 0 AND close > 0
          AND volume >= 0
    """, [load_start, latest_date])

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
        CREATE TEMP TABLE market_dates AS
        SELECT trade_date, COUNT(*) AS listed_bar_count
        FROM bars_listed
        GROUP BY trade_date
        HAVING COUNT(*) >= ?
        ORDER BY trade_date
    """, [MIN_DAILY_MARKET_BARS])
    market_date_count = int(con.execute("SELECT COUNT(*) FROM market_dates").fetchone()[0])
    if market_date_count < 80:
        raise RuntimeError(f"too few reliable U.S. market dates: {market_date_count}")

    con.execute("""
        CREATE TEMP TABLE calendar_map AS
        SELECT
            trade_date AS signal_date,
            LEAD(trade_date, 1) OVER (ORDER BY trade_date) AS entry_date,
            LEAD(trade_date, 2) OVER (ORDER BY trade_date) AS next_date
        FROM market_dates
    """)

    con.execute("""
        CREATE TEMP TABLE bars_calc AS
        WITH lagged AS (
            SELECT
                *,
                LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS previous_adjusted_close
            FROM bars_listed
        ), derived AS (
            SELECT
                *,
                open / adjustment_factor AS raw_open,
                high / adjustment_factor AS raw_high,
                low / adjustment_factor AS raw_low,
                close / adjustment_factor AS raw_close,
                CASE WHEN volume > 0 AND amount > 0 THEN amount / volume ELSE NULL END AS avg_trade_price,
                CASE WHEN turnover_rate_pct > 0 THEN volume / (turnover_rate_pct / 100.0) ELSE NULL END AS implied_float_shares,
                CASE
                    WHEN previous_adjusted_close > 0
                    THEN 100.0 * (close / previous_adjusted_close - 1.0)
                    ELSE NULL
                END AS drop_pct
            FROM lagged
        )
        SELECT
            *,
            raw_close * implied_float_shares AS implied_float_market_cap,
            CASE
                WHEN turnover_rate_pct > 0 AND avg_trade_price > 0 AND raw_close > 0
                THEN (turnover_rate_pct / 100.0) * (avg_trade_price / raw_close)
                ELSE NULL
            END AS activity_ratio,
            CASE
                WHEN avg_trade_price > 0 AND raw_low > 0 AND raw_high > 0
                 AND avg_trade_price BETWEEN raw_low * ? AND raw_high * ?
                THEN TRUE ELSE FALSE
            END AS avg_price_inside_ohlc,
            CASE
                WHEN drop_pct IS NOT NULL AND vendor_change_pct IS NOT NULL
                THEN ABS(drop_pct - vendor_change_pct)
                ELSE NULL
            END AS change_pct_abs_error
        FROM derived
    """, [1.0 - AVG_PRICE_TOLERANCE, 1.0 + AVG_PRICE_TOLERANCE])

    eligible = con.execute("""
        SELECT b.*
        FROM bars_calc b
        INNER JOIN market_dates m USING (trade_date)
        WHERE b.trade_date >= ?
          AND b.volume > 0
          AND b.amount > 0
          AND b.turnover_rate_pct > 0
          AND b.implied_float_shares > 0
          AND b.implied_float_market_cap > 0
          AND b.activity_ratio > 0
          AND b.avg_price_inside_ohlc
          AND b.drop_pct IS NOT NULL
    """, [analysis_start]).df()
    if eligible.empty:
        raise RuntimeError("no eligible signal rows after point-in-time turnover validation")

    prices = con.execute("""
        SELECT symbol, trade_date, open, close
        FROM bars_listed
        INNER JOIN market_dates USING (trade_date)
        WHERE open > 0 AND close > 0
    """).df()
    calendar = con.execute("SELECT * FROM calendar_map").df()

    for col in ("trade_date",):
        eligible[col] = pd.to_datetime(eligible[col]).dt.date
        prices[col] = pd.to_datetime(prices[col]).dt.date
    for col in ("signal_date", "entry_date", "next_date"):
        calendar[col] = pd.to_datetime(calendar[col]).dt.date

    all_rankings: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        frame = eligible.copy()
        if scenario.tradable:
            frame = frame[
                (frame["raw_close"] >= TRADABLE_MIN_RAW_CLOSE)
                & (frame["amount"] >= TRADABLE_MIN_DOLLAR_VOLUME)
                & (frame["implied_float_market_cap"] >= TRADABLE_MIN_FLOAT_MCAP)
            ].copy()
        frame = frame.sort_values(
            ["trade_date", "activity_ratio", "amount", "symbol"],
            ascending=[True, False, False, True],
            kind="mergesort",
        )
        frame["activity_rank"] = frame.groupby("trade_date").cumcount() + 1
        top_three = frame[frame["activity_rank"] <= 3].copy()
        top_three["scenario"] = scenario.name
        top_three = top_three.rename(columns={"trade_date": "signal_date"})
        all_rankings.append(top_three)
        for threshold in BIG_DROP_THRESHOLDS:
            chosen = top_three[top_three["drop_pct"] <= threshold].copy()
            chosen["big_drop_threshold_pct"] = threshold
            all_selected.append(chosen)

    rankings = pd.concat(all_rankings, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)

    selected = selected.merge(calendar, on="signal_date", how="left", validate="many_to_one")
    entry_prices = prices.rename(columns={"trade_date": "entry_date", "open": "entry_open", "close": "entry_close"})
    next_prices = prices.rename(columns={"trade_date": "next_date", "open": "next_open", "close": "next_close"})
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

    # Keep the primary threshold trade file concise while retaining sensitivity summaries.
    primary_trades = selected[selected["big_drop_threshold_pct"] == PRIMARY_BIG_DROP_PCT].copy()

    exit_labels = {
        "t_close": "当日收盘",
        "t1_open": "次日开盘",
        "t1_close": "次日收盘",
    }
    summary_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    for scenario in [item.name for item in SCENARIOS]:
        for threshold in BIG_DROP_THRESHOLDS:
            for exit_key, exit_label in exit_labels.items():
                for basis in ("gross", "net"):
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

    # Validation of daily turnover rate against the latest independent quote snapshot.
    con.execute("""
        CREATE TEMP TABLE quote_latest AS
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT
                CAST(symbol AS VARCHAR) AS symbol,
                TRY_CAST(quote_time AS TIMESTAMP) AS quote_time,
                TRY_CAST(float_shares AS DOUBLE) AS quote_float_shares,
                TRY_CAST(float_market_cap AS DOUBLE) AS quote_float_market_cap,
                TRY_CAST(last_price AS DOUBLE) AS quote_last_price,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(symbol AS VARCHAR)
                    ORDER BY TRY_CAST(quote_time AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM quote_raw
            WHERE TRY_CAST(float_shares AS DOUBLE) > 0
        )
        WHERE rn = 1
    """)
    tr_validation = con.execute("""
        WITH latest_bar AS (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    symbol,
                    trade_date,
                    volume,
                    turnover_rate_pct,
                    implied_float_shares,
                    raw_close,
                    amount,
                    activity_ratio,
                    ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date DESC) AS rn
                FROM bars_calc
                WHERE turnover_rate_pct >= 0.50
                  AND implied_float_shares > 0
                  AND amount > 0
                  AND avg_price_inside_ohlc
            )
            WHERE rn = 1
        )
        SELECT
            b.*,
            q.quote_time,
            q.quote_float_shares,
            q.quote_float_market_cap,
            q.quote_last_price,
            ABS(b.implied_float_shares / q.quote_float_shares - 1.0) AS float_relative_error,
            DATE_DIFF('day', b.trade_date, TRY_CAST(q.quote_time AS DATE)) AS quote_date_gap_days
        FROM latest_bar b
        INNER JOIN quote_latest q USING (symbol)
        WHERE q.quote_float_shares > 0
          AND ABS(DATE_DIFF('day', b.trade_date, TRY_CAST(q.quote_time AS DATE))) <= 5
    """).df()

    validation_clean = tr_validation[
        tr_validation["float_relative_error"].replace([np.inf, -np.inf], np.nan).notna()
    ].copy()
    validation_metrics = {
        "validation_pair_count": int(len(validation_clean)),
        "median_float_relative_error": float(validation_clean["float_relative_error"].median()) if len(validation_clean) else np.nan,
        "p75_float_relative_error": float(validation_clean["float_relative_error"].quantile(0.75)) if len(validation_clean) else np.nan,
        "p90_float_relative_error": float(validation_clean["float_relative_error"].quantile(0.90)) if len(validation_clean) else np.nan,
        "share_within_5pct": float((validation_clean["float_relative_error"] <= 0.05).mean()) if len(validation_clean) else np.nan,
        "share_within_10pct": float((validation_clean["float_relative_error"] <= 0.10).mean()) if len(validation_clean) else np.nan,
        "share_within_20pct": float((validation_clean["float_relative_error"] <= 0.20).mean()) if len(validation_clean) else np.nan,
    }

    quality_stats = con.execute("""
        SELECT
            COUNT(*) AS listed_bar_rows,
            SUM(CASE WHEN turnover_rate_pct > 0 THEN 1 ELSE 0 END) AS rows_with_turnover_rate,
            SUM(CASE WHEN amount > 0 AND volume > 0 THEN 1 ELSE 0 END) AS rows_with_amount_and_volume,
            SUM(CASE WHEN avg_price_inside_ohlc THEN 1 ELSE 0 END) AS rows_avg_price_inside_ohlc,
            SUM(CASE WHEN is_delisted THEN 1 ELSE 0 END) AS rows_from_currently_delisted_securities,
            COUNT(DISTINCT symbol) AS listed_symbols,
            COUNT(DISTINCT CASE WHEN is_delisted THEN symbol ELSE NULL END) AS currently_delisted_symbols_included
        FROM bars_calc
        WHERE trade_date >= ?
    """, [analysis_start]).df().iloc[0].to_dict()

    scenario_stats = []
    for scenario in [item.name for item in SCENARIOS]:
        rank_group = rankings[rankings["scenario"] == scenario]
        trade_group = primary_trades[primary_trades["scenario"] == scenario]
        scenario_stats.append({
            "scenario": scenario,
            "ranked_signal_dates": int(rank_group["signal_date"].nunique()),
            "top3_rows": int(len(rank_group)),
            "primary_selected_signals": int(len(trade_group)),
            "primary_selected_signal_days": int(trade_group["signal_date"].nunique()),
            "entry_fill_count": int(trade_group["entry_filled"].sum()),
            "t_close_available_count": int(trade_group["t_close_available"].sum()),
            "t1_open_available_count": int(trade_group["t1_open_available"].sum()),
            "t1_close_available_count": int(trade_group["t1_close_available"].sum()),
        })
    scenario_stats_df = pd.DataFrame(scenario_stats)

    audit_rows = [
        {"metric": "latest_market_date", "value": latest_date.isoformat()},
        {"metric": "analysis_start_calendar_date", "value": analysis_start.isoformat()},
        {"metric": "market_date_count_loaded", "value": market_date_count},
        {"metric": "eligible_signal_rows", "value": int(len(eligible))},
        {"metric": "primary_threshold_pct", "value": PRIMARY_BIG_DROP_PCT},
        *({"metric": key, "value": scalar(value)} for key, value in quality_stats.items()),
        *({"metric": key, "value": scalar(value)} for key, value in validation_metrics.items()),
    ]
    audit_summary = pd.DataFrame(audit_rows)

    top_symbols = (
        primary_trades.groupby(["scenario", "symbol", "company_name"], as_index=False)
        .agg(
            selection_count=("symbol", "size"),
            avg_activity_ratio=("activity_ratio", "mean"),
            avg_drop_pct=("drop_pct", "mean"),
            median_implied_float_mcap=("implied_float_market_cap", "median"),
        )
        .sort_values(["scenario", "selection_count", "avg_activity_ratio"], ascending=[True, False, False])
    )

    # Stable export column order.
    trade_columns = [
        "scenario", "big_drop_threshold_pct", "signal_date", "entry_date", "next_date", "activity_rank",
        "symbol", "company_name", "exchange", "drop_pct", "activity_ratio", "turnover_rate_pct",
        "avg_trade_price", "raw_close", "amount", "volume", "implied_float_shares",
        "implied_float_market_cap", "adjustment_factor", "vendor_change_pct", "change_pct_abs_error",
        "first_traded_date", "delisted_at", "is_delisted", "entry_open", "entry_close", "next_open", "next_close",
        "entry_filled", "t_close_available", "t1_open_available", "t1_close_available",
        "gross_t_close", "net_t_close", "gross_t1_open", "net_t1_open", "gross_t1_close", "net_t1_close",
    ]
    primary_trades = primary_trades[trade_columns].sort_values(["scenario", "entry_date", "activity_rank"])

    ranking_columns = [
        "scenario", "signal_date", "activity_rank", "symbol", "company_name", "exchange", "drop_pct",
        "activity_ratio", "turnover_rate_pct", "avg_trade_price", "raw_close", "amount", "volume",
        "implied_float_shares", "implied_float_market_cap", "first_traded_date", "delisted_at", "is_delisted",
    ]
    rankings = rankings[ranking_columns].sort_values(["scenario", "signal_date", "activity_rank"])

    primary_trades.to_csv(out_dir / "trades_primary_minus8.csv", index=False, encoding="utf-8-sig")
    rankings.to_csv(out_dir / "daily_top3_rankings.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary_all_thresholds.csv", index=False, encoding="utf-8-sig")
    daily_portfolios.to_csv(out_dir / "daily_portfolios.csv", index=False, encoding="utf-8-sig")
    top_symbols.to_csv(out_dir / "top_symbols_primary.csv", index=False, encoding="utf-8-sig")
    audit_summary.to_csv(out_dir / "audit_summary.csv", index=False, encoding="utf-8-sig")
    scenario_stats_df.to_csv(out_dir / "scenario_stats.csv", index=False, encoding="utf-8-sig")
    tr_validation.sort_values("float_relative_error", ascending=False).to_csv(
        out_dir / "turnover_rate_quote_validation.csv", index=False, encoding="utf-8-sig"
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_date.isoformat(),
        "analysis_start_calendar_date": analysis_start.isoformat(),
        "lookback_calendar_days": LOOKBACK_CALENDAR_DAYS,
        "primary_big_drop_threshold_pct": PRIMARY_BIG_DROP_PCT,
        "sensitivity_thresholds_pct": list(BIG_DROP_THRESHOLDS),
        "selection_rule": "rank prior-session activity across the historical eligible universe; keep rank <= 3; then require prior-session close-to-close drop <= threshold",
        "activity_rule": "(daily turnover rate percent / 100) * (daily average trade price / unadjusted close)",
        "activity_identity": "daily dollar amount / (unadjusted close * daily point-in-time implied free-float shares)",
        "implied_float_shares_rule": "daily volume / (daily turnover rate percent / 100)",
        "historical_universe_rule": "security must have a daily bar and the signal date must fall within first_traded_date through delisted_at; currently delisted securities are retained for dates when listed",
        "entry_rule": "next reliable U.S. market session open",
        "exit_rules": exit_labels,
        "one_way_cost": ONE_WAY_COST,
        "tradable_filters": {
            "min_unadjusted_signal_close_usd": TRADABLE_MIN_RAW_CLOSE,
            "min_signal_dollar_volume_usd": TRADABLE_MIN_DOLLAR_VOLUME,
            "min_signal_implied_float_market_cap_usd": TRADABLE_MIN_FLOAT_MCAP,
        },
        "quality_rules": {
            "daily_amount_and_volume_must_be_positive": True,
            "daily_turnover_rate_must_be_positive": True,
            "average_trade_price_must_lie_within_raw_daily_low_high_tolerance": AVG_PRICE_TOLERANCE,
            "signal_ranking_is_completed_before_entry_or_exit_availability_checks": True,
            "same_day_exit_does_not_require_a_following_day_bar": True,
        },
        "validation_metrics": {key: scalar(value) for key, value in validation_metrics.items()},
        "limitations": [
            "The daily turnover-rate field is vendor-supplied. It is validated against the latest independent quote float snapshot, but the vendor's complete historical float construction methodology is not publicly documented.",
            "Turnover rate is reported with finite precision, so implied free-float shares have rounding noise; ranking uses the rate itself rather than a rounded share estimate.",
            "Daily bars cannot model opening-auction queue position, volatility halts, bid-ask spread, partial fills, or market impact.",
            "Net returns assume 25 bps one-way cost; real microcap execution can be materially worse.",
            "Daily equal-weight compounding for next-day exits is diagnostic because positions initiated on adjacent days can overlap.",
        ],
        "sources": manifest,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, default=scalar, allow_nan=False)

    print("\nTURNOVER VALIDATION")
    print(json.dumps({key: scalar(value) for key, value in validation_metrics.items()}, indent=2))
    print("\nSCENARIO STATS")
    print(scenario_stats_df.to_string(index=False))
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
        print(f"ACCURATE BACKTEST FAILED: {exc}", file=sys.stderr, flush=True)
        raise
