#!/usr/bin/env python3
"""Rough six-month U.S. stock free-float activity backtest.

Signal date S uses only information known after S close. Entry is the next U.S.
regular-session open. Free-float shares are taken from the latest available quote
snapshot, so this is not a point-in-time float backtest.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

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
BIG_DROP_PCT = -8.0
ONE_WAY_COST = 0.0025
LOOKBACK_DAYS = 183
MIN_DAILY_UNIVERSE = 300
MIN_PRICE = 1.0
MIN_DOLLAR_VOLUME = 1_000_000.0
MIN_FLOAT_MCAP = 20_000_000.0


def session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    s = requests.Session()
    s.headers.update({"User-Agent": "rough-us-free-float-backtest/1.0"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def download(s: requests.Session, url: str, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return path.stat().st_size
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with s.get(url, stream=True, allow_redirects=True, timeout=(30, 900)) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if tmp.stat().st_size <= 1000:
                raise RuntimeError(f"download too small: {tmp.stat().st_size} bytes")
            tmp.replace(path)
            return path.stat().st_size
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise
            time.sleep(attempt * 5)
    raise AssertionError("unreachable")


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if frame.empty or column not in frame:
        values = pd.Series(dtype=float)
    else:
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    wins = values[values > 0]
    losses = values[values < 0]
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    payoff = avg_win / abs(avg_loss) if len(wins) and len(losses) and avg_loss else np.nan
    profit_factor = wins.sum() / abs(losses.sum()) if len(wins) and len(losses) and losses.sum() else np.nan
    return {
        "trade_count": int(len(values)),
        "win_rate": float((values > 0).mean()) if len(values) else np.nan,
        "avg_return": float(values.mean()) if len(values) else np.nan,
        "median_return": float(values.median()) if len(values) else np.nan,
        "avg_win": scalar(avg_win),
        "avg_loss": scalar(avg_loss),
        "payoff_ratio": scalar(payoff),
        "profit_factor": scalar(profit_factor),
        "best_return": float(values.max()) if len(values) else np.nan,
        "worst_return": float(values.min()) if len(values) else np.nan,
    }


def pick(base: pd.DataFrame, universe_rule: str, method: str) -> pd.DataFrame:
    data = base.copy()
    data["universe_rule"] = universe_rule
    data["selection_method"] = method
    if method == "全市场活跃前三后筛大跌":
        data = data.sort_values(
            ["signal_date", "activity_ratio", "signal_dollar_volume"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        data["rank"] = data.groupby("signal_date").cumcount() + 1
        data = data[(data["rank"] <= 3) & (data["drop_pct"] <= BIG_DROP_PCT)]
    elif method == "大跌池内活跃前三":
        data = data[data["drop_pct"] <= BIG_DROP_PCT].copy()
        data = data.sort_values(
            ["signal_date", "activity_ratio", "signal_dollar_volume"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        data["rank"] = data.groupby("signal_date").cumcount() + 1
        data = data[data["rank"] <= 3]
    else:
        raise ValueError(method)
    data["scenario"] = data["universe_rule"] + "｜" + data["selection_method"]
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".tmp_backtest_data")
    parser.add_argument("--out-dir", default="backtest_output")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s = session()
    files: dict[str, Path] = {}
    manifest: dict[str, Any] = {}
    for name, url in SOURCES.items():
        path = data_dir / f"{name}.parquet"
        size = download(s, url, path)
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
        CREATE TEMP TABLE info AS
        WITH normalized AS (
            SELECT
                NULLIF(CAST(ticker AS VARCHAR), '') AS symbol,
                COALESCE(CAST(short_name AS VARCHAR), CAST(long_name AS VARCHAR), CAST(ticker AS VARCHAR)) AS company_name,
                COALESCE(CAST(full_exchange_name AS VARCHAR), '') AS exchange,
                UPPER(COALESCE(CAST(quote_type AS VARCHAR), '')) AS quote_type,
                LOWER(COALESCE(CAST(market AS VARCHAR), '')) AS market,
                COALESCE(TRY_CAST(is_delisted AS BOOLEAN), FALSE) AS is_delisted,
                ROW_NUMBER() OVER (
                    PARTITION BY ticker
                    ORDER BY TRY_CAST(first_traded_date AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM info_raw
        )
        SELECT symbol, company_name, exchange
        FROM normalized
        WHERE rn = 1
          AND symbol IS NOT NULL
          AND market = 'us'
          AND quote_type = 'EQUITY'
          AND NOT is_delisted
          AND REGEXP_MATCHES(UPPER(exchange), 'NYSE|NASDAQ|AMEX|ARCA|CBOE|BATS')
          AND NOT REGEXP_MATCHES(
              UPPER(COALESCE(company_name, '')),
              '(^|[^A-Z])(WARRANTS?|RIGHTS?|UNITS?|PREFERRED|PREF)([^A-Z]|$)'
          )
    """)

    con.execute("""
        CREATE TEMP TABLE quote AS
        WITH valid AS (
            SELECT
                CAST(symbol AS VARCHAR) AS symbol,
                TRY_CAST(quote_time AS TIMESTAMP) AS quote_snapshot,
                TRY_CAST(float_shares AS DOUBLE) AS quoted_float_shares,
                TRY_CAST(float_market_cap AS DOUBLE) AS quoted_float_market_cap,
                TRY_CAST(last_price AS DOUBLE) AS last_price,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol
                    ORDER BY TRY_CAST(quote_time AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM quote_raw
            WHERE symbol IS NOT NULL
              AND (
                  TRY_CAST(float_shares AS DOUBLE) > 0
                  OR (TRY_CAST(float_market_cap AS DOUBLE) > 0 AND TRY_CAST(last_price AS DOUBLE) > 0)
              )
        )
        SELECT
            symbol,
            quote_snapshot,
            CASE
                WHEN quoted_float_shares > 0 THEN quoted_float_shares
                ELSE quoted_float_market_cap / last_price
            END AS float_shares,
            CASE
                WHEN quoted_float_shares > 0 THEN 'float_shares'
                ELSE 'float_market_cap/last_price'
            END AS float_source
        FROM valid
        WHERE rn = 1
    """)

    con.execute("""
        CREATE TEMP TABLE universe AS
        SELECT i.*, q.quote_snapshot, q.float_shares, q.float_source
        FROM info i
        INNER JOIN quote q USING (symbol)
        WHERE q.float_shares > 0
    """)
    universe_count = int(con.execute("SELECT COUNT(*) FROM universe").fetchone()[0])
    if universe_count < 500:
        raise RuntimeError(f"eligible universe too small: {universe_count}")

    latest_date = con.execute("""
        SELECT MAX(TRY_CAST(k.bar_time AS DATE))
        FROM kline_raw k
        INNER JOIN universe u USING (symbol)
        WHERE UPPER(COALESCE(CAST(k.kline_t AS VARCHAR), '1D')) = '1D'
          AND TRY_CAST(k.close AS DOUBLE) > 0
    """).fetchone()[0]
    if latest_date is None:
        raise RuntimeError("no joined U.S. daily bars")
    if isinstance(latest_date, datetime):
        latest_date = latest_date.date()
    analysis_start = latest_date - timedelta(days=LOOKBACK_DAYS)
    load_start = analysis_start - timedelta(days=14)

    con.execute("""
        CREATE TEMP TABLE bars AS
        WITH source AS (
            SELECT
                k.symbol,
                TRY_CAST(k.bar_time AS DATE) AS trade_date,
                TRY_CAST(k.open AS DOUBLE) AS open,
                TRY_CAST(k.close AS DOUBLE) AS close,
                TRY_CAST(k.vol AS DOUBLE) AS volume,
                TRY_CAST(k.amount AS DOUBLE) AS amount,
                COALESCE(TRY_CAST(k.splits AS DOUBLE), 0.0) AS split_value,
                ROW_NUMBER() OVER (
                    PARTITION BY k.symbol, TRY_CAST(k.bar_time AS DATE)
                    ORDER BY TRY_CAST(k.bar_time AS TIMESTAMP) DESC NULLS LAST
                ) AS rn
            FROM kline_raw k
            INNER JOIN universe u USING (symbol)
            WHERE UPPER(COALESCE(CAST(k.kline_t AS VARCHAR), '1D')) = '1D'
              AND TRY_CAST(k.bar_time AS DATE) BETWEEN ? AND ?
        )
        SELECT symbol, trade_date, open, close, volume, amount, split_value
        FROM source
        WHERE rn = 1
          AND trade_date IS NOT NULL
          AND open > 0 AND close > 0 AND volume >= 0
    """, [load_start, latest_date])

    con.execute("""
        CREATE TEMP TABLE market_dates AS
        SELECT trade_date, COUNT(*) AS bar_count
        FROM bars
        GROUP BY trade_date
        HAVING COUNT(*) >= ?
        ORDER BY trade_date
    """, [MIN_DAILY_UNIVERSE])
    market_days = int(con.execute("SELECT COUNT(*) FROM market_dates").fetchone()[0])
    if market_days < 80:
        raise RuntimeError(f"too few market dates: {market_days}")

    con.execute("""
        CREATE TEMP TABLE calendar_map AS
        SELECT
            trade_date AS signal_date,
            LEAD(trade_date, 1) OVER (ORDER BY trade_date) AS entry_date,
            LEAD(trade_date, 2) OVER (ORDER BY trade_date) AS next_date
        FROM market_dates
    """)
    con.execute("""
        CREATE TEMP TABLE calc AS
        SELECT
            *,
            LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS previous_close
        FROM bars
    """)

    base = con.execute("""
        SELECT
            b.trade_date AS signal_date,
            c.entry_date,
            c.next_date,
            b.symbol,
            u.company_name,
            u.exchange,
            u.float_shares,
            u.float_source,
            u.quote_snapshot,
            b.close AS signal_close,
            b.volume AS signal_volume,
            CASE WHEN b.amount > 0 THEN b.amount ELSE b.volume * b.close END AS signal_dollar_volume,
            b.close * u.float_shares AS signal_float_market_cap,
            (CASE WHEN b.amount > 0 THEN b.amount ELSE b.volume * b.close END)
                / (b.close * u.float_shares) AS activity_ratio,
            100.0 * (b.close / b.previous_close - 1.0) AS drop_pct,
            e.open AS entry_open,
            e.close AS entry_close,
            n.open AS next_open,
            n.close AS next_close
        FROM calc b
        INNER JOIN calendar_map c ON c.signal_date = b.trade_date
        INNER JOIN bars e ON e.symbol = b.symbol AND e.trade_date = c.entry_date
        INNER JOIN bars n ON n.symbol = b.symbol AND n.trade_date = c.next_date
        INNER JOIN universe u ON u.symbol = b.symbol
        WHERE c.entry_date >= ?
          AND c.next_date IS NOT NULL
          AND b.previous_close > 0
          AND b.close > 0
          AND u.float_shares > 0
          AND (CASE WHEN b.amount > 0 THEN b.amount ELSE b.volume * b.close END) > 0
          AND e.open > 0 AND e.close > 0 AND n.open > 0 AND n.close > 0
          AND ABS(b.split_value) < 1e-12
          AND ABS(e.split_value) < 1e-12
          AND ABS(n.split_value) < 1e-12
    """, [analysis_start]).df()
    if base.empty:
        raise RuntimeError("no valid symbol-date rows")

    tradable = base[
        (base["signal_close"] >= MIN_PRICE)
        & (base["signal_dollar_volume"] >= MIN_DOLLAR_VOLUME)
        & (base["signal_float_market_cap"] >= MIN_FLOAT_MCAP)
    ].copy()

    methods = ["全市场活跃前三后筛大跌", "大跌池内活跃前三"]
    scenario_defs: list[tuple[str, str, str]] = []
    selected: list[pd.DataFrame] = []
    for universe_rule, frame in [("原始口径", base), ("可交易口径", tradable)]:
        for method in methods:
            scenario = f"{universe_rule}｜{method}"
            scenario_defs.append((scenario, universe_rule, method))
            chosen = pick(frame, universe_rule, method)
            if not chosen.empty:
                selected.append(chosen)
    trades = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if not trades.empty:
        trades["gross_t_close"] = trades["entry_close"] / trades["entry_open"] - 1.0
        trades["gross_t1_open"] = trades["next_open"] / trades["entry_open"] - 1.0
        trades["gross_t1_close"] = trades["next_close"] / trades["entry_open"] - 1.0
        factor = (1.0 - ONE_WAY_COST) / (1.0 + ONE_WAY_COST)
        for suffix in ("t_close", "t1_open", "t1_close"):
            trades[f"net_{suffix}"] = (1.0 + trades[f"gross_{suffix}"]) * factor - 1.0
        for col in ("signal_date", "entry_date", "next_date"):
            trades[col] = pd.to_datetime(trades[col]).dt.date
        trades["quote_snapshot"] = pd.to_datetime(trades["quote_snapshot"], errors="coerce")

    exits = {"t_close": "当日收盘", "t1_open": "次日开盘", "t1_close": "次日收盘"}
    summary_rows: list[dict[str, Any]] = []
    cohort_frames: list[pd.DataFrame] = []
    for scenario, universe_rule, method in scenario_defs:
        group = trades[trades["scenario"] == scenario].copy() if not trades.empty else pd.DataFrame()
        signal_days = int(group["signal_date"].nunique()) if not group.empty else 0
        for suffix, exit_rule in exits.items():
            for basis in ("gross", "net"):
                col = f"{basis}_{suffix}"
                if group.empty:
                    cohort = pd.DataFrame(columns=["entry_date", "cohort_return"])
                else:
                    cohort = group.groupby("entry_date", as_index=False)[col].mean().rename(columns={col: "cohort_return"})
                    cohort = cohort.sort_values("entry_date")
                    cohort["scenario"] = scenario
                    cohort["exit_rule"] = exit_rule
                    cohort["return_basis"] = "毛收益" if basis == "gross" else "净收益"
                    cohort["cohort_win"] = cohort["cohort_return"] > 0
                    cohort["cohort_cumulative"] = (1.0 + cohort["cohort_return"]).cumprod() - 1.0
                    cohort_frames.append(cohort)
                summary_rows.append({
                    "scenario": scenario,
                    "universe_rule": universe_rule,
                    "selection_method": method,
                    "exit_rule": exit_rule,
                    "return_basis": "毛收益" if basis == "gross" else "净收益",
                    "signal_days": signal_days,
                    "cohort_win_rate": float((cohort["cohort_return"] > 0).mean()) if len(cohort) else np.nan,
                    "cohort_avg_return": float(cohort["cohort_return"].mean()) if len(cohort) else np.nan,
                    "cohort_compounded": float(((1.0 + cohort["cohort_return"]).prod() - 1.0)) if len(cohort) else np.nan,
                    **metrics(group, col),
                })

    summary = pd.DataFrame(summary_rows)
    cohorts = pd.concat(cohort_frames, ignore_index=True) if cohort_frames else pd.DataFrame()
    if trades.empty:
        top_symbols = pd.DataFrame(columns=["scenario", "symbol", "company_name", "selection_count", "avg_activity", "avg_drop_pct"])
    else:
        top_symbols = (
            trades.groupby(["scenario", "symbol", "company_name"], as_index=False)
            .agg(selection_count=("symbol", "size"), avg_activity=("activity_ratio", "mean"), avg_drop_pct=("drop_pct", "mean"))
            .sort_values(["scenario", "selection_count", "avg_activity"], ascending=[True, False, False])
        )

    universe_stats = pd.DataFrame([
        {"metric": "eligible_us_common_equities_with_float", "value": universe_count},
        {"metric": "market_dates_loaded", "value": market_days},
        {"metric": "valid_symbol_dates", "value": int(len(base))},
        {"metric": "tradable_symbol_dates", "value": int(len(tradable))},
        {"metric": "selected_trade_rows_all_scenarios", "value": int(len(trades))},
        {"metric": "latest_market_date", "value": latest_date.isoformat()},
        {"metric": "entry_window_start_calendar_date", "value": analysis_start.isoformat()},
    ])

    if not trades.empty:
        order = [
            "scenario", "universe_rule", "selection_method", "signal_date", "entry_date", "next_date", "rank",
            "symbol", "company_name", "exchange", "drop_pct", "activity_ratio", "signal_close", "signal_volume",
            "signal_dollar_volume", "float_shares", "signal_float_market_cap", "float_source", "quote_snapshot",
            "entry_open", "entry_close", "next_open", "next_close", "gross_t_close", "net_t_close",
            "gross_t1_open", "net_t1_open", "gross_t1_close", "net_t1_close",
        ]
        trades = trades[order].sort_values(["scenario", "entry_date", "rank"])

    trades.to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    cohorts.to_csv(out_dir / "cohorts.csv", index=False, encoding="utf-8-sig")
    top_symbols.to_csv(out_dir / "top_symbols.csv", index=False, encoding="utf-8-sig")
    universe_stats.to_csv(out_dir / "universe_stats.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_market_date": latest_date.isoformat(),
        "entry_window_start_calendar_date": analysis_start.isoformat(),
        "lookback_calendar_days": LOOKBACK_DAYS,
        "big_drop_threshold_pct": BIG_DROP_PCT,
        "activity_rule": "signal-day dollar volume / (signal close * latest available free-float shares)",
        "entry_rule": "next regular-session open",
        "exit_rules": list(exits.values()),
        "one_way_cost": ONE_WAY_COST,
        "tradable_filters": {
            "min_signal_price_usd": MIN_PRICE,
            "min_signal_dollar_volume_usd": MIN_DOLLAR_VOLUME,
            "min_signal_float_market_cap_usd": MIN_FLOAT_MCAP,
        },
        "limitations": [
            "Free-float shares are latest-snapshot values, not historical point-in-time values.",
            "The current instrument master introduces survivorship bias and can omit stocks delisted during the window.",
            "Daily bars do not model opening-auction constraints, halts, bid-ask spread, partial fills, or market impact.",
            "Net returns assume 25 bps one-way cost; microcap slippage can be materially larger.",
            "Cohort compounding is diagnostic, not a fully funded equity curve, because next-day positions can overlap.",
        ],
        "sources": manifest,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2, default=scalar, allow_nan=False)

    print("\nSUMMARY")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BACKTEST FAILED: {exc}", file=sys.stderr, flush=True)
        raise
