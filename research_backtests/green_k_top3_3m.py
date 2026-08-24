#!/usr/bin/env python3
"""Three-month backtest for the clarified Top-3 green-candle entry logic.

Primary rule
------------
* On signal day D, rank the SEC point-in-time free-float covered U.S. universe by
  dollar volume / free-float market cap.
* Take the market-wide activity Top 3 without a return gate.
* Keep only names whose D daily candle is green in the Chinese-app sense:
  close < open.
* Buy the retained names equally at D+1 regular-session open.
* Use the previously selected exit policy: +10% take profit, -15% stop loss,
  otherwise D+3 close; if both barriers are touched in one daily bar, stop first.
* One batch at a time: while a prior all-in batch is still open, later signals are
  recorded but not funded. This avoids hidden leverage.

The script also reports close-to-close-negative and <=-8% variants for audit.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import exchange_calendars as xcals
import pandas as pd

import us_daily_picker.picker as daily

ONE_WAY_COST = 0.0025
TAKE_PROFIT = 0.10
STOP_LOSS = 0.15
MAX_HOLD_OFFSET = 3


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


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def xnys_sessions(start: date, end: date) -> list[date]:
    calendar = xcals.get_calendar("XNYS")
    return [item.date() for item in calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))]


def previous_session_map(entry_start: date, end_date: date) -> pd.DataFrame:
    sessions = xnys_sessions(entry_start - timedelta(days=20), end_date + timedelta(days=7))
    rows: list[dict[str, date]] = []
    for index in range(1, len(sessions)):
        entry = sessions[index]
        if entry_start <= entry <= end_date:
            rows.append({"signal_date": sessions[index - 1], "entry_date": entry})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No XNYS signal/entry dates in requested window")
    return frame


def build_rankings(
    *,
    info_path: Path,
    kline_path: Path,
    sec_facts: pd.DataFrame,
    signal_calendar: pd.DataFrame,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    first_signal = min(signal_calendar["signal_date"])
    history_start = first_signal - timedelta(
        days=daily.MAX_PUBLIC_FLOAT_AGE_DAYS
        + daily.MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS
        + daily.BAR_HISTORY_BUFFER_DAYS
    )

    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute(f"CREATE VIEW info_raw AS SELECT * FROM read_parquet('{qpath(info_path)}')")
    con.execute(f"CREATE VIEW kline_raw AS SELECT * FROM read_parquet('{qpath(kline_path)}')")
    con.register("sec_float_facts_raw", sec_facts)
    con.register("signal_calendar_raw", signal_calendar)

    latest_market_date = con.execute(
        """
        SELECT MAX(TRY_CAST(bar_time AS DATE))
        FROM kline_raw
        WHERE UPPER(COALESCE(CAST(kline_t AS VARCHAR), '1D')) = '1D'
          AND TRY_CAST(close AS DOUBLE) > 0
        """
    ).fetchone()[0]
    if isinstance(latest_market_date, datetime):
        latest_market_date = latest_market_date.date()
    if latest_market_date is None or latest_market_date < end_date:
        raise RuntimeError(f"Daily source ends at {latest_market_date}, requested {end_date}")

    con.execute(
        f"""
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
          AND REGEXP_MATCHES(UPPER(COALESCE(CAST(full_exchange_name AS VARCHAR), '')), '{daily.EXCHANGE_RE}')
          AND NOT REGEXP_MATCHES(
                UPPER(COALESCE(CAST(short_name AS VARCHAR), '') || ' ' ||
                      COALESCE(CAST(long_name AS VARCHAR), '') || ' ' ||
                      COALESCE(CAST(type_disp AS VARCHAR), '')),
                '{daily.EXCLUDED_INSTRUMENT_RE}'
              )
        """
    )

    con.execute(
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
                CASE WHEN TRY_CAST(adj_factor_cum AS DOUBLE) > 0
                     THEN TRY_CAST(adj_factor_cum AS DOUBLE) ELSE 1.0 END AS adjustment_factor,
                CASE WHEN TRY_CAST(splits AS DOUBLE) > 0
                     THEN TRY_CAST(splits AS DOUBLE) ELSE 1.0 END AS split_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY UPPER(REPLACE(CAST(symbol AS VARCHAR), '.', '-')),
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
          AND adjusted_open > 0 AND adjusted_high > 0
          AND adjusted_low > 0 AND adjusted_close > 0
          AND volume >= 0
        """,
        [history_start, end_date],
    )

    con.execute(
        """
        CREATE TEMP TABLE bars_listed AS
        SELECT * EXCLUDE (interval_rank)
        FROM (
            SELECT
                bars.*, info.company_name, info.exchange,
                info.first_traded_date, info.delisted_at,
                info.is_delisted, info.country,
                ROW_NUMBER() OVER (
                    PARTITION BY bars.symbol, bars.trade_date
                    ORDER BY info.first_traded_date DESC NULLS LAST,
                             info.delisted_at DESC NULLS LAST
                ) AS interval_rank
            FROM bars_source bars
            INNER JOIN info_intervals info
              ON info.ticker = bars.symbol
             AND (info.first_traded_date IS NULL OR bars.trade_date >= info.first_traded_date)
             AND (info.delisted_at IS NULL OR bars.trade_date <= info.delisted_at)
        )
        WHERE interval_rank = 1
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE bars_calc AS
        WITH raw AS (
            SELECT
                *,
                adjusted_open / adjustment_factor AS raw_open,
                adjusted_high / adjustment_factor AS raw_high,
                adjusted_low / adjustment_factor AS raw_low,
                adjusted_close / adjustment_factor AS raw_close,
                CASE WHEN amount > 0 AND volume > 0 THEN amount / volume ELSE NULL END AS average_trade_price,
                EXP(SUM(LN(split_ratio)) OVER (
                    PARTITION BY symbol ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )) AS cumulative_split_factor,
                LAG(adjusted_close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prior_adjusted_close
            FROM bars_listed
        )
        SELECT
            *,
            CASE WHEN prior_adjusted_close > 0
                 THEN 100.0 * (adjusted_close / prior_adjusted_close - 1.0)
                 ELSE NULL END AS close_to_close_pct,
            adjusted_close < adjusted_open AS green_body,
            CASE WHEN average_trade_price > 0 AND raw_low > 0 AND raw_high > 0
                       AND average_trade_price BETWEEN raw_low * ? AND raw_high * ?
                 THEN TRUE ELSE FALSE END AS average_price_inside_raw_ohlc
        FROM raw
        """,
        [1.0 - daily.AVG_PRICE_TOLERANCE, 1.0 + daily.AVG_PRICE_TOLERANCE],
    )

    con.execute(
        """
        CREATE TEMP TABLE priced_sec_facts AS
        SELECT * EXCLUDE (price_rank)
        FROM (
            SELECT
                facts.*,
                bars.trade_date AS measurement_price_date,
                bars.raw_close AS measurement_raw_close,
                bars.cumulative_split_factor AS measurement_split_factor,
                facts.public_float_value_usd / bars.raw_close AS measurement_float_shares,
                DATE_DIFF('day', bars.trade_date, facts.measurement_date) AS measurement_price_lag_days,
                ROW_NUMBER() OVER (
                    PARTITION BY facts.ticker, facts.measurement_date, facts.filed_date, facts.accession
                    ORDER BY bars.trade_date DESC
                ) AS price_rank
            FROM sec_float_facts_raw facts
            INNER JOIN bars_calc bars
              ON bars.symbol = facts.ticker
             AND bars.trade_date <= facts.measurement_date
             AND bars.trade_date >= facts.measurement_date - INTERVAL ? DAY
             AND bars.raw_close > 0
        )
        WHERE price_rank = 1 AND measurement_float_shares > 0
        """,
        [daily.MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS],
    )

    con.execute(
        """
        CREATE TEMP TABLE signal_base AS
        SELECT bars.*, cal.entry_date
        FROM bars_calc bars
        INNER JOIN signal_calendar_raw cal ON cal.signal_date = bars.trade_date
        WHERE bars.amount > 0 AND bars.volume > 0 AND bars.raw_close > 0
          AND bars.average_price_inside_raw_ohlc
          AND bars.close_to_close_pct IS NOT NULL
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE covered_signal_rows AS
        SELECT * EXCLUDE (fact_rank)
        FROM (
            SELECT
                bars.*,
                facts.cik, facts.sec_entity_name, facts.public_float_value_usd,
                facts.measurement_date, facts.measurement_price_date,
                facts.measurement_raw_close, facts.measurement_split_factor,
                facts.measurement_float_shares, facts.measurement_price_lag_days,
                facts.filed_date, facts.form, facts.accession, facts.sec_source_url,
                DATE_DIFF('day', facts.measurement_date, bars.trade_date) AS public_float_age_days,
                DATE_DIFF('day', facts.filed_date, bars.trade_date) AS filing_known_days,
                facts.measurement_float_shares
                    * (bars.cumulative_split_factor / facts.measurement_split_factor)
                    AS signal_float_shares,
                bars.raw_close * facts.measurement_float_shares
                    * (bars.cumulative_split_factor / facts.measurement_split_factor)
                    AS signal_float_market_cap,
                bars.amount / (
                    bars.raw_close * facts.measurement_float_shares
                    * (bars.cumulative_split_factor / facts.measurement_split_factor)
                ) AS activity_ratio,
                ROW_NUMBER() OVER (
                    PARTITION BY bars.symbol, bars.trade_date
                    ORDER BY facts.measurement_date DESC, facts.filed_date DESC, facts.accession DESC
                ) AS fact_rank
            FROM signal_base bars
            INNER JOIN priced_sec_facts facts
              ON facts.ticker = bars.symbol
             AND facts.filed_date < bars.trade_date
             AND facts.measurement_date <= bars.trade_date
             AND DATE_DIFF('day', facts.measurement_date, bars.trade_date) BETWEEN 0 AND ?
        )
        WHERE fact_rank = 1
          AND signal_float_shares > 0
          AND signal_float_market_cap > 0
          AND activity_ratio > 0
        """,
        [daily.MAX_PUBLIC_FLOAT_AGE_DAYS],
    )

    ranked = con.execute(
        """
        WITH tradable AS (
            SELECT *
            FROM covered_signal_rows
            WHERE raw_close >= ?
              AND amount >= ?
              AND signal_float_market_cap >= ?
        )
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY trade_date
            ORDER BY activity_ratio DESC, amount DESC, symbol ASC
        ) AS activity_rank
        FROM tradable
        ORDER BY trade_date, activity_rank
        """,
        [daily.MIN_RAW_CLOSE, daily.MIN_DOLLAR_VOLUME, daily.MIN_FLOAT_MARKET_CAP],
    ).df()

    top3 = ranked[ranked["activity_rank"] <= 3].copy()
    mrna = ranked[ranked["symbol"] == "MRNA"].copy()
    bars_window = con.execute(
        """
        SELECT symbol, trade_date, adjusted_open, adjusted_high, adjusted_low, adjusted_close
        FROM bars_calc
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY symbol, trade_date
        """,
        [min(signal_calendar["entry_date"]), end_date],
    ).df()

    for frame in (top3, mrna, bars_window):
        for column in ("trade_date", "entry_date", "measurement_date", "measurement_price_date", "filed_date"):
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date

    diagnostics = {
        "latest_market_date": latest_market_date,
        "first_signal_date": first_signal,
        "last_signal_date": max(signal_calendar["signal_date"]),
        "entry_start": min(signal_calendar["entry_date"]),
        "entry_end": max(signal_calendar["entry_date"]),
        "ranked_rows": len(ranked),
        "top3_rows": len(top3),
        "top3_signal_days": int(top3["trade_date"].nunique()),
    }
    return top3, mrna, bars_window, diagnostics


@dataclass
class LegExit:
    symbol: str
    entry_date: date
    entry_price: float
    exit_date: date
    exit_price: float
    exit_reason: str
    holding_offset: int
    net_return: float


def simulate_leg(
    *,
    symbol: str,
    entry_date: date,
    entry_price: float,
    bars: dict[tuple[str, date], tuple[float, float, float, float]],
    sessions: list[date],
    end_date: date,
    policy: str,
) -> LegExit | None:
    if entry_date not in sessions:
        return None
    if (symbol, entry_date) not in bars:
        return None
    start = sessions.index(entry_date)

    if policy == "same_day_close":
        close_price = bars[(symbol, entry_date)][3]
        net = (close_price * (1.0 - ONE_WAY_COST)) / (entry_price * (1.0 + ONE_WAY_COST)) - 1.0
        return LegExit(symbol, entry_date, entry_price, entry_date, close_price, "same_day_close", 0, net)

    target = entry_price * (1.0 + TAKE_PROFIT)
    stop = entry_price * (1.0 - STOP_LOSS)
    latest_bar: tuple[date, tuple[float, float, float, float], int] | None = None
    for offset in range(MAX_HOLD_OFFSET + 1):
        if start + offset >= len(sessions):
            break
        trade_date = sessions[start + offset]
        if trade_date > end_date:
            break
        bar = bars.get((symbol, trade_date))
        if bar is None:
            continue
        latest_bar = (trade_date, bar, offset)
        open_px, high_px, low_px, close_px = bar
        if open_px <= stop:
            exit_px, reason = open_px, "gap_stop"
        elif open_px >= target:
            exit_px, reason = open_px, "gap_target"
        elif low_px <= stop:
            exit_px, reason = stop, "stop_loss"
        elif high_px >= target:
            exit_px, reason = target, "take_profit"
        elif offset == MAX_HOLD_OFFSET:
            exit_px, reason = close_px, "max_hold_close"
        else:
            continue
        net = (exit_px * (1.0 - ONE_WAY_COST)) / (entry_price * (1.0 + ONE_WAY_COST)) - 1.0
        return LegExit(symbol, entry_date, entry_price, trade_date, exit_px, reason, offset, net)

    if latest_bar is None:
        return None
    trade_date, bar, offset = latest_bar
    exit_px = bar[3]
    net = (exit_px * (1.0 - ONE_WAY_COST)) / (entry_price * (1.0 + ONE_WAY_COST)) - 1.0
    return LegExit(symbol, entry_date, entry_price, trade_date, exit_px, "end_mark_to_market", offset, net)


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def run_portfolio(
    *,
    top3: pd.DataFrame,
    bars_window: pd.DataFrame,
    sessions: list[date],
    initial_capital: float,
    mode: str,
    policy: str,
    end_date: date,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = top3.copy()
    if mode == "green_body":
        condition = frame["green_body"].fillna(False)
    elif mode == "negative_close_to_close":
        condition = frame["close_to_close_pct"] < 0
    elif mode == "green_and_negative":
        condition = frame["green_body"].fillna(False) & (frame["close_to_close_pct"] < 0)
    elif mode == "drop8":
        condition = frame["close_to_close_pct"] <= -8.0
    else:
        raise ValueError(mode)
    selected = frame[condition].copy()

    bars = {
        (row.symbol, row.trade_date): (
            float(row.adjusted_open), float(row.adjusted_high),
            float(row.adjusted_low), float(row.adjusted_close),
        )
        for row in bars_window.itertuples(index=False)
    }

    capital = float(initial_capital)
    last_batch_exit: date | None = None
    batch_rows: list[dict[str, Any]] = []
    leg_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    equity_points: list[dict[str, Any]] = []

    for entry_date, group in selected.groupby("entry_date", sort=True):
        entry_date = pd.Timestamp(entry_date).date() if not isinstance(entry_date, date) else entry_date
        symbols = group.sort_values("activity_rank")["symbol"].tolist()
        if last_batch_exit is not None and entry_date <= last_batch_exit:
            skipped_rows.append({
                "entry_date": entry_date,
                "signal_date": min(group["trade_date"]),
                "symbols": ",".join(symbols),
                "reason": "prior_batch_still_open",
                "prior_batch_exit_date": last_batch_exit,
            })
            continue

        executable: list[tuple[Any, float]] = []
        for row in group.sort_values("activity_rank").itertuples(index=False):
            bar = bars.get((row.symbol, entry_date))
            if bar is not None and bar[0] > 0:
                executable.append((row, float(bar[0])))
        if not executable:
            skipped_rows.append({
                "entry_date": entry_date,
                "signal_date": min(group["trade_date"]),
                "symbols": ",".join(symbols),
                "reason": "no_entry_open",
                "prior_batch_exit_date": last_batch_exit,
            })
            continue

        capital_before = capital
        allocation = capital_before / len(executable)
        exits: list[LegExit] = []
        for row, entry_price in executable:
            leg = simulate_leg(
                symbol=row.symbol,
                entry_date=entry_date,
                entry_price=entry_price,
                bars=bars,
                sessions=sessions,
                end_date=end_date,
                policy=policy,
            )
            if leg is not None:
                exits.append(leg)
        if not exits:
            skipped_rows.append({
                "entry_date": entry_date,
                "signal_date": min(group["trade_date"]),
                "symbols": ",".join(symbols),
                "reason": "no_exit_or_mark",
                "prior_batch_exit_date": last_batch_exit,
            })
            continue

        # Reallocate equally over the actually executable/simulatable legs.
        allocation = capital_before / len(exits)
        capital_after = sum(allocation * (1.0 + leg.net_return) for leg in exits)
        batch_exit = max(leg.exit_date for leg in exits)
        batch_return = capital_after / capital_before - 1.0
        batch_rows.append({
            "signal_date": min(group["trade_date"]),
            "entry_date": entry_date,
            "batch_exit_date": batch_exit,
            "symbols": ",".join(leg.symbol for leg in exits),
            "leg_count": len(exits),
            "capital_before": capital_before,
            "capital_after": capital_after,
            "batch_net_return": batch_return,
        })
        for leg in exits:
            leg_rows.append({
                "signal_date": min(group["trade_date"]),
                "entry_date": entry_date,
                "symbol": leg.symbol,
                "entry_price": leg.entry_price,
                "exit_date": leg.exit_date,
                "exit_price": leg.exit_price,
                "exit_reason": leg.exit_reason,
                "holding_offset": leg.holding_offset,
                "net_return": leg.net_return,
                "allocation": allocation,
                "pnl": allocation * leg.net_return,
            })

        start_index = sessions.index(entry_date)
        end_index = sessions.index(batch_exit)
        for idx in range(start_index, end_index + 1):
            d = sessions[idx]
            value = 0.0
            for leg in exits:
                if d >= leg.exit_date:
                    value += allocation * (1.0 + leg.net_return)
                else:
                    bar = bars.get((leg.symbol, d))
                    if bar is None:
                        value += allocation
                    else:
                        shares = allocation / (leg.entry_price * (1.0 + ONE_WAY_COST))
                        value += shares * bar[3] * (1.0 - ONE_WAY_COST)
            equity_points.append({"trade_date": d, "equity": value})

        capital = capital_after
        last_batch_exit = batch_exit

    batches = pd.DataFrame(batch_rows)
    legs = pd.DataFrame(leg_rows)
    skipped = pd.DataFrame(skipped_rows)
    equity = pd.DataFrame(equity_points)
    if not equity.empty:
        equity = equity.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
    equity_values = [initial_capital] + (equity["equity"].tolist() if not equity.empty else [])

    summary = {
        "mode": mode,
        "exit_policy": policy,
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return": capital / initial_capital - 1.0,
        "max_drawdown": max_drawdown(equity_values),
        "raw_signal_days": int(selected["trade_date"].nunique()) if not selected.empty else 0,
        "raw_signal_legs": int(len(selected)),
        "executed_batches": int(len(batches)),
        "executed_legs": int(len(legs)),
        "unique_symbols": int(legs["symbol"].nunique()) if not legs.empty else 0,
        "skipped_batches": int(len(skipped)),
        "batch_win_rate": float((batches["batch_net_return"] > 0).mean()) if not batches.empty else None,
        "average_batch_return": float(batches["batch_net_return"].mean()) if not batches.empty else None,
        "median_batch_return": float(batches["batch_net_return"].median()) if not batches.empty else None,
        "leg_win_rate": float((legs["net_return"] > 0).mean()) if not legs.empty else None,
        "average_leg_return": float(legs["net_return"].mean()) if not legs.empty else None,
        "end_marked_legs": int((legs["exit_reason"] == "end_mark_to_market").sum()) if not legs.empty else 0,
    }
    return summary, batches, legs, skipped


def build_markdown(
    *,
    entry_start: date,
    end_date: date,
    summaries: list[dict[str, Any]],
    primary_batches: pd.DataFrame,
    primary_legs: pd.DataFrame,
    mrna: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> str:
    labels = {
        "green_body": "绿K（收盘<开盘）",
        "negative_close_to_close": "收盘低于前收",
        "green_and_negative": "绿K且低于前收",
        "drop8": "收盘跌幅≤-8%",
    }
    lines = [
        "| 信号定义 | 卖出规则 | 最终资金 | 总收益 | 最大回撤 | 执行批次 | 交易腿 | 去重标的 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        policy = "+10%/-15%/D+3" if item["exit_policy"] == "bracket" else "当日收盘"
        lines.append(
            f"| {labels[item['mode']]} | {policy} | ¥{item['final_capital']:,.0f} | "
            f"{item['total_return']:+.2%} | {item['max_drawdown']:.2%} | "
            f"{item['executed_batches']} | {item['executed_legs']} | {item['unique_symbols']} |"
        )
    batch_preview = primary_batches.tail(12).to_markdown(index=False) if not primary_batches.empty else "无"
    leg_preview = primary_legs.tail(15).to_markdown(index=False) if not primary_legs.empty else "无"
    if mrna.empty:
        mrna_text = "三个月窗口内，MRNA没有进入日度活跃度全市场排名表的可核验记录。"
    else:
        mrna_view = mrna[[
            "trade_date", "entry_date", "activity_rank", "activity_ratio",
            "adjusted_open", "adjusted_close", "close_to_close_pct", "green_body"
        ]].copy()
        mrna_text = mrna_view.to_markdown(index=False)
    return f"""# 活跃度Top 3 + 绿K次日开盘：三个月资金回测

- 入场窗口：**{entry_start} 至 {end_date}**
- 初始本金：**¥1,000,000**
- 主信号：信号日先排全市场活跃度Top 3，再保留**收盘低于开盘的绿K**，下一交易日开盘等权满仓。
- 主卖出：+10%止盈、-15%止损、最迟D+3收盘；同日双触发按止损优先。
- 资金纪律：上一批未全部退出，不重复加杠杆开新批次。
- 成本：单边25bp。

## 结果

{chr(10).join(lines)}

## 主口径最近执行批次

{batch_preview}

## 主口径最近交易腿

{leg_preview}

## MRNA核对

{mrna_text}

## 数据诊断

```json
{json.dumps(diagnostics, ensure_ascii=False, indent=2, default=scalar)}
```
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--end-date", default="2026-08-21")
    parser.add_argument("--cache-dir", default=".cache/us_daily_picker/sec")
    parser.add_argument("--kline-cache-dir", default=".cache/us_intraday_picker/kline")
    parser.add_argument("--output-dir", default="backtest_green_k_top3_3m")
    parser.add_argument("--refresh-sec", action="store_true")
    args = parser.parse_args()

    requested_end = date.fromisoformat(args.end_date)
    entry_start = (pd.Timestamp(requested_end) - pd.DateOffset(months=3)).date()
    signal_calendar = previous_session_map(entry_start, requested_end)

    cache_dir = Path(args.cache_dir)
    kline_cache = Path(args.kline_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info_path, sec_facts, sec_manifest = daily.ensure_weekly_reference_cache(
        cache_dir, requested_end, force_refresh=args.refresh_sec
    )
    kline_cache.mkdir(parents=True, exist_ok=True)
    kline_path = kline_cache / "stock_kline.parquet"
    kline_cache_hit = kline_path.exists() and kline_path.stat().st_size > 1_000_000
    if not kline_cache_hit:
        daily.download(daily.http_session(False), daily.ALPHA_SOURCES["kline"], kline_path, force=True)

    top3, mrna, bars_window, diagnostics = build_rankings(
        info_path=info_path,
        kline_path=kline_path,
        sec_facts=sec_facts,
        signal_calendar=signal_calendar,
        end_date=requested_end,
    )
    diagnostics["kline_cache_hit"] = kline_cache_hit
    diagnostics["sec_manifest"] = sec_manifest

    sessions = xnys_sessions(entry_start - timedelta(days=7), requested_end)
    configurations = [
        ("green_body", "bracket"),
        ("green_body", "same_day_close"),
        ("negative_close_to_close", "bracket"),
        ("green_and_negative", "bracket"),
        ("drop8", "bracket"),
    ]
    summaries: list[dict[str, Any]] = []
    result_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for mode, policy in configurations:
        summary, batches, legs, skipped = run_portfolio(
            top3=top3,
            bars_window=bars_window,
            sessions=sessions,
            initial_capital=args.capital,
            mode=mode,
            policy=policy,
            end_date=requested_end,
        )
        summaries.append(summary)
        key = f"{mode}__{policy}"
        result_frames[key] = (batches, legs, skipped)
        batches.to_csv(output_dir / f"batches__{key}.csv", index=False, encoding="utf-8-sig")
        legs.to_csv(output_dir / f"legs__{key}.csv", index=False, encoding="utf-8-sig")
        skipped.to_csv(output_dir / f"skipped__{key}.csv", index=False, encoding="utf-8-sig")

    top3.to_csv(output_dir / "daily_top3.csv", index=False, encoding="utf-8-sig")
    mrna.to_csv(output_dir / "mrna_rank_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")

    primary_batches, primary_legs, _ = result_frames["green_body__bracket"]
    payload = {
        "generated_at_utc": datetime.now(timezone.utc),
        "entry_start": entry_start,
        "end_date": requested_end,
        "initial_capital": args.capital,
        "primary_definition": "activity Top 3 on D; D close < D open; buy D+1 open",
        "primary_exit": "+10% TP / -15% SL / D+3 close; stop-first if both touched",
        "capital_rule": "one all-in equal-weight batch at a time; no overlapping leverage",
        "one_way_cost": ONE_WAY_COST,
        "summaries": summaries,
        "diagnostics": diagnostics,
    }
    write_json(output_dir / "result.json", payload)
    markdown = build_markdown(
        entry_start=entry_start,
        end_date=requested_end,
        summaries=summaries,
        primary_batches=primary_batches,
        primary_legs=primary_legs,
        mrna=mrna,
        diagnostics=diagnostics,
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")

    primary = next(item for item in summaries if item["mode"] == "green_body" and item["exit_policy"] == "bracket")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=scalar))
    github_output = Path(str(Path.cwd() / ".github_output_unused"))
    if "GITHUB_OUTPUT" in __import__("os").environ:
        with open(__import__("os").environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as handle:
            handle.write(f"final_capital={primary['final_capital']:.6f}\n")
            handle.write(f"total_return={primary['total_return']:.12f}\n")
            handle.write(f"executed_batches={primary['executed_batches']}\n")
            handle.write(f"executed_legs={primary['executed_legs']}\n")
    del github_output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
