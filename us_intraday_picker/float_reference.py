#!/usr/bin/env python3
"""Correct point-in-time SEC public-float share inference for intraday ranking.

AlphaDojo's OHLC fields are on the traded-price scale for each session. The
``adj_factor_cum`` field is metadata and must not be divided into the close when
recovering the market price used by an SEC public-float disclosure. Doing so can
inflate historical prices (and shrink inferred float shares) by split factors.

This module therefore:

* uses the daily ``close`` directly as the measurement-date market price;
* adjusts inferred shares only by the product of explicit ``splits`` events
  between the SEC measurement date and the latest completed session; and
* keeps the same strict filing-date and maximum-age rules as the audited model.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

import us_daily_picker.picker as daily

MAX_PUBLIC_FLOAT_AGE_DAYS = daily.MAX_PUBLIC_FLOAT_AGE_DAYS
MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS = daily.MAX_MEASUREMENT_PRICE_LOOKBACK_DAYS
BAR_HISTORY_BUFFER_DAYS = daily.BAR_HISTORY_BUFFER_DAYS


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
                TRY_CAST(close AS DOUBLE) AS raw_close,
                TRY_CAST(vol AS DOUBLE) AS volume,
                TRY_CAST(amount AS DOUBLE) AS amount,
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
            WHERE row_rank = 1
              AND raw_close > 0
        )
        SELECT
            *,
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
                bars.average_trade_price AS measurement_average_price,
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
        WHERE price_rank = 1
          AND measurement_float_shares > 0
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
            measurement_price_date,
            measurement_raw_close,
            measurement_average_price,
            filed_date,
            form,
            accession,
            sec_source_url,
            reference_price_date,
            public_float_age_days
        FROM candidates
        WHERE fact_rank = 1
          AND inferred_float_shares > 0
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
    for column in (
        "measurement_date",
        "measurement_price_date",
        "filed_date",
        "reference_price_date",
    ):
        reference[column] = pd.to_datetime(reference[column], errors="coerce").dt.date
    return reference
