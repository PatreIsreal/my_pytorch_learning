#!/usr/bin/env python3
"""Validated Beijing-timed latest-snapshot Top 3 task.

The activity ranking is based on the latest public market snapshot, but the
-8% signal annotation is based on the *previous completed U.S. session's*
close-to-close return.  The ranking itself is never filtered by the decline
condition: all three activity leaders are always shown.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

import us_intraday_picker.intraday_top3 as application
from us_intraday_picker.float_reference import build_float_reference

MIN_ESTIMATED_DOLLAR_VOLUME = 1_000_000.0
MIN_FLOAT_TO_TOTAL_RATIO = 0.01
MAX_FLOAT_TO_TOTAL_RATIO = 1.20
SPOT_CHECK_SYMBOLS = ("MRNA",)


def previous_market_session(reference_date: date) -> date:
    calendar = application.xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        pd.Timestamp(reference_date - timedelta(days=14)),
        pd.Timestamp(reference_date),
    )
    dates = [stamp.date() for stamp in sessions]
    if reference_date not in dates:
        raise RuntimeError(f"Reference date is not an XNYS session: {reference_date}")
    index = dates.index(reference_date)
    if index < 1:
        raise RuntimeError(f"No previous XNYS session for {reference_date}")
    return dates[index - 1]


def qpath(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def load_or_download_kline(cache_directory: Path) -> tuple[Path, int, bool]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    destination = cache_directory / "stock_kline.parquet"
    cache_hit = destination.exists() and destination.stat().st_size > 1_000_000
    if not cache_hit:
        application.daily.download(
            application.daily.http_session(False),
            application.daily.ALPHA_SOURCES["kline"],
            destination,
            force=True,
        )
    return destination, destination.stat().st_size, cache_hit


def build_previous_session_returns(
    *,
    kline_path: Path,
    reference_date: date,
    previous_date: date,
) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    frame = connection.execute(
        f"""
        WITH prepared AS (
            SELECT
                UPPER(REPLACE(CAST(symbol AS VARCHAR), '.', '-')) AS symbol,
                TRY_CAST(bar_time AS DATE) AS trade_date,
                TRY_CAST(close AS DOUBLE) AS close_price,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        UPPER(REPLACE(CAST(symbol AS VARCHAR), '.', '-')),
                        TRY_CAST(bar_time AS DATE)
                    ORDER BY TRY_CAST(bar_time AS TIMESTAMP) DESC NULLS LAST
                ) AS row_rank
            FROM read_parquet('{qpath(kline_path)}')
            WHERE UPPER(COALESCE(CAST(kline_t AS VARCHAR), '1D')) = '1D'
              AND TRY_CAST(bar_time AS DATE) IN (?, ?)
        ), dedup AS (
            SELECT symbol, trade_date, close_price
            FROM prepared
            WHERE row_rank = 1 AND close_price > 0
        ), pivoted AS (
            SELECT
                symbol,
                MAX(close_price) FILTER (WHERE trade_date = ?) AS previous_close,
                MAX(close_price) FILTER (WHERE trade_date = ?) AS reference_close
            FROM dedup
            GROUP BY symbol
        )
        SELECT
            symbol,
            previous_close,
            reference_close,
            100.0 * (reference_close / previous_close - 1.0)
                AS previous_day_change_pct
        FROM pivoted
        WHERE previous_close > 0 AND reference_close > 0
        """,
        [previous_date, reference_date, previous_date, reference_date],
    ).df()
    frame["previous_session_date"] = previous_date
    frame["reference_session_date"] = reference_date
    return frame


def load_or_build_previous_returns(
    *,
    cache_directory: Path,
    kline_path: Path,
    reference_date: date,
    previous_date: date,
) -> tuple[pd.DataFrame, bool]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / f"previous_returns_{reference_date.isoformat()}.parquet"
    cache_hit = path.exists() and path.stat().st_size > 1_000
    if cache_hit:
        frame = pd.read_parquet(path)
        for column in ("previous_session_date", "reference_session_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
        return frame, True
    frame = build_previous_session_returns(
        kline_path=kline_path,
        reference_date=reference_date,
        previous_date=previous_date,
    )
    frame.to_parquet(path, index=False)
    return frame, False


def load_or_build_float_reference(
    *,
    cache_directory: Path,
    kline_path: Path,
    sec_facts: pd.DataFrame,
    reference_date: date,
    snapshot_date: date,
) -> tuple[pd.DataFrame, bool]:
    cache_directory.mkdir(parents=True, exist_ok=True)
    path = cache_directory / (
        f"float_reference_{reference_date.isoformat()}_{snapshot_date.isoformat()}.parquet"
    )
    cache_hit = path.exists() and path.stat().st_size > 1_000
    if cache_hit:
        frame = pd.read_parquet(path)
        for column in (
            "measurement_date",
            "measurement_price_date",
            "filed_date",
            "reference_price_date",
        ):
            frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
        return frame, True
    frame = build_float_reference(
        kline_path=kline_path,
        sec_facts=sec_facts,
        reference_date=reference_date,
        snapshot_date=snapshot_date,
    )
    frame.to_parquet(path, index=False)
    return frame, False


def compact_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    number = float(value)
    if abs(number) >= 1e9:
        return f"${number / 1e9:.2f}B"
    if abs(number) >= 1e6:
        return f"${number / 1e6:.2f}M"
    if abs(number) >= 1e3:
        return f"${number / 1e3:.1f}K"
    return f"${number:.2f}"


def build_markdown(
    ranking: pd.DataFrame,
    *,
    now_bjt: datetime,
    now_et: datetime,
    previous_date: date,
    reference_date: date,
    snapshot_meta: dict[str, Any],
    diagnostics: dict[str, Any],
    spot_checks: dict[str, Any],
) -> str:
    lines = [
        "| 排名 | 代码 | 最新价 | 当前涨跌幅 | 前一交易日涨跌幅 | 活跃度 | 累计成交量 | 估算成交额 | 自由流通市值 | 前日跌幅≤-8% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ranking.itertuples(index=False):
        current_change = "—" if pd.isna(row.change_pct) else f"{row.change_pct:+.2f}%"
        previous_change = (
            "—"
            if pd.isna(row.previous_day_change_pct)
            else f"{row.previous_day_change_pct:+.2f}%"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.activity_rank)),
                    f"**{row.symbol}**",
                    f"${row.last_price:.2f}",
                    current_change,
                    previous_change,
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

    spot_lines: list[str] = []
    for symbol, record in spot_checks.items():
        if not record:
            spot_lines.append(f"- **{symbol}**：未进入当前可排名股票池。")
            continue
        spot_lines.append(
            f"- **{symbol}**：当前活跃度排名第{record['activity_rank']}，"
            f"当前涨跌幅{record['current_change_pct']:+.2f}%，"
            f"前一交易日涨跌幅{record['previous_day_change_pct']:+.2f}%，"
            f"前日大跌条件{'通过' if record['drop_filter_pass'] else '未通过'}。"
        )
    spot_text = "\n".join(spot_lines)

    timings = diagnostics["timings_seconds"]
    return f"""# 美股最新公开快照活跃度前三

> 北京时间：**{now_bjt.strftime('%Y-%m-%d %H:%M:%S')}**  
> 美东时间：**{now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}**  
> 前一交易日信号区间：**{previous_date} → {reference_date}**

## 当前活跃度前三

{table}

## 正确的跌幅信号

排名不受涨跌幅限制，始终输出当前活跃度前三；交易信号只检查**前一完整交易日**是否下跌至少8%。
当前前三中，前一交易日跌幅≤-8%的标的：{passed_text}。

## MRNA专项核对

{spot_text}

## 口径

- 当前活跃度 = 当前累计成交金额 ÷ 当前自由流通市值。
- 当前成交金额以“最新价 × 当前累计成交量”估算；排序等价于“累计成交量 ÷ 推算自由流通股数”。
- **跌幅条件使用{reference_date}相对{previous_date}的日线收盘涨跌幅，不使用今天盘中涨跌幅。**
- 因此，某只股票前一日下跌30%、今天上涨15%，仍然可以通过前日大跌信号；是否进入前三只取决于今天的活跃度排名。
- 当前价格与成交量来自Nasdaq公开筛选器，可能延迟，不是交易所直连实时行情。

## 数据与耗时诊断

- Nasdaq原始行数：{snapshot_meta['raw_row_count']:,}
- SEC流通值参考数：{diagnostics['float_reference_count']:,}
- 参与排名数：{diagnostics['rankable_count']:,}
- 日线缓存命中：{'是' if diagnostics['cache_hits']['kline'] else '否'}
- 流通值派生缓存命中：{'是' if diagnostics['cache_hits']['float_reference'] else '否'}
- 前日收益缓存命中：{'是' if diagnostics['cache_hits']['previous_returns'] else '否'}
- 选股程序总耗时：**{timings['selector_total']:.2f}秒**
- Nasdaq快照：{timings['nasdaq_snapshot']:.2f}秒；日线准备：{timings['kline']:.2f}秒；SEC流通值参考：{timings['float_reference']:.2f}秒

> 本任务只生成研究排名和前日跌幅标记，不回测，也不连接券商下单。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/us_intraday_picker/sec")
    parser.add_argument("--kline-cache-dir", default=".cache/us_intraday_picker/kline")
    parser.add_argument("--derived-cache-dir", default=".cache/us_intraday_picker/derived")
    parser.add_argument("--work-dir", default=".work/us_intraday_picker")
    parser.add_argument("--output-dir", default="intraday_picks_top3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-sec", action="store_true")
    args = parser.parse_args()

    selector_started = time.perf_counter()
    stage_started = selector_started
    timings: dict[str, float] = {}

    now_utc = datetime.now(timezone.utc)
    now_bjt = now_utc.astimezone(application.BJT)
    now_et = now_utc.astimezone(application.ET)
    reference_date, today_is_session = application.latest_completed_session(now_et)
    previous_date = previous_market_session(reference_date)
    if not today_is_session:
        result = {
            "status": "skipped_non_session",
            "generated_at_bjt": now_bjt,
            "generated_at_et": now_et,
        }
        application.append_output(status=result["status"], notify="false")
        print(json.dumps(result, default=application.scalar, ensure_ascii=False, indent=2))
        return 0

    cache_dir = Path(args.cache_dir)
    kline_cache_dir = Path(args.kline_cache_dir)
    derived_cache_dir = Path(args.derived_cache_dir)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    info_path, sec_facts, reference_manifest = application.daily.ensure_weekly_reference_cache(
        cache_dir,
        reference_date,
        force_refresh=args.refresh_sec,
    )
    del info_path
    timings["sec_reference"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    kline_path, kline_size, kline_cache_hit = load_or_download_kline(kline_cache_dir)
    timings["kline"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    snapshot, snapshot_meta = application.download_nasdaq_snapshot(now_utc)
    timings["nasdaq_snapshot"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    previous_returns, previous_returns_cache_hit = load_or_build_previous_returns(
        cache_directory=derived_cache_dir,
        kline_path=kline_path,
        reference_date=reference_date,
        previous_date=previous_date,
    )
    timings["previous_returns"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    float_reference, float_reference_cache_hit = load_or_build_float_reference(
        cache_directory=derived_cache_dir,
        kline_path=kline_path,
        sec_facts=sec_facts,
        reference_date=reference_date,
        snapshot_date=now_et.date(),
    )
    timings["float_reference"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    merged = (
        snapshot.merge(float_reference, on="symbol", how="inner")
        .merge(previous_returns, on="symbol", how="left")
    )
    merged["estimated_dollar_volume"] = (
        merged["last_price"] * merged["cumulative_volume"]
    )
    merged["current_float_market_cap"] = (
        merged["last_price"] * merged["inferred_float_shares"]
    )
    merged["activity_ratio"] = (
        merged["estimated_dollar_volume"] / merged["current_float_market_cap"]
    )
    merged["float_to_total_market_cap"] = (
        merged["current_float_market_cap"] / merged["nasdaq_market_cap"]
    )
    merged["company_name"] = merged["nasdaq_name"].where(
        merged["nasdaq_name"].astype(str).str.len() > 0,
        merged["sec_entity_name"],
    )

    base_mask = (
        (merged["last_price"] >= application.MIN_PRICE)
        & (merged["current_float_market_cap"] >= application.MIN_FLOAT_MARKET_CAP)
        & (merged["estimated_dollar_volume"] >= MIN_ESTIMATED_DOLLAR_VOLUME)
        & (merged["cumulative_volume"] > 0)
        & merged["activity_ratio"].notna()
        & (merged["activity_ratio"] > 0)
    )
    market_cap_sanity = (
        merged["nasdaq_market_cap"].notna()
        & (merged["nasdaq_market_cap"] > 0)
        & merged["float_to_total_market_cap"].between(
            MIN_FLOAT_TO_TOTAL_RATIO,
            MAX_FLOAT_TO_TOTAL_RATIO,
            inclusive="both",
        )
    )
    base_candidates = merged[base_mask].copy()
    rankable = merged[base_mask & market_cap_sanity].copy()
    rankable = rankable.sort_values(
        ["activity_ratio", "estimated_dollar_volume", "symbol"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if len(rankable) < application.RANK_LIMIT:
        raise RuntimeError(f"Only {len(rankable)} stocks passed ranking quality controls")
    rankable["activity_rank"] = range(1, len(rankable) + 1)
    rankable["drop_filter_pass"] = (
        rankable["previous_day_change_pct"].fillna(math.inf)
        <= application.DROP_THRESHOLD_PCT
    )
    ranking = rankable.head(application.RANK_LIMIT).copy()

    spot_checks: dict[str, Any] = {}
    for symbol in SPOT_CHECK_SYMBOLS:
        row = rankable[rankable["symbol"] == symbol]
        if row.empty:
            spot_checks[symbol] = None
        else:
            item = row.iloc[0]
            spot_checks[symbol] = {
                "activity_rank": int(item["activity_rank"]),
                "activity_ratio": float(item["activity_ratio"]),
                "current_change_pct": float(item["change_pct"]),
                "previous_day_change_pct": float(item["previous_day_change_pct"]),
                "drop_filter_pass": bool(item["drop_filter_pass"]),
            }
    timings["ranking"] = time.perf_counter() - stage_started
    timings["selector_total"] = time.perf_counter() - selector_started

    diagnostics = {
        "previous_session_date": previous_date,
        "reference_session_date": reference_date,
        "float_reference_count": len(float_reference),
        "matched_count": len(merged),
        "base_candidate_count": len(base_candidates),
        "market_cap_sanity_excluded_count": int(len(base_candidates) - len(rankable)),
        "rankable_count": len(rankable),
        "kline_size_bytes": kline_size,
        "today_is_xnys_session": today_is_session,
        "minimum_estimated_dollar_volume": MIN_ESTIMATED_DOLLAR_VOLUME,
        "float_to_total_market_cap_range": [
            MIN_FLOAT_TO_TOTAL_RATIO,
            MAX_FLOAT_TO_TOTAL_RATIO,
        ],
        "cache_hits": {
            "kline": kline_cache_hit,
            "float_reference": float_reference_cache_hit,
            "previous_returns": previous_returns_cache_hit,
        },
        "timings_seconds": timings,
        "reference_manifest": reference_manifest,
    }
    markdown = build_markdown(
        ranking,
        now_bjt=now_bjt,
        now_et=now_et,
        previous_date=previous_date,
        reference_date=reference_date,
        snapshot_meta=snapshot_meta,
        diagnostics=diagnostics,
        spot_checks=spot_checks,
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
        "previous_session_date": previous_date,
        "reference_date": reference_date,
        "issue_title": issue_title,
        "ranking": ranking.to_dict("records"),
        "spot_checks": spot_checks,
        "snapshot_meta": snapshot_meta,
        "diagnostics": diagnostics,
        "methodology": {
            "activity": (
                "estimated cumulative dollar volume / current inferred "
                "free-float market cap"
            ),
            "estimated_dollar_volume": "last sale * cumulative share volume",
            "drop_signal": (
                "previous completed session close-to-close return <= -8%; "
                "not current snapshot change"
            ),
            "drop_threshold_pct": application.DROP_THRESHOLD_PCT,
            "rank_limit": application.RANK_LIMIT,
            "ranking_is_unconditional_on_drop": True,
            "minimum_estimated_dollar_volume": MIN_ESTIMATED_DOLLAR_VOLUME,
            "float_to_total_market_cap_range": [
                MIN_FLOAT_TO_TOTAL_RATIO,
                MAX_FLOAT_TO_TOTAL_RATIO,
            ],
        },
    }
    application.write_json(result_dir / "ranking.json", payload)
    application.write_json(output_dir / "latest.json", payload)
    application.append_output(
        status="ready",
        notify="true",
        issue_title=issue_title,
        markdown_path=str(markdown_path),
        run_key=run_key,
        top3=",".join(tickers),
        runtime_seconds=f"{timings['selector_total']:.2f}",
    )
    application.append_summary(markdown)
    print(json.dumps(payload, default=application.scalar, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        application.append_output(status="failed", notify="false")
        print(f"INTRADAY TOP3 FAILED: {error}", file=application.sys.stderr, flush=True)
        raise
