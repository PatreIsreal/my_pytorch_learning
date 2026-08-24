#!/usr/bin/env python3
"""Validated entrypoint for the Beijing-timed latest-snapshot Top 3 task."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import us_intraday_picker.intraday_top3 as application
from us_intraday_picker.float_reference import build_float_reference

MIN_ESTIMATED_DOLLAR_VOLUME = 1_000_000.0
MIN_FLOAT_TO_TOTAL_RATIO = 0.01
MAX_FLOAT_TO_TOTAL_RATIO = 1.20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=".cache/us_intraday_picker/sec")
    parser.add_argument("--work-dir", default=".work/us_intraday_picker")
    parser.add_argument("--output-dir", default="intraday_picks_top3")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-sec", action="store_true")
    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    now_bjt = now_utc.astimezone(application.BJT)
    now_et = now_utc.astimezone(application.ET)
    reference_date, today_is_session = application.latest_completed_session(now_et)
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
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    info_path, sec_facts, reference_manifest = application.daily.ensure_weekly_reference_cache(
        cache_dir,
        reference_date,
        force_refresh=args.refresh_sec,
    )
    del info_path
    kline_path, kline_size = application.daily.download_fresh_kline(work_dir)
    snapshot, snapshot_meta = application.download_nasdaq_snapshot(now_utc)
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
    ranking = rankable.head(application.RANK_LIMIT).copy()
    ranking["drop_filter_pass"] = (
        ranking["change_pct"].fillna(math.inf) <= application.DROP_THRESHOLD_PCT
    )

    diagnostics = {
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
        "reference_manifest": reference_manifest,
    }
    markdown = application.build_markdown(
        ranking,
        now_bjt=now_bjt,
        now_et=now_et,
        reference_date=reference_date,
        snapshot_meta=snapshot_meta,
        diagnostics=diagnostics,
    )
    markdown = markdown.replace(
        f"- 参与排名数：{len(rankable):,}",
        f"- 基础候选数：{len(base_candidates):,}\n"
        f"- 市值一致性剔除数：{diagnostics['market_cap_sanity_excluded_count']:,}\n"
        f"- 参与排名数：{len(rankable):,}",
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
            "activity": (
                "estimated cumulative dollar volume / current inferred "
                "free-float market cap"
            ),
            "estimated_dollar_volume": "last sale * cumulative share volume",
            "drop_threshold_pct": application.DROP_THRESHOLD_PCT,
            "rank_limit": application.RANK_LIMIT,
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
