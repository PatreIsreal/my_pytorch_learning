#!/usr/bin/env python3
"""Rewrite the daily picker output as a concise top-10 ranking report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def money(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if abs(number) >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if abs(number) >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    if abs(number) >= 1_000:
        return f"${number / 1_000:.1f}K"
    return f"${number:.2f}"


def pct(value: Any) -> str:
    return "—" if value is None else f"{float(value):+.2f}%"


def ratio(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}x"


def markdown_table(rows: list[dict[str, Any]], selected: set[str]) -> str:
    headers = ["排名", "代码", "公司", "当日涨跌幅", "活跃度", "成交额", "自由流通市值", "跌幅筛选"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for index, row in enumerate(rows, start=1):
        ticker = str(row.get("symbol") or "")
        company = str(row.get("company_name") or "").replace("|", "/")
        passed = ticker in selected
        lines.append(
            "| "
            + " | ".join(
                [
                    str(int(row.get("activity_rank") or index)),
                    f"**{ticker}**" if passed else ticker,
                    company,
                    pct(row.get("drop_pct")),
                    ratio(row.get("activity_ratio")),
                    money(row.get("amount")),
                    money(row.get("signal_float_market_cap")),
                    "**通过（≤-8%）**" if passed else "未通过",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    latest_path = output_dir / "latest.json"
    if not latest_path.exists():
        return 0

    payload = json.loads(latest_path.read_text(encoding="utf-8"))
    signal_date = str(payload["signal_date"])
    entry_date = str(payload["entry_date"])
    result_dir = output_dir / signal_date
    rows = payload.pop("top_three", payload.get("top_ten", []))
    rows = list(rows)[:10]
    payload["top_ten"] = rows
    payload["ranking_limit"] = 10
    selected_rows = list(payload.get("selected") or [])
    selected_tickers = {str(row.get("symbol")) for row in selected_rows}

    table = markdown_table(rows, selected_tickers)
    if selected_rows:
        filtered = "、".join(f"**{row['symbol']}**" for row in selected_rows)
        conclusion = f"前10名中共有 **{len(selected_rows)}** 只股票满足当日跌幅≤-8%：{filtered}。"
    else:
        conclusion = "前10名中没有股票满足当日跌幅≤-8%，本次跌幅筛选结果为空。"

    diagnostics = payload.get("diagnostics") or {}
    markdown = f"""# 美股活跃度前10排名｜{signal_date} ET

> 下一交易日：**{entry_date}**  
> 排名公式：当日成交金额 ÷ 点时自由流通市值  
> 跌幅条件：当日复权收盘涨跌幅 ≤ **-8%**

## 前10排名

{table}

## 跌幅筛选结果

{conclusion}

## 数据诊断

- 质量合格股票数：{int(diagnostics.get('quality_eligible_count') or 0):,}
- 点时SEC流通值覆盖数：{int(diagnostics.get('sec_public_float_covered_count') or 0):,}
- 通过实盘过滤的覆盖数：{int(diagnostics.get('tradable_covered_count') or 0):,}
- 行情源最新交易日：{diagnostics.get('source_latest_market_date', '—')}

> 本任务只输出活跃度前10排名及跌幅筛选，不执行历史回测，也不连接券商下单。
"""

    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "picks.md").write_text(markdown, encoding="utf-8")
    (output_dir / "latest.md").write_text(markdown, encoding="utf-8")
    write_csv(result_dir / "top10.csv", rows)
    legacy = result_dir / "top3.csv"
    legacy.unlink(missing_ok=True)
    (result_dir / "picks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
