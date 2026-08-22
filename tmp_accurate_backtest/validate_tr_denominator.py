#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SOURCES = {
    "quote": "https://huggingface.co/datasets/AlphaDojo/dojo_quote/resolve/main/data.parquet?download=true",
    "kline": "https://huggingface.co/datasets/AlphaDojo/dojo_stock_kline/resolve/main/data.parquet?download=true",
}


def session() -> requests.Session:
    retry = Retry(total=7, connect=7, read=7, backoff_factor=2,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    s = requests.Session()
    s.headers.update({"User-Agent": "turnover-denominator-audit/2026-08-22"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def download(s: requests.Session, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1000:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, 4):
        try:
            with s.get(url, stream=True, timeout=(30, 1200), allow_redirects=True) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.replace(path)
            return
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt == 3:
                raise
            time.sleep(attempt * 5)


def q(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def metrics(df: pd.DataFrame, col: str) -> dict:
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n": int(len(s)),
        "median": float(s.median()) if len(s) else None,
        "p75": float(s.quantile(.75)) if len(s) else None,
        "p90": float(s.quantile(.90)) if len(s) else None,
        "within_5pct": float((s <= .05).mean()) if len(s) else None,
        "within_10pct": float((s <= .10).mean()) if len(s) else None,
        "within_20pct": float((s <= .20).mean()) if len(s) else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".tmp_denominator_data")
    ap.add_argument("--out-dir", default="denominator_validation_output")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = session()
    files = {}
    for name, url in SOURCES.items():
        path = data_dir / f"{name}.parquet"
        download(s, url, path)
        files[name] = path
        print(name, path.stat().st_size, flush=True)

    con = duckdb.connect()
    con.execute(f"CREATE VIEW qraw AS SELECT * FROM read_parquet('{q(files['quote'])}')")
    con.execute(f"CREATE VIEW kraw AS SELECT * FROM read_parquet('{q(files['kline'])}')")
    df = con.execute("""
        WITH q AS (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    CAST(symbol AS VARCHAR) symbol,
                    TRY_CAST(quote_time AS TIMESTAMP) quote_time,
                    TRY_CAST(float_shares AS DOUBLE) float_shares,
                    TRY_CAST(total_shares AS DOUBLE) total_shares,
                    TRY_CAST(float_market_cap AS DOUBLE) float_market_cap,
                    TRY_CAST(market_cap AS DOUBLE) market_cap,
                    TRY_CAST(last_price AS DOUBLE) last_price,
                    TRY_CAST(turn_rate AS DOUBLE) quote_turn_rate,
                    ROW_NUMBER() OVER (PARTITION BY CAST(symbol AS VARCHAR)
                                       ORDER BY TRY_CAST(quote_time AS TIMESTAMP) DESC NULLS LAST) rn
                FROM qraw
            ) WHERE rn=1
        ), k AS (
            SELECT * EXCLUDE (rn)
            FROM (
                SELECT
                    CAST(symbol AS VARCHAR) symbol,
                    TRY_CAST(bar_time AS DATE) trade_date,
                    TRY_CAST(vol AS DOUBLE) volume,
                    TRY_CAST(tr AS DOUBLE) tr,
                    TRY_CAST(adj_factor_cum AS DOUBLE) adj_factor,
                    ROW_NUMBER() OVER (PARTITION BY CAST(symbol AS VARCHAR)
                                       ORDER BY TRY_CAST(bar_time AS DATE) DESC NULLS LAST) rn
                FROM kraw
                WHERE UPPER(COALESCE(CAST(kline_t AS VARCHAR), '1D'))='1D'
                  AND TRY_CAST(vol AS DOUBLE)>0
                  AND TRY_CAST(tr AS DOUBLE)>0
            ) WHERE rn=1
        )
        SELECT
            k.*, q.* EXCLUDE(symbol),
            k.volume/(k.tr/100.0) implied_denominator,
            ABS(k.volume/(k.tr/100.0)/q.float_shares-1.0) err_vs_float,
            ABS(k.volume/(k.tr/100.0)/q.total_shares-1.0) err_vs_total,
            ABS(k.tr/q.quote_turn_rate-1.0) err_vs_quote_turn_rate,
            DATE_DIFF('day', k.trade_date, TRY_CAST(q.quote_time AS DATE)) date_gap
        FROM k JOIN q USING(symbol)
        WHERE ABS(DATE_DIFF('day', k.trade_date, TRY_CAST(q.quote_time AS DATE)))<=5
          AND q.float_shares>0 AND q.total_shares>0
    """).df()
    df.to_csv(out_dir / "denominator_pairs.csv", index=False, encoding="utf-8-sig")

    rows=[]
    for threshold in [0.1,0.5,1.0,5.0,10.0]:
        x=df[df.tr>=threshold]
        for target,col in [("float_shares","err_vs_float"),("total_shares","err_vs_total"),("quote_turn_rate","err_vs_quote_turn_rate")]:
            rows.append({"min_tr_pct":threshold,"target":target,**metrics(x,col)})
    summary=pd.DataFrame(rows)
    summary.to_csv(out_dir / "denominator_summary.csv",index=False,encoding="utf-8-sig")

    ratio = df[["symbol","tr","implied_denominator","float_shares","total_shares","err_vs_float","err_vs_total","quote_turn_rate","err_vs_quote_turn_rate","date_gap"]].copy()
    ratio["closer_target"] = np.where(ratio.err_vs_float <= ratio.err_vs_total, "float_shares", "total_shares")
    by_closer = ratio.groupby("closer_target").size().rename("count").reset_index()
    by_closer.to_csv(out_dir / "closer_target_counts.csv",index=False,encoding="utf-8-sig")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("CLOSER")
    print(by_closer.to_string(index=False))
    print("selected examples")
    print(ratio[ratio.symbol.isin(["DFNS","CAR","RXT","ODD","AAPL","MSFT","NVDA"])].to_string(index=False))
    with (out_dir/"metadata.json").open("w") as f:
        json.dump({"rows":len(df)},f)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
