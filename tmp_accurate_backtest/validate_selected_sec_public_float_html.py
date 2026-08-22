#!/usr/bin/env python3
"""Cross-check selected EntityPublicFloat facts against inline-XBRL filing HTML."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "OpenAI selected public-float filing audit support@openai.com"
SELECTED = [{"symbol":"AAP","cik":1158449,"accession":"0001193125-26-051305","public_float_value_usd":1900000000.0,"measurement_date":"2025-07-12","form":"10-K"},{"symbol":"ALDX","cik":1341235,"accession":"0001193125-26-083012","public_float_value_usd":231896127.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"ANVS","cik":1477845,"accession":"0001104659-26-027751","public_float_value_usd":36076450.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"ARRY","cik":1820721,"accession":"0001820721-26-000008","public_float_value_usd":707055499.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"AXTI","cik":1051627,"accession":"0001437749-26-008612","public_float_value_usd":83786290.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"BATL","cik":1282648,"accession":"0001104659-26-033373","public_float_value_usd":5000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"BIOA","cik":1709941,"accession":"0001193125-26-120873","public_float_value_usd":112700000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"CAR","cik":723612,"accession":"0000723612-26-000012","public_float_value_usd":2832437145.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"CMPX","cik":1738021,"accession":"0001171843-26-001337","public_float_value_usd":277000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"CTMX","cik":1501989,"accession":"0001193125-26-107146","public_float_value_usd":372300000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"CVRX","cik":1235912,"accession":"0001104659-26-014708","public_float_value_usd":112400000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"DGXX","cik":1854368,"accession":"0001213900-26-050340","public_float_value_usd":92039538.0,"measurement_date":"2025-06-30","form":"10-K/A"},{"symbol":"EOSE","cik":1805077,"accession":"0001628280-26-011961","public_float_value_usd":1283000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"FEIM","cik":39020,"accession":"0001185185-26-002997","public_float_value_usd":240400000.0,"measurement_date":"2025-10-31","form":"10-K"},{"symbol":"FIVN","cik":1288847,"accession":"0001288847-26-000023","public_float_value_usd":1158500000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"FROG","cik":1800667,"accession":"0001193125-26-051382","public_float_value_usd":4400000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"FULC","cik":1680581,"accession":"0001193125-26-065197","public_float_value_usd":296079672.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"IBRX","cik":1326110,"accession":"0001326110-26-000030","public_float_value_usd":690200000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"INDI","cik":1841925,"accession":"0001193125-26-082535","public_float_value_usd":693500000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"LASE","cik":1807887,"accession":"0001493152-26-018110","public_float_value_usd":26131561.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"ONDS","cik":1646188,"accession":"0001213900-26-035981","public_float_value_usd":392000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"PEPG","cik":1835597,"accession":"0001193125-26-091536","public_float_value_usd":16900000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"PROP","cik":1162896,"accession":"0001140361-26-012036","public_float_value_usd":74131123.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"PUSA","cik":2009312,"accession":"0001493152-26-014320","public_float_value_usd":26700000.0,"measurement_date":"2025-12-31","form":"10-K"},{"symbol":"RDW","cik":1819810,"accession":"0001819810-26-000029","public_float_value_usd":895600000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"REPL","cik":1737953,"accession":"0001628280-26-045886","public_float_value_usd":234500000.0,"measurement_date":"2025-09-30","form":"10-K"},{"symbol":"ROLR","cik":1947210,"accession":"0001753926-26-000462","public_float_value_usd":8000000.0,"measurement_date":"2025-12-31","form":"10-K"},{"symbol":"RXT","cik":1810019,"accession":"0001810019-26-000015","public_float_value_usd":131000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"SION","cik":2036042,"accession":"0001628280-26-012996","public_float_value_usd":342014009.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"SOC","cik":1831481,"accession":"0001831481-26-000026","public_float_value_usd":1200000000.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"TE","cik":1992243,"accession":"0001213900-26-049667","public_float_value_usd":141000000.0,"measurement_date":"2025-06-30","form":"10-K/A"},{"symbol":"UMAC","cik":1956955,"accession":"0001683168-26-001730","public_float_value_usd":197646688.0,"measurement_date":"2025-06-30","form":"10-K"},{"symbol":"WVE","cik":1631574,"accession":"0001193125-26-073472","public_float_value_usd":851013538.0,"measurement_date":"2025-06-30","form":"10-K"}]


def session() -> requests.Session:
    retry = Retry(total=6, connect=6, read=6, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset({"GET"}))
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def parse_number(text: str, scale: int = 0, sign: str | None = None) -> float | None:
    raw = text.replace("\xa0", " ").strip()
    lower = raw.lower()
    multiplier = 1.0
    if "billion" in lower:
        multiplier = 1e9
    elif "million" in lower:
        multiplier = 1e6
    elif "thousand" in lower:
        multiplier = 1e3
    negative = ("(" in raw and ")" in raw) or sign == "-"
    cleaned = re.sub(r"[^0-9.\-]", "", raw)
    if cleaned in {"", "-", "."}:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    value *= (10.0 ** scale) * multiplier
    return -abs(value) if negative else value


def context_date(soup: BeautifulSoup, context_ref: str | None) -> str | None:
    if not context_ref:
        return None
    context = soup.find(attrs={"id": context_ref})
    if context is None:
        return None
    for tag in context.find_all(True):
        if str(tag.name).lower().endswith("instant"):
            value = tag.get_text(" ", strip=True)
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return value
    return None


def document_candidates(index_payload: dict[str, Any]) -> list[str]:
    items = (((index_payload.get("directory") or {}).get("item")) or [])
    html = []
    for item in items:
        name = str(item.get("name") or "")
        if not name.lower().endswith((".htm", ".html")):
            continue
        if name.lower().endswith("-index.html"):
            continue
        size = int(item.get("size") or 0)
        html.append((name, size))
    # Main filing documents are usually among the largest HTML files.
    return [name for name, _ in sorted(html, key=lambda pair: pair[1], reverse=True)]


def validate_one(s: requests.Session, item: dict[str, Any]) -> dict[str, Any]:
    cik = int(item["cik"])
    accession = str(item["accession"])
    no_dashes = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{no_dashes}/"
    index_url = base + "index.json"
    result = {**item, "index_url": index_url}
    try:
        response = s.get(index_url, timeout=60)
        response.raise_for_status()
        index_payload = response.json()
    except Exception as exc:
        return {**result, "status": "index_error", "error": str(exc)}

    candidates = document_candidates(index_payload)
    found_values: list[float] = []
    found_dates: list[str] = []
    found_text: list[str] = []
    matched_document = None
    for name in candidates[:12]:
        url = base + name
        try:
            response = s.get(url, timeout=90)
            response.raise_for_status()
        except Exception:
            continue
        soup = BeautifulSoup(response.content, "lxml")
        tags = []
        for tag in soup.find_all(True):
            fact_name = str(tag.attrs.get("name") or "")
            if fact_name.lower().endswith("entitypublicfloat"):
                tags.append(tag)
        if not tags:
            continue
        matched_document = url
        for tag in tags:
            text = tag.get_text(" ", strip=True)
            try:
                scale = int(str(tag.attrs.get("scale") or "0"))
            except ValueError:
                scale = 0
            value = parse_number(text, scale=scale, sign=tag.attrs.get("sign"))
            if value is not None:
                found_values.append(value)
            found_text.append(text[:300])
            found_dates.append(context_date(soup, tag.attrs.get("contextref") or tag.attrs.get("contextRef")) or "")
        break

    expected = float(item["public_float_value_usd"])
    errors = [abs(value / expected - 1.0) for value in found_values if expected > 0]
    min_error = min(errors) if errors else None
    expected_date = str(item["measurement_date"])
    date_match = expected_date in found_dates
    if min_error is not None and min_error <= 0.005 and date_match:
        status = "verified_exact"
    elif min_error is not None and min_error <= 0.02 and date_match:
        status = "verified_rounding"
    elif min_error is not None and min_error <= 0.02:
        status = "value_verified_date_mismatch"
    elif found_values:
        status = "value_mismatch"
    else:
        status = "tag_not_found"
    return {
        **result,
        "status": status,
        "matched_document_url": matched_document,
        "parsed_values_usd": json.dumps(found_values),
        "parsed_context_dates": json.dumps(found_dates),
        "tag_texts": json.dumps(found_text),
        "minimum_relative_error": min_error,
        "measurement_date_match": date_match,
        "candidate_document_count": len(candidates),
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="selected_sec_html_validation_output")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = session()
    rows = []
    for index, item in enumerate(SELECTED, start=1):
        row = validate_one(s, item)
        rows.append(row)
        print(index, item["symbol"], row.get("status"), row.get("minimum_relative_error"), flush=True)
        time.sleep(0.12)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "selected_public_float_html_validation.csv", index=False, encoding="utf-8-sig")
    summary = (
        frame.groupby("status", dropna=False).size().rename("count").reset_index()
        .sort_values("count", ascending=False)
    )
    summary.to_csv(out_dir / "validation_status_summary.csv", index=False, encoding="utf-8-sig")
    metrics = {
        "selected_fact_count": int(len(frame)),
        "verified_exact_or_rounding": int(frame["status"].isin(["verified_exact", "verified_rounding"]).sum()),
        "value_verified_date_mismatch": int((frame["status"] == "value_verified_date_mismatch").sum()),
        "value_mismatch": int((frame["status"] == "value_mismatch").sum()),
        "tag_not_found": int((frame["status"] == "tag_not_found").sum()),
        "index_error": int((frame["status"] == "index_error").sum()),
    }
    (out_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
