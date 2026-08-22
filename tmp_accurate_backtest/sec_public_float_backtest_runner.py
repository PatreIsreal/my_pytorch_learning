#!/usr/bin/env python3
"""Execute the SEC backtest after a targeted DuckDB SQL compatibility patch."""
from pathlib import Path

script = Path(__file__).with_name("sec_public_float_backtest.py")
source = script.read_text(encoding="utf-8")
old = "b.trade_date >= f.measurement_date - INTERVAL ? DAY"
new = "b.trade_date >= f.measurement_date - ?"
if old not in source:
    raise RuntimeError("expected DuckDB interval expression not found")
source = source.replace(old, new, 1)
namespace = {"__name__": "__main__", "__file__": str(script)}
exec(compile(source, str(script), "exec"), namespace)
