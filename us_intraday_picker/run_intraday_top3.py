#!/usr/bin/env python3
"""Entrypoint that installs the corrected SEC float-reference implementation."""
from __future__ import annotations

import us_intraday_picker.intraday_top3 as application
from us_intraday_picker.float_reference import build_float_reference

application.build_float_reference = build_float_reference


if __name__ == "__main__":
    raise SystemExit(application.main())
