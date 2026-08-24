# Green-K activity strategy: stability-filtered candidate

This is a research-only full-sample stability candidate, not the pure walk-forward winner.

## Search protocol

- Screened 1,200 entry parameter sets.
- Tuned 216 TP/SL/maximum-hold combinations for each of the strongest 30 entry sets.
- Used two training quarters, one validation quarter, and a final untouched quarter.
- The automatic train/validation winner was rejected because it lost 43.44% in the final quarter.

## Candidate parameters (`sig_0245`)

Signal day D:

- Daily candle body: close/open - 1 between **-15% and -3%**.
- Close-to-close return: between **-15% and -2%**.
- Current global free-float activity rank: **50 or better**.
- Raw close: **at least $5**.
- Dollar volume: **at least $10 million**.
- SEC point-in-time inferred free-float market cap: **at least $20 million**.
- Do not require a previous-day surge, previous-day activity rank, or activity-persistence threshold.

Entry on D+1 regular-session open only when the opening gap versus D close is between **-5% and 0%**. Rank the qualifying pool by activity and buy up to three names equal-weight.

Exit:

- Take profit: **+15%**.
- Stop loss: **-20%**.
- Entry day is offset 0; if neither threshold is triggered, exit no later than the following trading day's close (`max_hold=1`).
- Stop-first assumption when both thresholds are touched in the same daily bar.
- One-way cost: 25 bp.
- One all-in batch at a time; no overlapping leverage.

## Quarterly results

| Window | Total return | Return excluding best batch | Max drawdown | Executed batches | Batch win rate |
|---|---:|---:|---:|---:|---:|
| 2025-08-18 to 2025-11-18 | +12.05% | +2.62% | -6.45% | 10 | 50.00% |
| 2025-11-19 to 2026-02-19 | +51.61% | +32.50% | -12.64% | 26 | 65.38% |
| 2026-02-20 to 2026-05-20 | +61.66% | +44.14% | -15.50% | 30 | 60.00% |
| 2026-05-21 to 2026-08-21 | +14.26% | -0.14% | -25.93% | 30 | 46.67% |

## Latest-quarter comparison from CNY 1,000,000

| Rule | Final capital | Return | Max drawdown | Batches | Return excluding best batch |
|---|---:|---:|---:|---:|---:|
| Original green-pool Top 3; TP10/SL15/D+3 | CNY 863,972.69 | -13.60% | -40.04% | 20 | -21.06% |
| Stability candidate; strong green decline, no gap-up; TP15/SL20/max-hold 1 | CNY 1,142,627.31 | +14.26% | -25.93% | 30 | -0.14% |

The latest-quarter capital improvement is CNY 278,654.62, or 27.87 percentage points of return. However, the latest-quarter gain is almost entirely explained by the best batch: after removing it, the result is approximately flat. Because this candidate was identified after inspecting all four quarters, it must be treated as a forward-test candidate rather than a proven out-of-sample edge.
