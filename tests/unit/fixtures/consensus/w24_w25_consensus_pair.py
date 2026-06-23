"""Sanitized W24→W25 consensus holdings fixture for turnover anchor test.

Derived from state/ledger.db on 2026-06-22 (real run data).
PII: none (tickers + weights only; user_id not stored here).
Source: ledger portfolios+holdings WHERE type='consensus' AND week_id IN ('2026-W24','2026-W25').

Expected one-way turnover: 48.45%  (sum_abs_diff=0.968944, *0.5=0.484472)
Expected breakdown: added=19, removed=17, re-weighted=19
"""

W24_HOLDINGS: dict[str, float] = {
    "CASH":  0.24178571428571427,
    "JPM":   0.06714285714285714,
    "GE":    0.05333333333333334,
    "CMCSA": 0.051666666666666666,
    "UNH":   0.038571428571428576,
    "BRK.B": 0.037142857142857144,
    "AMD":   0.036,
    "META":  0.03333333333333333,
    "CI":    0.032857142857142856,
    "CB":    0.03,
    "LLY":   0.03,
    "GOOGL": 0.025714285714285714,
    "NVDA":  0.024999999999999998,
    "JNJ":   0.022857142857142857,
    "MO":    0.022857142857142857,
    "CRWD":  0.02,
    "KO":    0.02,
    "PM":    0.02,
    "VZ":    0.02,
    "AAPL":  0.018571428571428572,
    "PG":    0.018571428571428572,
    "NOW":   0.018,
    "UBER":  0.0175,
    "WMT":   0.017142857142857144,
    "T":     0.016666666666666666,
    "VRT":   0.016,
    "AXON":  0.015,
    "NEM":   0.014285714285714287,
    "TXN":   0.011428571428571429,
    "CVX":   0.008571428571428572,
    # Zero-weight tickers present in the ledger row:
    "AMZN":  0.0,
    "AVGO":  0.0,
    "CHTR":  0.0,
    "ELV":   0.0,
    "HUM":   0.0,
    "MSFT":  0.0,
    "MSI":   0.0,
    "PLTR":  0.0,
    "TMUS":  0.0,
    "TSM":   0.0,
    "XOM":   0.0,
}

W25_HOLDINGS: dict[str, float] = {
    "JPM":   0.09751012769183388,
    "UNH":   0.05685721731302267,
    "CB":    0.05472507166378432,
    "CMCSA": 0.04975006514889484,
    "BRK.B": 0.0469072042832437,
    "STX":   0.04378005733102745,
    "JNJ":   0.04311672312904219,
    "KLAC":  0.042642912984767,
    "MU":    0.04179005472507166,
    "APP":   0.039800052119115865,
    "LLY":   0.033166710099263225,
    "META":  0.033166710099263225,
    "NVDA":  0.033166710099263225,
    "WDC":   0.0318400416952927,
    "CI":    0.03127146952216247,
    "UBER":  0.0298500390893369,
    "WMT":   0.028428608656511336,
    "ELV":   0.02819170358437374,
    "T":     0.02786003648338111,
    "GOOGL": 0.0255857477908602,
    "KMB":   0.024875032574447422,
    "WSM":   0.024875032574447422,
    "CRWD":  0.022387529317002674,
    "VZ":    0.019900026059557933,
    "VRT":   0.01592002084764635,
    "GDDY":  0.01326668403970529,
    "SPG":   0.01326668403970529,
    "NTAP":  0.011608348534742128,
    "NEM":   0.009950013029778966,
    "NOW":   0.007960010423823174,
    "CLX":   0.006633342019852645,
    "MSFT":  0.006633342019852645,
    "HUM":   0.0033166710099263226,
    # Zero-weight tickers present in the ledger row:
    "AVGO":  0.0,
    "CASH":  0.0,
    "DELL":  0.0,
    "IT":    0.0,
    "PINS":  0.0,
    "TSM":   0.0,
    "TTD":   0.0,
    "VRSN":  0.0,
}

# Anchors verified by hand calculation on 2026-06-22:
# Breakdown definition: ticker "in" a book iff weight > 0; zero-weight ledger rows
# are treated as absent (same as compute_consensus_turnover implementation).
EXPECTED_TURNOVER_PCT = 48.45   # rounded to 2 dp; exact = 48.4472...
EXPECTED_SUM_ABS_DIFF = 0.968944
EXPECTED_N_ADDED = 14       # in W25 (w>0), not in W24 (w=0 or absent)
EXPECTED_N_REMOVED = 11     # in W24 (w>0), not in W25 (w=0 or absent)
EXPECTED_N_REWEIGHTED = 19  # in both with w>0 and weight changed
