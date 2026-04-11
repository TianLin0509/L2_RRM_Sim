# Layer 3: L2 Functions — TDD Massive MIMO Calibration

## Overview

Verify the L2 stack (PF scheduler, link adaptation, HARQ) spectral efficiency
in TDD Massive MIMO configuration (64T, DDDSU pattern).

## Common Configuration

| Parameter | Value |
|-----------|-------|
| Duplex | TDD DDDSU |
| Bandwidth | 100 MHz |
| SCS | 30 kHz |
| PRBs | 273 |
| Carrier freq | 3.5 GHz |
| TX antennas | 64 |
| BS power | 46 dBm |
| Traffic | Full buffer |
| Channel | Statistical UMa |
| Slots | 2000 (warmup: 500) |
| Scheduler | PF (beta=0.98) |
| BLER target | 0.1 |

## Known Deviations

1. **CSI feedback disabled**: Engine bug — OLLA `_offset` (torch.Tensor) incompatible
   with `sinr_to_cqi` scalar interface.
2. **Single-cell, no ICI**: SE is higher than multi-cell deployment.
3. **SU-MIMO only**: No MU-MIMO pairing, max layers limited by min(TX ports, RX ant).

## Results

### Scenario 1: TDD 64T4R 4-layer

| Parameter | Value |
|-----------|-------|
| TX antennas | 64 |
| TX ports | 4 |
| Max layers | 4 |
| RX antennas | 4 |
| UEs | 20 |
| Duplex | TDD DDDSU |
| DL ratio | ~80% (3D + 0.7S out of 5 slots) |

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| Spectral Efficiency | 2.299 bps/Hz | [2.0, 6.0] | OK |
| Cell throughput | 229.9 Mbps | — | — |
| Cell edge (5%) | 3.9 Mbps | — | — |
| Avg BLER | 0.0345 | [0.05, 0.15] | OUT |
| Avg MCS | 6.7 | — | — |
| Avg Rank | 4.00 | — | — |
| Jain fairness | 0.8540 | — | — |
| PRB utilization | 80.0% | >70% | OK |
| **Overall** | | | **FAIL** |
### Scenario 2: TDD 64T2R 2-layer

| Parameter | Value |
|-----------|-------|
| TX antennas | 64 |
| TX ports | 4 |
| Max layers | 2 |
| RX antennas | 2 |
| UEs | 20 |
| Duplex | TDD DDDSU |
| DL ratio | ~80% (3D + 0.7S out of 5 slots) |

| Metric | Value | Criterion | Status |
|--------|-------|-----------|--------|
| Spectral Efficiency | 1.222 bps/Hz | [1.5, 4.0] | OUT |
| Cell throughput | 122.2 Mbps | — | — |
| Cell edge (5%) | 2.6 Mbps | — | — |
| Avg BLER | 0.0162 | [0.05, 0.15] | OUT |
| Avg MCS | 6.5 | — | — |
| Avg Rank | 2.00 | — | — |
| Jain fairness | 0.8955 | — | — |
| PRB utilization | 80.0% | >70% | OK |
| **Overall** | | | **FAIL** |


## Figures

![SE Comparison](figures/layer3_se_comparison.png)

## Conclusion

**SOME METRICS OUT OF RANGE** — see individual scenario results.
