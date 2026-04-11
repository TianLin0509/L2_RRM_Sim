# Layer 3: L2 Functions Spectral Efficiency Calibration

## Overview

Verify that the complete L2 stack (PF scheduler, link adaptation, HARQ)
produces reasonable spectral efficiency compared to published references.

## Common Configuration

| Parameter | Value |
|-----------|-------|
| Duplex | FDD |
| Bandwidth | 100 MHz |
| SCS | 30 kHz |
| PRBs | 273 |
| Carrier freq | 3.5 GHz |
| Traffic | Full buffer |
| Channel | Statistical UMa |
| BS height | 25 m |
| UE height | 1.5 m |
| UE distance | [35, 500] m |
| Cell radius | 500 m |
| BS power | 46 dBm |

## Deviations

1. **CSI feedback disabled**: Sionna OLLA `_offset` (torch.Tensor) is incompatible
   with `sinr_to_cqi` (expects scalar float). The engine code at
   `simulation_engine.py:473` subtracts a torch tensor from a numpy scalar,
   producing a tensor that downstream `sinr_to_cqi` cannot handle.
   CSI was disabled to allow simulation to complete. This may affect SE
   (typically CSI improves MCS selection accuracy).

2. **CPU-only execution**: Sionna PHYAbstraction rejects `cuda:0` device string.
   Forced CPU via `torch.cuda.is_available` monkey-patch.

3. **No inter-cell interference (ICI)**: Single-cell simulation without neighboring
   cells. Reference values (Vienna SLS, R1-1801360) include 7-site 21-cell
   deployment with full ICI. Missing ICI significantly inflates SINR and SE.
   This is the primary cause of SE exceeding reference ranges.

## Results

### Scenario 1: FDD 4x2 PF

| Parameter | Value |
|-----------|-------|
| TX antennas | 4 |
| TX ports | 4 |
| Max layers | 2 |
| RX antennas | 2 |
| UEs | 20 |
| Scheduler | PF (beta=0.98) |
| BLER target | 0.1 |
| MCS table | 1 |
| Channel | Statistical UMa |
| Slots | 2000 (warmup: 500) |

| Metric | Value |
|--------|-------|
| Spectral Efficiency | 3.427 bps/Hz |
| Reference range | [1.8, 2.2] bps/Hz |
| Acceptable range (+/-20%) | [1.44, 2.64] bps/Hz |
| Cell throughput | 342.7 Mbps |
| Avg BLER | 0.0713 |
| Avg MCS | 11.9 |
| Jain fairness | 0.7824 |
| PRB utilization | 100.0% |
| Elapsed time | 72.2s |
| **Status** | **FAIL** |
### Scenario 2: FDD 4x4 PF

| Parameter | Value |
|-----------|-------|
| TX antennas | 4 |
| TX ports | 4 |
| Max layers | 4 |
| RX antennas | 4 |
| UEs | 20 |
| Scheduler | PF (beta=0.98) |
| BLER target | 0.1 |
| MCS table | 1 |
| Channel | Statistical UMa |
| Slots | 2000 (warmup: 500) |

| Metric | Value |
|--------|-------|
| Spectral Efficiency | 5.444 bps/Hz |
| Reference range | [2.0, 2.8] bps/Hz |
| Acceptable range (+/-20%) | [1.60, 3.36] bps/Hz |
| Cell throughput | 544.4 Mbps |
| Avg BLER | 0.0927 |
| Avg MCS | 10.8 |
| Jain fairness | 0.6891 |
| PRB utilization | 100.0% |
| Elapsed time | 70.9s |
| **Status** | **FAIL** |


## Figures

![SE Comparison](figures/layer3_se_comparison.png)

## Conclusion

Pass criterion: Cell-average SE within reference range +/-20%

**SOME SCENARIOS FAIL**
