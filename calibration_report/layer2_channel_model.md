# Layer 2: Channel Model Calibration

## Overview

Verify the statistical channel model's pathloss distribution and geometry SINR CDF
against TR 38.901 section 7.8 calibration references for UMa 3.5 GHz.

## Configuration

| Parameter | Value |
|-----------|-------|
| Scenario | UMa |
| Carrier frequency | 3.5 GHz |
| BS height | 25.0 m |
| UE height | 1.5 m |
| Cell radius | 500.0 m |
| Min distance | 35.0 m |
| Shadow fading (LOS) | 4.0 dB |
| Shadow fading (NLOS) | 6.0 dB |
| Part A UEs | 2000 |
| Part B UEs | 200 |
| Part B slots | 50 |
| Part B config | 4Tx/2Rx, FDD, statistical channel |

## Part A: Coupling Loss CDF

LOS ratio: 8.8%

![Coupling Loss CDF](figures/layer2_channel_cdf.png)

![Pathloss vs Distance](figures/layer2_pathloss_vs_distance.png)

## Part B: Geometry SINR CDF

## CDF Comparison

| Metric | Simulated Median | Reference | Deviation | Status |
|--------|-----------------|-----------|-----------|--------|
| Coupling Loss (dB) | 122.5 | 110.0 | +12.5 | FAIL |
| Geometry SINR (dB) | 22.3 | 6.0 | +16.3 | FAIL |

Pass criterion: CDF median deviation <= 2 dB

## Conclusion

**SOME METRICS EXCEED THRESHOLD**

One or more metrics deviate by more than 2 dB from reference.
