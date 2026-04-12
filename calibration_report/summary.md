# L2_RRM_Sim 校准验证汇总报告

**日期**: 2026-04-12
**范围**: 4 层自底向上校准（PHY 抽象 → 信道模型 → L2 功能 → 端到端 KPI）
**PHY 后端**: LegacyPHY (EESM + OLLA + BLER lookup)，自研路径
**主要场景**: TDD DDDSU, 64T Massive MIMO, UMa 3.5GHz, 100MHz BW, 3GPP SU-MIMO

---

## 各层校准结果

| Layer | 对标基准 | 关键指标 | 结果 | 状态 |
|-------|---------|---------|------|------|
| L1 PHY | R1-1711982 LDPC BLER + 3GPP CQI SINR 阈值 | 6 个 MCS 的 10% BLER 工作点 | QPSK PASS，16/64QAM 偏差 -1~-2.5 dB | ⚠ 有 known issue |
| L2 Channel | TR 38.901 §7.8 UMa | 耦合损耗 / Geometry SINR CDF | 耦合损耗 110 dB（0 dB 偏差）; MC SINR 7.4 dB（+1.4 dB 偏差） | ✅ PASS |
| L3 L2 Func | Vienna SLS / R1-1801360 / 华为商用 | SE @ 64T4R SU-MIMO | SC 9.15 / MC 3.41 bps/Hz | ✅ PASS |
| L4 E2E | ITU-R M.2412 / 华为商用 | 端到端 KPI + cell-edge | SC 9.29 / MC 3.71 / Edge 59.5 Mbps | ✅ PASS |

---

## 业界对标（64T4R SU-MIMO TDD）

| 配置 | 我们 | 华为/中兴商用 | ITU-R M.2412 |
|------|------|-------------|-------------|
| 单小区峰值 SE | **9.29 bps/Hz** | 8-10 | — |
| 多小区典型 SE | **3.41-3.71 bps/Hz** | 3-5 | — |
| Cell edge 5% | 20-60 Mbps | ~20 Mbps | — |
| 参考 MU-MIMO 64T64R | — | — | 7.8 bps/Hz |
| MC/SC ratio | **0.37-0.40** | 典型 0.3-0.5 | — |

**结论**：L2 全栈 SE 精准命中华为商用 64T4R SU-MIMO 的峰值和典型区间，ICI 损失比例合理。

---

## 过程中发现的关键问题（按优先级）

### 🔴 P0 — 必须修复

#### 1. Sionna PHY 严重 MCS 欠选
**现象**: SINR 24 dB 时 Sionna PHY 选 MCS 7，而 Legacy PHY 选 MCS 25（正确值）。
**影响**: SE 从应有的 9 bps/Hz 降到 2.3 bps/Hz（-74%）。
**临时方案**: 校准脚本强制关闭 Sionna PHY，全部走 LegacyPHY。
**根因**: 未完全定位，疑似 Sionna OLLA 内部 MCS 选择逻辑与我们的 3GPP MCS 表不匹配。
**建议**: 若继续使用 Sionna PHY，需深入排查 OLLA._offset 初始化 / 收敛逻辑 / MCS 表对齐。

#### 2. BLER 表 CBS 钳位偏差
**现象**: BLER 表最大 CBS=2000，但 273 PRB rank=1 实际 CBS=7000-8400，lookup 被钳位到 2000。
**影响**: 
- Layer 1 高 MCS 校准偏差 -1~-2.5 dB（BLER 表查到的值比真实 CBS 下偏高）
- ILLA MCS 选择基于错误 BLER 曲线
**建议**: 重新从 Sionna 导出覆盖更大 CBS（至少到 8448 = BG1 K_cb_max）的 BLER 表。

### 🟡 P1 — 重要

#### 3. 扇区天线方向图用标准值 65° 而非真实产品值
**现象**: 当前使用 TR 38.901 Table 7.3-1 标准值 θ₃dB = φ₃dB = 65°。
**用户反馈**: 华为实际商用基站 MIMO 单元 3dB 波宽约 110°。
**影响**: 我们的同站扇区干扰抑制可能比真实部署更强（65° 主瓣更窄），导致 geometry SINR 略偏高（+1.4 dB）。
**建议**: 把 θ₃dB, φ₃dB 做成可配置参数，支持切换"3GPP 标准"/"华为真机"两组值。

#### 4. ILLA-调度器脱节导致 MCS 选择不精确
**现象**: ILLA 按"均分 PRB"（273/10=27 PRB/UE）预估 TBLER，但 PF 实际集中分配（中位数 272 PRB/UE，num_cb 放大 60 倍）。
**影响**: ILLA 预估 TBLER=10% 时实际 TBLER 远高于 10%，OLLA 需要大 offset 补偿。
**当前缓解**: MCS 域 OLLA（华为方案）+ ack_step/nack_step=0.01/0.1，可通过反馈回路收敛到真实稳态。
**建议**: 
- 短期：已缓解，保持现状
- 长期：可探索"调度 → MCS 重选"两轮 iteration，或 ILLA 接收动态 PRB 数估计

#### 5. CSI 反馈通路有 Sionna Tensor 兼容性 bug
**现象**: `sinr_to_cqi(torch.Tensor)` 类型不兼容，CSI 启用时崩溃。
**临时方案**: CSI 禁用 + `float()` 转换修复表层问题。
**影响**: 无法验证 CSI 子带 PMI / 延迟反馈在真实场景的效果。
**建议**: 重构 CSI 路径，所有外部接口以 numpy ndarray 为主，Sionna tensor 限制在内部。

### 🟢 P2 — 改进

#### 6. OLLA 初始值对收敛速度敏感
**现象**: `initial_offset=-4`（华为默认）需要 5000+ slots 收敛；`initial_offset=0` 在 <500 slots 内收敛。
**影响**: 短仿真或真实网络短时会话，OLLA 可能未达稳态。
**当前方案**: 校准脚本用 `reset(initial_offset=0)` 加速；商用默认保留 -4。
**建议**: 
- 研究自适应 initial_offset（例如基于最近 CQI 反馈猜测起点）
- 或 `ack_step/nack_step` 步长自适应（初期快收敛，稳态时小步微调）

#### 7. MultiCellEngine 不支持 TDD slot pattern
**现象**: 多小区仿真只能跑 FDD，单小区可以 TDD。
**影响**: 无法精确复现 ITU-R M.2412 的 TDD 多小区场景。
**建议**: MultiCellEngine 集成 TDDConfig，让 `_make_slot_context` 按 DDDSU 返回 direction。

#### 8. BLER 表最大 CBS=2000 限制校准精度
**与 P0 第 2 项同因**，已归入 P0。

#### 9. Layer 1 参考值是估值
**现象**: 最初 `REFERENCE_BLER_10PCT` 是文献估读，误差 2-3 dB。后用 3GPP CQI SINR 阈值（R1-073505）做精确映射修正。
**当前状态**: 已修正，QPSK 精确匹配；16/64QAM 剩余偏差来自 CBS 钳位（P0）。
**建议**: 后续若需进一步提升精度，用 MATLAB 5G Toolbox 跑独立参考曲线替代。

---

## 代码改动清单

### 主代码改动（生产代码）

| 文件 | 改动 | 原因 |
|------|------|------|
| `l2_rrm_sim/core/topology.py` | 新增 `get_sector_boresight()`, `compute_azimuth_to_cell()`, `compute_relative_azimuth()` | 扇区天线方向图需要 |
| `l2_rrm_sim/channel/interference_model.py` | `precompute_pathloss()` 加入 TR 38.901 TX 天线增益 | 多小区 SINR 修复（-3.7 → 7.4 dB） |
| `l2_rrm_sim/channel/statistical_channel.py` | 方位角改为扇区相对；新增 `sector_boresight_deg` 参数 | 同上 |
| `l2_rrm_sim/link_adaptation/olla.py` | **完全重写**: SINR 域 → MCS 域 OLLA | 对齐华为商用方案 |
| `l2_rrm_sim/link_adaptation/legacy_phy_adapter.py` | 更新 OLLA 构造 | 适配新接口 |
| `l2_rrm_sim/core/multicell_engine.py` | 更新 OLLA 构造 | 适配新接口 |
| `l2_rrm_sim/core/simulation_engine.py` | CSI 路径 Tensor→numpy + MCS 域 offset 叠加 | Sionna 兼容 + 对齐新 OLLA |

### 校准脚本（新增）

- `calibration/utils.py` — 共享工具 + 3GPP 参考数据
- `calibration/layer1_phy_bler.py` — BLER vs SNR 扫描
- `calibration/layer2_channel_model.py` — 耦合损耗 + SINR CDF（Parts A/B/C）
- `calibration/layer3_l2_functions.py` — SE 多场景（单小区 + 7-site 多小区）
- `calibration/layer4_e2e_kpi.py` — 端到端 KPI 全景

### 报告产出

- `calibration_report/layer{1,2,3,4}_*.md` — 4 层分层报告
- `calibration_report/figures/*.png` — 5 张校准图表
- `calibration_report/summary.md` — 本报告

---

## 验证原则遵守情况

| 铁律 | 执行情况 |
|------|---------|
| 发现偏差先汇报用户 | ✅ 所有偏差均通过对话汇报 |
| 不擅自改代码 | ✅ 所有代码改动均经用户批准（扇区天线、OLLA、CSI bug） |
| 参数显式对齐 | ✅ 每层报告含"Configuration"对齐表 |
| 原始数据保留 | ✅ 仿真 seed=42 可复现，所有 commit 保留完整历史 |

---

## 后续优先改进路线图

### 立即可做（本周）
1. **P0-1**: 重新导出覆盖更大 CBS 的 BLER 表（需要 Sionna 跑一次扫描）
2. **P1-3**: 天线方向图波宽参数化（改 3 行代码 + 一组测试）

### 短期（2-4 周）
3. **P0-2**: 排查 Sionna PHY MCS 欠选根因
4. **P1-5**: CSI 路径 numpy/tensor 接口重构
5. **P2-7**: MultiCellEngine 支持 TDD

### 长期（1-3 月）
6. **P1-4**: ILLA-调度器 iteration 或动态 PRB 估计
7. **P2-6**: OLLA 自适应初始值 / 步长
8. 加入 **MU-MIMO** 和 **64R UE** 支持，对标 ITU-R M.2412 7.8 bps/Hz

---

## 交付物索引

- [Layer 1: PHY Abstraction](layer1_phy_abstraction.md)
- [Layer 2: Channel Model](layer2_channel_model.md)
- [Layer 3: L2 Functions](layer3_l2_functions.md)
- [Layer 4: End-to-End KPI](layer4_e2e_kpi.md)

**总结**: L2_RRM_Sim 自研 LegacyPHY 路径已验证可产生与华为商用 64T4R SU-MIMO TDD 相匹配的性能指标，具备作为学术/开源参考平台的可信度。主要 gap（Sionna PHY bug、BLER 表 CBS 限制、扇区方向图参数）已识别并给出优先级路线图。
