# L2_RRM_Sim 工作进度记录

更新时间：2026-04-12

---

## 🎯 后续改进路线图（来自 2026-04-12 四层校准）

完整分析见 `calibration_report/summary.md`。按优先级：

### 🔴 P0 — 必须修复

- [ ] **P0-0: MultiCellEngine 支持 TDD DDDSU slot pattern**
  - 当前多小区只能 FDD，用户主场景是 TDD Massive MIMO
  - 需要集成 TDDConfig 到 `_make_slot_context` 和 `_run_cell_slot`
- [ ] **P0-1: BLER 表扩展到大 CBS (>=8448)**
  - 当前表最大 CBS=2000，实际 273 PRB rank=1 CBS=7000-8400，被钳位
  - 重新从 Sionna 导出覆盖更大 CBS 的 PDSCH_table*.json
- [ ] **P0-2: 排查 Sionna PHY MCS 严重欠选**
  - SINR 24 dB 时 Sionna 选 MCS 7 (应为 25)，SE 降 74%
  - 疑似 Sionna OLLA _offset 初始化/收敛逻辑不匹配

### 🟡 P1 — 重要

- [ ] **P1-3: 天线方向图波宽参数化**
  - 当前 3GPP 标准值 θ₃dB=65°，华为实际产品 110°
  - `antenna_gain_3gpp_element` 支持从 config 读取波宽
- [ ] **P1-4: ILLA-调度器 iteration 或动态 PRB 估计**
  - ILLA 用均分假设（27 PRB/UE），实际集中分配（272 PRB/UE）
  - 探索两轮 iteration 或动态 PRB 数估计
- [ ] **P1-5: CSI 路径 Sionna Tensor 兼容性**
  - `sinr_to_cqi(torch.Tensor)` 类型不兼容
  - 重构 CSI 路径接口统一为 numpy ndarray

### 🟢 P2 — 改进

- [ ] **P2-6: OLLA 自适应初始值/步长**
  - 商用默认 init=-4 需 5000+ slots 收敛
  - 探索基于最近 CQI 反馈猜测起点，或步长自适应
- [ ] **P2-7: Layer 1 参考值用 MATLAB 5G Toolbox 精确曲线替代**
  - 当前用 3GPP CQI SINR 阈值映射，有码率不匹配插值
- [ ] **P2-8: 加入 MU-MIMO + 64R UE 支持**
  - 对标 ITU-R M.2412 Dense Urban eMBB 7.8 bps/Hz

---

## 项目简介

5G NR L2 RRM 无线系统级仿真平台。支持单小区/多小区、PF调度、HARQ、CSI反馈、链路自适应、多种业务模型，提供 Streamlit Web UI。

GitHub: https://github.com/TianLin0509/L2_RRM_Sim
当前分支: master
最新 commit: `25ee2e9`

---

## 本次会话已完成的工作

### 第一步：代码同步
- 将本地修改（SRS管理器、PF调度增强、BLER表重构、信道估计优化）推送到 GitHub
- commit: `be38bf9`

### 第二步：代码审查 + 优化计划制定
对全项目 17 个核心文件做了深度审查，发现并分级 11 类问题（P0~P4）。

### 第三步：全量修复（已全部完成）

#### P0 — 阻塞性 Bug ✅
| # | 文件 | 问题 | 修复方式 |
|---|------|------|---------|
| Fix 1 | `channel/statistical_channel.py:119` | 多层 SINR 全为零，只填了 layer 0 | 向量化批量 SVD，正确计算每层 SINR |
| Fix 2 | `core/simulation_engine.py:447` | HARQ 重传进程被 pop 后若 UE 未调度则永久丢失 | 新增 `peek_retx_info` + `consume_retx` 两阶段接口 |
| Fix 3 | `core/simulation_engine.py:279` | `run()` 中引用不存在的 `config` 变量 | 改为传 `None` |

#### P1 — 算法正确性 ✅
| # | 文件 | 问题 | 修复方式 |
|---|------|------|---------|
| Fix 4 | `csi/csi_feedback.py:178` | RI 选择中 `num_prb` 在分子分母相消，噪声基准错误 | 去掉多余的 `num_prb` 因子 |
| Fix 5 | `scheduler/rank_adaptation.py:37` | `sinr_perb` typo + 错误的 `/r` 功率分摊模型 | 修复 typo，去掉 `/r` |
| Fix 6 | `harq/harq_entity.py:160` | `max_retx` off-by-one，允许 5 次总传输超出 3GPP | 改为 `>= max_retx - 1` |
| Fix 7 | `channel/channel_estimator.py:52` | 噪声标准差用实部均值，低估约 3 倍 | 改为 `np.abs(h_actual)` |

#### P2 — 功能完整性 ✅
| # | 文件 | 问题 | 修复方式 |
|---|------|------|---------|
| Fix 8 | `core/multicell_engine.py:226` | `KPICollector.collect` 未传 buffer 数据，多小区体验速率全为 0 | 在 dequeue 前后记录快照，传入 collect |
| Fix 9 | `csi/srs_manager.py:83` | 延迟队列清理重复 pop 已消费的 slot，丢弃其他到期测量 | 循环中跳过 `latest_ready` |

#### P3 — 性能 ✅
| # | 文件 | 问题 | 修复方式 |
|---|------|------|---------|
| Fix 10 | `scheduler/mu_mimo_scheduler.py:108` | rank 统计中 O(num_prb × num_ue) Python 循环 | 向量化，用 `group_sizes > 1` 快速定位 MU PRB |

#### P4 — 测试补全 ✅
| # | 文件 | 内容 |
|---|------|------|
| Fix 11 | `tests/test_smoke.py` | 补充 buffer 非负断言；新增 `test_harq_max_retx`、`test_harq_peek_does_not_consume`、`test_rank_selection_multi_layer` |

### 第四步：推送
- commit `f31df74`: P0-P4 全量修复
- commit `25ee2e9`: MU-MIMO 性能优化
- 均已推送到 GitHub master

### 测试结果
```
12 passed in 16.88s  ✅ 全部通过
```

---

## 尚未处理的已知问题

### I5 — csi_feedback.py — PMI 只用宽带平均信道
没有子带 PMI，不符合 TS 38.214 §5.2.2.2。
**建议**：实现子带 PMI 选择（可选增强）。

### I6 — harq_entity.py — Chase Combining SINR 累加过于简化
应用 IR-HARQ 的 LLR 合并，而非简单线性累加。
**建议**：改为对数域 LLR 合并（可选增强）。

---

## 已关闭问题

| # | 状态 | 说明 |
|---|------|------|
| I1 | ✅ 已修复 | 删除 simulation_engine.py 死代码块 |
| I2 | ✅ 已修复 | 多小区引擎复用完整复高斯信道矩阵+批量SVD |
| I3 | ✅ 已修复 | legacy PHY `_last_sinr_eff` 从全1改为实际EESM值 |
| I4 | ✅ 误报 | RMa 断点距离公式正确，d_bp≈2.7km符合3GPP规范 |
| I7 | ✅ 已修复 | rank_adaptation.py SVD预编码写法统一为 `.conj().T` |

---

## 下一步建议任务（优先级排序）

1. **子带 PMI I5**（较大，可选增强）
2. **IR-HARQ LLR 合并 I6**（较大，可选增强）

---

## 项目关键文件速查

| 模块 | 文件 |
|------|------|
| 仿真主引擎 | `l2_rrm_sim/core/simulation_engine.py` |
| 多小区引擎 | `l2_rrm_sim/core/multicell_engine.py` |
| 统计信道 | `l2_rrm_sim/channel/statistical_channel.py` |
| PF 调度器 | `l2_rrm_sim/scheduler/pf_scheduler.py` |
| MU-MIMO 调度 | `l2_rrm_sim/scheduler/mu_mimo_scheduler.py` |
| Rank 自适应 | `l2_rrm_sim/scheduler/rank_adaptation.py` |
| HARQ 实体 | `l2_rrm_sim/harq/harq_entity.py` |
| HARQ 管理器 | `l2_rrm_sim/harq/harq_buffer.py` |
| CSI 反馈 | `l2_rrm_sim/csi/csi_feedback.py` |
| SRS 管理 | `l2_rrm_sim/csi/srs_manager.py` |
| 信道估计 | `l2_rrm_sim/channel/channel_estimator.py` |
| KPI 采集 | `l2_rrm_sim/kpi/kpi_collector.py` |
| Web UI | `app.py` |
| 测试 | `tests/test_smoke.py` |

## 运行测试

```bash
cd C:\Users\lintian\Documents\GitHub\L2_RRM_Sim
.venv312/Scripts/python.exe -m pytest tests/test_smoke.py -v
```
