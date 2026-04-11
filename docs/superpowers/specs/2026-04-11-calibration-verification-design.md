# L2_RRM_Sim 深度校准验证设计

## 目标

对 L2_RRM_Sim 平台进行自底向上的四层校准验证，对标 3GPP 公开基准，产出校准报告。三重驱动：

1. **开源公信力** — 发布前确保无明显 bug
2. **学术可引用** — 对标 3GPP 参考值，偏差可量化
3. **自身信心** — 系统性确认当前版本无隐藏问题

## 验证范围

- **单小区**：深度验证（四层校准）
- **多小区**：Smoke 级（跑通不 crash，干扰趋势合理）
- **PHY 后端**：Legacy PHY (EESM) 为主，不验 Sionna PHY
- **交付物**：校准报告（图表 + 偏差分析），不含 pytest 回归套件

## 验证铁律

1. **发现偏差先汇报，不擅自改代码** — 记录问题、分析可能原因、等用户决策
2. **参数必须对齐** — 仿真配置尽量匹配参考源，差异显式标注
3. **原始数据保留** — 仿真输出的 raw data 保存，方便复查

## 四层校准架构

```
Layer 1: PHY 抽象层  → AWGN BLER vs SNR 对标
Layer 2: 信道模型    → 耦合损耗/SINR CDF 对标
Layer 3: L2 功能     → 频谱效率对标
Layer 4: 端到端 KPI  → 完整系统 KPI 对标
```

每层完成后交付报告，等用户确认后再进入下一层。

---

### Layer 1: PHY 抽象层校准

**目标**：验证 EESM + BLER 表的 AWGN 映射准确性

**对标基准**：
- 3GPP R1-1711982 / R1-1713601：LDPC BLER 曲线
- MATLAB 5G Toolbox "NR PDSCH Throughput" 参考点

**验证方法**：
- 选取 5-6 个代表性 MCS（QPSK 低码率、64QAM 中码率、256QAM 高码率）
- 扫 SNR 范围，记录 BLER
- 绘制 BLER vs SNR 曲线，与参考值叠加对比

**通过标准**：10% BLER 工作点偏差 ≤ 1 dB

---

### Layer 2: 信道模型校准

**目标**：验证统计信道的路损、阴影、快衰落分布

**对标基准**：TR 38.901 v17 §7.8 校准曲线（UMa ISD=500m, 3.5GHz）

**验证方法**：
- 撒 UE 采样耦合损耗（pathloss + shadow fading）→ 绘 CDF
- 采样宽带 SINR（geometry factor）→ 绘 CDF
- 与 TR 38.901 Figure 7.8.x 叠加对比

**通过标准**：CDF 中位数偏差 ≤ 2 dB，形状趋势一致

---

### Layer 3: L2 功能校准

**目标**：验证 CSI 反馈、HARQ、调度器的综合效果

**对标基准**：
- R1-1801360：频谱效率（UMa 4GHz, 4T4R, PF, full buffer）
- Vienna 5G SLS 论文（UMa FDD, 4x2, PF）：SE ~1.8-2.2 bps/Hz

**验证方法**：
- 配置匹配参考场景的参数（ISD、天线、UE 数等）
- 跑 1000+ slots，统计 cell-average SE 和 cell-edge SE（5th percentile）
- 对比参考范围

**通过标准**：Cell-average SE 落入参考范围 ±20%

---

### Layer 4: 端到端 KPI 校验

**目标**：完整系统在典型场景下的 KPI 合理性

**对标基准**：ITU-R M.2412 Dense Urban eMBB 自评结果

**验证方法**：
- 复现 ITU 配置（200MHz TDD, 30kHz, 4x4, PF, 10 UE/cell）
- 对比 DL average SE 和 5th-percentile SE
- 多小区 smoke：7-site 跑通不 crash，干扰导致 SE 下降合理（相比单小区降 30-50%）

**通过标准**：SE 量级正确（同一数量级），趋势合理

---

## 报告交付结构

```
calibration_report/
├── layer1_phy_abstraction.md    — BLER vs SNR 曲线 + 偏差表
├── layer2_channel_model.md      — 耦合损耗/SINR CDF 图 + 对比
├── layer3_l2_functions.md       — 频谱效率 + cell-edge 对标
├── layer4_e2e_kpi.md            — 端到端 KPI 汇总
├── figures/                     — 所有图表 (PNG/SVG)
└── summary.md                   — 总结：各层通过/偏差/待修复项
```

每层报告包含：
1. **配置参数表** — 仿真参数 vs 对标源参数
2. **结果图表** — 仿真曲线 vs 参考曲线叠加图
3. **偏差分析表** — 关键指标数值偏差
4. **结论** — Pass / 偏差待查 / Fail + 原因初步分析
5. **问题清单** — 只记录不改代码，等用户决策

## 执行顺序

```
Layer 1 (PHY) → 交付报告 → 用户确认
    → Layer 2 (信道) → 交付报告 → 用户确认
        → Layer 3 (L2功能) → 交付报告 → 用户确认
            → Layer 4 (端到端) → 交付报告 → 用户确认
                → 汇总报告
```

## 不做的事情

- 不改主代码（除非用户审批）
- 不跑 Sionna PHY 验证（Legacy 为主）
- 不写 pytest 回归套件（交付物是报告）
- 不做多小区深度验证（只 smoke）

## 对标基准参考源

| 来源 | 文档编号 | 用途 |
|------|---------|------|
| 3GPP LDPC BLER | R1-1711982, R1-1713601 | Layer 1 PHY 校准 |
| MATLAB 5G Toolbox | "NR PDSCH Throughput" 示例 | Layer 1 参考点 |
| 3GPP TR 38.901 v17 | §7.8 校准曲线 | Layer 2 信道校准 |
| 3GPP RAN1 | R1-1801360 | Layer 3 频谱效率 |
| Vienna 5G SLS | Pratschner et al., EURASIP 2018 | Layer 3 交叉参考 |
| ITU-R M.2412 | Dense Urban eMBB | Layer 4 端到端 |
| 3GPP TR 37.910 | 评估方法论 | Layer 4 配置参考 |
