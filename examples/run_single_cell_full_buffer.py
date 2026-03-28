#!/usr/bin/env python3
"""单小区全缓冲仿真示例

用法:
    python run_single_cell_full_buffer.py [config_path]
"""

import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from l2_rrm_sim.config import load_config
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.kpi.kpi_plotter import KPIPlotter


def main():
    # 加载配置
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'default_single_cell.yaml'
    )
    print(f"加载配置: {config_path}")
    config = load_config(config_path)

    # 创建仿真引擎
    engine = SimulationEngine(config)

    # 运行仿真
    report = engine.run()

    # 绘制 KPI 图表
    plotter = KPIPlotter(engine.kpi, engine.carrier_config)
    plotter.plot_all()

    return report


if __name__ == '__main__':
    main()
