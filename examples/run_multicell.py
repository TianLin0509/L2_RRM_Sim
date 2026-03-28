#!/usr/bin/env python3
"""多小区仿真示例"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from l2_rrm_sim.config import load_config
from l2_rrm_sim.core.multicell_engine import MultiCellSimulationEngine


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'multicell_7site.yaml'
    )
    print(f"加载配置: {config_path}")
    config = load_config(config_path)

    engine = MultiCellSimulationEngine(
        config,
        num_rings=1,
        num_ue_per_cell=config['ue'].num_ue,
        ici_load_factor=0.8,
    )

    report = engine.run()
    return report


if __name__ == '__main__':
    main()
