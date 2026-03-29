#!/usr/bin/env python3
"""使用 Sionna 3GPP 信道的单小区仿真

需要在 .venv312 环境下运行:
  .venv312/Scripts/python.exe examples/run_sionna_channel.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from l2_rrm_sim.config.sim_config import (
    SimConfig, CarrierConfig, CellConfig, UEConfig,
    SchedulerConfig, LinkAdaptationConfig, TrafficConfig, ChannelConfig
)
from l2_rrm_sim.core.simulation_engine import SimulationEngine


def main():
    config = {
        'sim': SimConfig(num_slots=2000, random_seed=42, warmup_slots=200),
        'carrier': CarrierConfig(
            subcarrier_spacing=30, num_prb=273,
            bandwidth_mhz=100, carrier_freq_ghz=3.5,
        ),
        'cell': CellConfig(
            num_tx_ant=32, num_tx_ports=4, max_layers=4,
            total_power_dbm=46.0, cell_radius_m=500, height_m=25.0,
            scenario='uma',
        ),
        'ue': UEConfig(num_ue=10, num_rx_ant=1, speed_kmh=3.0),
        'scheduler': SchedulerConfig(type='pf', beta=0.98),
        'link_adaptation': LinkAdaptationConfig(bler_target=0.1),
        'traffic': TrafficConfig(type='full_buffer'),
        'channel': ChannelConfig(type='sionna', scenario='uma'),
    }

    print("=" * 60)
    print("  Sionna 3GPP TR 38.901 UMa 信道 + L2 RRM 仿真")
    print("=" * 60)

    engine = SimulationEngine(config)
    report = engine.run()
    return report


if __name__ == '__main__':
    main()
