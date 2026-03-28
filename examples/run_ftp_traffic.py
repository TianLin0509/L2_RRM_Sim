#!/usr/bin/env python3
"""FTP Model 3 流量仿真示例"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from l2_rrm_sim.config import load_config
from l2_rrm_sim.core.simulation_engine import SimulationEngine
from l2_rrm_sim.traffic.ftp_model import FTPModel3


def main():
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', 'default_single_cell.yaml'
    )
    config = load_config(config_path)
    config['sim'].num_slots = 10000

    engine = SimulationEngine(config)

    # 替换流量模型为 FTP Model 3
    ftp = FTPModel3(
        file_size_bytes=512_000,    # 500 KB per file
        arrival_rate=2.0,            # 2 files/s per UE
        slot_duration_s=engine.carrier_config.slot_duration_s,
        num_ue=engine.num_ue,
        rng=engine.rng.traffic,
    )
    engine.traffic = ftp

    # 重写 buffer 管理: 使用 FTP 的 dequeue
    original_run_slot = engine.run_slot

    def run_slot_with_ftp(slot_idx):
        result = original_run_slot(slot_idx)
        # FTP dequeue: 跟踪文件完成
        for ue_idx in range(engine.num_ue):
            decoded_bytes = int(result.ue_decoded_bits[ue_idx]) // 8
            if decoded_bytes > 0:
                ftp.dequeue_bytes(ue_idx, decoded_bytes, slot_idx)
        return result

    engine.run_slot = run_slot_with_ftp

    report = engine.run()

    # FTP 时延统计
    latency_stats = ftp.get_file_latency_stats()
    print("\n  FTP 文件传输时延:")
    print(f"    完成文件数: {latency_stats['count']}")
    print(f"    平均时延:   {latency_stats['mean_ms']:.1f} ms")
    print(f"    50%ile:     {latency_stats['p50_ms']:.1f} ms")
    print(f"    95%ile:     {latency_stats['p95_ms']:.1f} ms")
    print(f"    99%ile:     {latency_stats['p99_ms']:.1f} ms")

    return report


if __name__ == '__main__':
    main()
