"""仿真配置数据类"""

from dataclasses import dataclass, field
from pathlib import Path
import yaml


@dataclass
class SimConfig:
    """仿真级配置"""
    num_slots: int = 10000
    random_seed: int = 42
    warmup_slots: int = 500
    kpi_trim_percent: float = 5.0


@dataclass
class CarrierConfig:
    """NR 载波配置"""
    subcarrier_spacing: int = 30       # kHz (15/30/60/120)
    num_prb: int = 273                 # PRB 数量
    bandwidth_mhz: float = 100.0       # 带宽 (MHz)
    carrier_freq_ghz: float = 3.5      # 载频 (GHz)
    num_pdcch_symbols: int = 2         # PDCCH 占用 OFDM 符号数
    dmrs_type: int = 1                 # DMRS 类型 (1 or 2)
    dmrs_cdm_groups: int = 2           # DMRS CDM groups without data
    num_dmrs_symbols: int = 1          # DMRS 符号数 (1 or 2)

    @property
    def mu(self) -> int:
        """Numerology index"""
        return {15: 0, 30: 1, 60: 2, 120: 3}[self.subcarrier_spacing]

    @property
    def slot_duration_s(self) -> float:
        """Slot 时长 (秒)"""
        return 1e-3 / (2 ** self.mu)

    @property
    def slots_per_second(self) -> float:
        return 1.0 / self.slot_duration_s


@dataclass
class CellConfig:
    """小区配置"""
    num_tx_ant: int = 64               # 发射天线数
    num_tx_ports: int = 4              # CSI-RS 端口数 (rank 上限)
    max_layers: int = 4                # 最大传输层数
    total_power_dbm: float = 46.0      # 总发射功率 (dBm)
    cell_radius_m: float = 500.0       # 小区半径 (m)
    height_m: float = 25.0             # 基站高度 (m)
    scenario: str = "uma"              # 场景 (uma/umi/rma)
    noise_figure_db: float = 5.0       # UE 噪声系数 (dB)


@dataclass
class UEConfig:
    """UE 配置"""
    num_ue: int = 20                   # UE 数量
    num_rx_ant: int = 4                # 接收天线数
    min_distance_m: float = 35.0       # 最小距离 (m)
    max_distance_m: float = 500.0      # 最大距离 (m)
    height_m: float = 1.5              # UE 高度 (m)
    speed_kmh: float = 3.0             # UE 速度 (km/h)


@dataclass
class SchedulerConfig:
    """调度器配置"""
    type: str = "pf"                   # 调度算法类型
    beta: float = 0.98                 # PF 遗忘因子


@dataclass
class LinkAdaptationConfig:
    """链路自适应配置"""
    bler_target: float = 0.1           # 目标 BLER
    olla_delta_up: float = 1.0         # OLLA NACK 步长 (dB)
    olla_offset_min: float = -20.0     # OLLA 偏移下限 (dB)
    olla_offset_max: float = 20.0      # OLLA 偏移上限 (dB)
    mcs_table_index: int = 1           # MCS 表索引


@dataclass
class TrafficConfig:
    """流量模型配置"""
    type: str = "full_buffer"          # 流量类型
    ftp_file_size_bytes: int = 512000  # FTP 文件大小 (bytes)
    ftp_lambda: float = 0.5            # FTP 到达率 (files/s)


@dataclass
class ChannelConfig:
    """信道配置"""
    type: str = "statistical"          # 信道类型 (statistical/ray_tracing)
    scenario: str = "uma"              # 3GPP 场景
    shadow_fading_std_db: float = 4.0  # 阴影衰落标准差 (dB) — LOS UMa


def load_config(yaml_path: str) -> dict:
    """从 YAML 加载配置，返回各配置对象的字典"""
    path = Path(yaml_path)
    with open(path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    config = {
        'sim': SimConfig(**raw.get('sim', {})),
        'carrier': CarrierConfig(**raw.get('carrier', {})),
        'cell': CellConfig(**raw.get('cell', {})),
        'ue': UEConfig(**raw.get('ue', {})),
        'scheduler': SchedulerConfig(**raw.get('scheduler', {})),
        'link_adaptation': LinkAdaptationConfig(**raw.get('link_adaptation', {})),
        'traffic': TrafficConfig(**raw.get('traffic', {})),
        'channel': ChannelConfig(**raw.get('channel', {})),
    }
    return config
