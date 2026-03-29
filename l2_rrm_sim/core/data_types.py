"""核心数据类型定义"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SlotContext:
    """每个 TTI 的上下文信息"""
    slot_idx: int                      # 当前 slot 序号
    time_s: float                      # 绝对时间 (秒)
    frame_idx: int = 0                 # 帧号
    subframe_idx: int = 0              # 子帧号
    slot_in_subframe: int = 0          # 子帧内 slot 号
    slot_direction: str = 'D'          # TDD: 'D'=下行, 'U'=上行, 'S'=特殊
    num_dl_symbols: int = 14           # 可用 DL OFDM 符号数


@dataclass
class UEState:
    """单个 UE 的状态"""
    ue_id: int
    position: np.ndarray               # (3,) xyz 坐标 [m]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    buffer_bytes: int = 0              # 当前缓冲区字节数
    selected_rank: int = 1             # 当前秩 (1-4)
    olla_offset_db: float = 0.0        # OLLA SINR 偏移 [dB]
    sinr_eff_db_last: float = 0.0      # 上一次有效 SINR [dB]
    throughput_avg: float = 1.0        # PF 时间平均吞吐量 (bits)
    last_harq_ack: bool = True         # 上一次 HARQ 反馈


@dataclass
class ChannelState:
    """信道状态"""
    pathloss_db: np.ndarray            # (num_ue,) 路径损耗 [dB]
    shadow_fading_db: np.ndarray       # (num_ue,) 阴影衰落 [dB]
    sinr_per_prb: np.ndarray           # (num_ue, max_layers, num_prb) SINR [linear]
    wideband_sinr_db: np.ndarray       # (num_ue,) 宽带 SINR [dB]
    # 实际信道 (Ground Truth), (num_ue, rx_ant, tx_ant, num_prb)
    actual_channel_matrix: np.ndarray = None
    # 估计信道 (含估计误差/反馈延迟), (num_ue, rx_ant, tx_ant, num_prb)
    estimated_channel_matrix: np.ndarray = None


@dataclass
class SchedulingDecision:
    """调度决策 (gNB 下发的调度指令集)"""
    prb_assignment: np.ndarray         # (num_prb,) 主调度 UE ID (-1=未分配)
    ue_mcs: np.ndarray                 # (num_ue,) 每 UE 的 MCS index
    ue_rank: np.ndarray                # (num_ue,) 每 UE 的传输层数
    ue_num_prbs: np.ndarray            # (num_ue,) 每 UE 分配的 PRB 数
    ue_tbs_bits: np.ndarray            # (num_ue,) 每 UE 的 TBS [bits]
    ue_num_re: np.ndarray              # (num_ue,) 每 UE 的有效 RE 数
    
    # --- 扩展支持 MU-MIMO 和 HARQ ---
    mu_groups: list = None             # 长度为 num_prb 的列表，每个元素是该 PRB 配对的 UE ID 列表
    precoders: list = None             # 长度为 num_prb 的列表，每个元素是 (tx_ports, num_paired_ue) 的预编码矩阵
    ue_harq_pid: np.ndarray = None     # (num_ue,) 选定的 HARQ 进程 ID
    ue_rv: np.ndarray = None           # (num_ue,) 选定的冗余版本 (RV 0,2,3,1)


@dataclass
class SlotResult:
    """单 slot 仿真结果"""
    slot_idx: int
    ue_decoded_bits: np.ndarray        # (num_ue,) 每 UE 成功解码比特数
    ue_bler: np.ndarray                # (num_ue,) 每 UE 的 TBLER
    ue_tb_success: np.ndarray          # (num_ue,) 每 UE TB 是否成功 (bool)
    ue_mcs: np.ndarray                 # (num_ue,) 使用的 MCS
    ue_rank: np.ndarray                # (num_ue,) 使用的 rank
    ue_sinr_eff_db: np.ndarray         # (num_ue,) 有效 SINR [dB]
    ue_throughput_inst: np.ndarray     # (num_ue,) 瞬时吞吐量 [bits/s]
    scheduling_decision: SchedulingDecision = None
