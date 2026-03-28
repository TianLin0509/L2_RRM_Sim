"""枚举定义"""

from enum import Enum, IntEnum


class Direction(Enum):
    """传输方向"""
    DOWNLINK = "DL"
    UPLINK = "UL"


class ChannelScenario(Enum):
    """信道场景 (3GPP TR 38.901)"""
    UMA = "uma"      # Urban Macro
    UMI = "umi"      # Urban Micro
    RMA = "rma"      # Rural Macro


class TrafficType(Enum):
    """流量模型类型"""
    FULL_BUFFER = "full_buffer"
    FTP_MODEL3 = "ftp_model3"
    BURSTY = "bursty"


class SchedulerType(Enum):
    """调度器类型"""
    PF = "pf"            # Proportional Fair
    RR = "rr"            # Round Robin
    MAX_CIR = "max_cir"  # Max C/I (Best CQI)


class DMRSType(IntEnum):
    """DMRS 类型"""
    TYPE1 = 1
    TYPE2 = 2


class MCSTableIndex(IntEnum):
    """MCS 表索引 (TS 38.214)"""
    TABLE1 = 1   # 64QAM, 默认
    TABLE2 = 2   # 256QAM
    TABLE3 = 3   # 低频谱效率 64QAM (URLLC)
