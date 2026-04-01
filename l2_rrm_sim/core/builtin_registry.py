"""内置组件注册

导入此模块会触发所有内置组件向 registry 注册。
SimulationEngine 在 __init__ 开头调用一次即可。
"""

def ensure_loaded():
    """确保所有内置组件已注册到 registry"""
    # 每个模块的 import 会触发 @register_xxx 装饰器执行
    from ..scheduler import pf_scheduler          # noqa: F401 → "pf"
    from ..scheduler import rank_adaptation       # noqa: F401 → "shannon"
    from ..channel import statistical_channel     # noqa: F401 → "statistical"
    from ..channel import channel_estimator       # noqa: F401 → "ls"
    from ..traffic import full_buffer             # noqa: F401 → "full_buffer"
    from ..traffic import ftp_model               # noqa: F401 → "ftp_model3"
    from ..traffic import bursty_traffic          # noqa: F401 → "bursty"
