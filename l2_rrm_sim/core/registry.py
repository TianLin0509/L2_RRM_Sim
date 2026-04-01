"""组件注册表 — 可插拔模块化架构

用法:
    # 注册自定义调度器
    from l2_rrm_sim.core.registry import register_scheduler

    @register_scheduler("my_scheduler")
    class MyScheduler(SchedulerBase):
        ...

    # 在配置中指定
    config = {..., 'scheduler': SchedulerConfig(type='my_scheduler'), ...}
"""

from __future__ import annotations
from typing import Type, Dict, Callable, Any

# ── 全局注册表 ──
_REGISTRY: Dict[str, Dict[str, Type]] = {
    "scheduler": {},
    "channel": {},
    "channel_estimator": {},
    "rank_adapter": {},
    "phy": {},
    "traffic": {},
}


def _make_register(category: str):
    """生成注册装饰器工厂"""
    def register(name: str):
        def decorator(cls: Type) -> Type:
            _REGISTRY[category][name] = cls
            return cls
        return decorator
    return register


def _make_getter(category: str):
    """生成查找函数"""
    def get(name: str) -> Type:
        if name not in _REGISTRY[category]:
            available = list(_REGISTRY[category].keys())
            raise KeyError(
                f"Unknown {category} '{name}'. Available: {available}"
            )
        return _REGISTRY[category][name]
    return get


# ── 公开 API ──
register_scheduler = _make_register("scheduler")
register_channel = _make_register("channel")
register_channel_estimator = _make_register("channel_estimator")
register_rank_adapter = _make_register("rank_adapter")
register_phy = _make_register("phy")
register_traffic = _make_register("traffic")

get_scheduler_class = _make_getter("scheduler")
get_channel_class = _make_getter("channel")
get_channel_estimator_class = _make_getter("channel_estimator")
get_rank_adapter_class = _make_getter("rank_adapter")
get_phy_class = _make_getter("phy")
get_traffic_class = _make_getter("traffic")


def list_registered(category: str = None) -> dict:
    """列出已注册的组件"""
    if category:
        return dict(_REGISTRY.get(category, {}))
    return {k: list(v.keys()) for k, v in _REGISTRY.items()}
