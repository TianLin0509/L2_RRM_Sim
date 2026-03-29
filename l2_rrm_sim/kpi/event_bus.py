"""事件驱动 KPI 系统 (Observer Pattern)

仿真主循环只发事件，异步监听器负责数据记录和导出。

事件类型:
- SlotCompleted: 每 slot 结束时触发 (含完整 SlotResult)
- SimulationStarted: 仿真开始
- SimulationCompleted: 仿真结束

监听器:
- ArrayCollector: 写入内存数组 (替代原 KPICollector 的同步写入)
- CSVLogger: 异步写入 CSV 文件
- HDF5Logger: 高性能批量写入 HDF5
"""

import numpy as np
import threading
import queue
import csv
import time
from dataclasses import dataclass
from typing import Callable, Any
from pathlib import Path


# ============================================================
# 事件类型
# ============================================================

@dataclass
class SimEvent:
    """仿真事件基类"""
    event_type: str
    slot_idx: int = -1
    data: Any = None
    timestamp: float = 0.0


class EventBus:
    """事件总线

    支持同步和异步监听器。
    异步监听器通过后台线程 + 队列处理，不阻塞主循环。
    """

    def __init__(self):
        self._sync_listeners = {}    # {event_type: [callback, ...]}
        self._async_queue = queue.Queue(maxsize=10000)
        self._async_listeners = {}   # {event_type: [callback, ...]}
        self._async_thread = None
        self._running = False

    def on(self, event_type: str, callback: Callable, async_mode: bool = False):
        """注册监听器

        Args:
            event_type: 事件类型 ('slot_completed', 'sim_started', 'sim_completed')
            callback: 回调函数 (event: SimEvent) -> None
            async_mode: 是否异步执行
        """
        if async_mode:
            self._async_listeners.setdefault(event_type, []).append(callback)
        else:
            self._sync_listeners.setdefault(event_type, []).append(callback)

    def emit(self, event: SimEvent):
        """发射事件"""
        event.timestamp = time.time()

        # 同步监听器: 立即执行
        for cb in self._sync_listeners.get(event.event_type, []):
            cb(event)

        # 异步监听器: 入队
        if self._async_listeners.get(event.event_type):
            try:
                self._async_queue.put_nowait(event)
            except queue.Full:
                pass  # 队列满时丢弃 (避免阻塞主循环)

    def start(self):
        """启动异步处理线程"""
        if self._async_thread is not None:
            return
        self._running = True
        self._async_thread = threading.Thread(target=self._async_worker, daemon=True)
        self._async_thread.start()

    def stop(self):
        """停止异步处理"""
        self._running = False
        if self._async_thread:
            self._async_queue.put(None)  # 唤醒线程
            self._async_thread.join(timeout=5)
            self._async_thread = None

    def _async_worker(self):
        """后台线程: 消费事件队列"""
        while self._running:
            try:
                event = self._async_queue.get(timeout=1)
                if event is None:
                    break
                for cb in self._async_listeners.get(event.event_type, []):
                    try:
                        cb(event)
                    except Exception:
                        pass  # 异步回调不能崩溃主线程
            except queue.Empty:
                continue


# ============================================================
# 内置监听器
# ============================================================

class CSVLogger:
    """CSV 日志记录器 (异步)

    每 slot 记录一行: slot_idx, cell_throughput, avg_bler, avg_mcs, ...
    """

    def __init__(self, output_path: str = "sim_log.csv"):
        self._path = Path(output_path)
        self._file = None
        self._writer = None
        self._header_written = False

    def on_sim_started(self, event: SimEvent):
        self._file = open(self._path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'slot_idx', 'cell_throughput_bits', 'num_scheduled_ue',
            'avg_mcs', 'avg_bler', 'avg_sinr_db'
        ])
        self._header_written = True

    def on_slot_completed(self, event: SimEvent):
        if self._writer is None:
            return
        d = event.data
        if d is None:
            return
        self._writer.writerow([
            event.slot_idx,
            int(np.sum(d.ue_decoded_bits)),
            int(np.sum(d.scheduling_decision.ue_num_prbs > 0)) if d.scheduling_decision else 0,
            f"{np.mean(d.ue_mcs):.1f}",
            f"{np.mean(d.ue_bler):.4f}",
            f"{np.mean(d.ue_sinr_eff_db):.1f}",
        ])

    def on_sim_completed(self, event: SimEvent):
        if self._file:
            self._file.close()
            self._file = None

    def register(self, bus: EventBus):
        """注册到事件总线"""
        bus.on('sim_started', self.on_sim_started, async_mode=True)
        bus.on('slot_completed', self.on_slot_completed, async_mode=True)
        bus.on('sim_completed', self.on_sim_completed, async_mode=True)


class PerSlotStatsCollector:
    """per-slot 统计收集器 (同步, 轻量)

    只收集聚合统计, 不存原始 per-UE 数据。
    用于实时进度监控。
    """

    def __init__(self):
        self.slot_throughput = []
        self.slot_bler = []
        self.slot_mcs = []

    def on_slot_completed(self, event: SimEvent):
        d = event.data
        if d is None:
            return
        self.slot_throughput.append(int(np.sum(d.ue_decoded_bits)))
        sched = d.scheduling_decision
        if sched is not None:
            mask = sched.ue_num_prbs > 0
            if np.any(mask):
                self.slot_bler.append(float(1.0 - np.mean(d.ue_tb_success[mask])))
                self.slot_mcs.append(float(np.mean(d.ue_mcs[mask])))

    def register(self, bus: EventBus):
        bus.on('slot_completed', self.on_slot_completed, async_mode=False)
