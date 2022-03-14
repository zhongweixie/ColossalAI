from pathlib import Path
from typing import Union
from colossalai.engine import Engine
from torch.utils.tensorboard import SummaryWriter
from colossalai.engine.ophooks import MemTracerOpHook
from colossalai.utils.profiler import BaseProfiler


class MemProfiler(BaseProfiler):
    """Wraper of MemOpHook, used to show GPU memory usage through each iteration

    """

    def __init__(self, engine: Engine, warmup: int = 50, refreshrate: int = 10) -> None:
        super().__init__(profiler_name="MemoryProfiler", priority=0)
        self._mem_tracer = MemTracerOpHook(warmup=warmup, refreshrate=refreshrate)
        self._engine = engine

    def enable(self) -> None:
        self._engine.add_hook(self._mem_tracer)

    def disable(self) -> None:
        self._engine.remove_hook(self._mem_tracer)

    def to_tensorboard(self, writer: SummaryWriter) -> None:
        stats = self._mem_tracer.async_mem_monitor.state_dict()['mem_stats']
        for i in range(len(stats)):
            writer.add_scalar(
                "memory_usage/GPU",
                stats[i],
                i
            )

    def to_file(self, log_dir: Union[str, Path]) -> None:
        if isinstance(log_dir, str):
            log_dir = Path(log_dir)
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)
        
        self._mem_tracer.save_results(log_dir)

    def show(self) -> None:
        pass

    def get_latest(self) -> float:
        pass

    def get_avg(self) -> float:
        pass
