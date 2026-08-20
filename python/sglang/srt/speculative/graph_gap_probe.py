"""Asynchronous CUDA-event gaps between speculative graph launches."""

from __future__ import annotations

import atexit
import json
import queue
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

from sglang.srt.utils.device_timer import GapTimer


class GraphGapProbe:
    """Record launch-boundary gaps without synchronizing the forward stream.

    ``GapTimer`` queries only completed events.  The reporter hands tiny Python
    records to a bounded writer thread, so neither event timing nor file I/O
    inserts a device synchronization into speculative decode.
    """

    def __init__(self, path: str, max_samples: int):
        if max_samples <= 0:
            raise ValueError("graph-gap max_samples must be positive")
        self.path = Path(path)
        self.max_samples = max_samples
        self.count = 0
        self.dropped = 0
        self.error: Optional[BaseException] = None
        self.queue: queue.Queue[Optional[dict]] = queue.Queue(maxsize=128)
        self.timer = GapTimer(self._report)
        self.thread = threading.Thread(
            target=self._write,
            name=f"graph-gap-{self.path.name}",
            daemon=True,
        )
        self.thread.start()

    def wrap(self, category: str):
        if self.count >= self.max_samples or self.error is not None:
            return nullcontext()
        return self.timer.wrap(metadata={"category": category})

    def cancel(self) -> None:
        self.timer.cancel()

    def _report(self, *, t: float, category: str) -> None:
        if self.count >= self.max_samples:
            return
        record = {
            "schema_version": 1,
            "artifact_type": "speculative_graph_gap",
            "ordinal": self.count,
            "gap_before": category,
            "elapsed_ms": t * 1000.0,
            "timing": "cuda_events_async_query",
        }
        try:
            self.queue.put_nowait(record)
            self.count += 1
        except queue.Full:
            self.dropped += 1

    def _write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as output:
                while True:
                    record = self.queue.get()
                    if record is None:
                        self.queue.task_done()
                        break
                    output.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                    output.flush()
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        self.timer.cancel()
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join()
        if self.error is not None:
            raise RuntimeError("graph-gap writer failed") from self.error
        if self.dropped:
            raise RuntimeError(
                f"graph-gap writer dropped {self.dropped} timing records"
            )


_probes: list[GraphGapProbe] = []
_lock = threading.Lock()


def create_graph_gap_probe(path: Optional[str], max_samples: int) -> Optional[GraphGapProbe]:
    if not path:
        return None
    probe = GraphGapProbe(path, max_samples)
    with _lock:
        _probes.append(probe)
    return probe


def close_graph_gap_probes() -> None:
    with _lock:
        probes = list(_probes)
        _probes.clear()
    for probe in probes:
        probe.close()


atexit.register(close_graph_gap_probes)
