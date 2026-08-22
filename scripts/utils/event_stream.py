from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, deque
import threading


class EventStream:
    def __init__(self, max_per_key: int = 50):
        self._lock = threading.Lock()
        # model_id -> (event_type, module_path) -> deque[dict]
        self._events: Dict[str, Dict[Tuple[str, Optional[str]], deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_per_key))
        )

    def push(self, model_id: str, event: Dict[str, Any]) -> None:
        event_type = event.get("event")
        module_path = event.get("module_path")
        if not event_type:
            return
        key = (event_type, module_path)
        with self._lock:
            self._events[model_id][key].append(event)

    def get_last(
        self, model_id: str, event_type: str, module_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        key = (event_type, module_path)
        with self._lock:
            dq = self._events.get(model_id, {}).get(key)
            if dq and len(dq) > 0:
                return dq[-1]
        return None

    def get_all(
        self, model_id: str, event_type: str, module_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        key = (event_type, module_path)
        with self._lock:
            dq = self._events.get(model_id, {}).get(key)
            if dq:
                return list(dq)
        return []
