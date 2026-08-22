import logging
from typing import Set, Dict, Callable, Any
from fastapi import WebSocket, APIRouter
from datetime import datetime
import asyncio
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Maximum queued messages before backpressure kicks in
_MAX_QUEUE_SIZE = 2000


class MessageBus:
    def __init__(self):
        self.subscribed_clients: Dict[str, Set[Callable]] = {}
        self.websocket_clients: Set[WebSocket] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
        self.running = False
        self._dropped_messages = 0
        # Event loop used for publishing from inference worker threads.
        self._loop: asyncio.AbstractEventLoop | None = None

    def subscribe(self, client_token: str, callback: Callable):
        if client_token not in self.subscribed_clients:
            self.subscribed_clients[client_token] = set()
        self.subscribed_clients[client_token].add(callback)

    def unsubscribe(self, client_token: str, callback: Callable):
        if (
            client_token in self.subscribed_clients
            and callback in self.subscribed_clients[client_token]
        ):
            self.subscribed_clients[client_token].remove(callback)

    def add_websocket(self, websocket: WebSocket):
        self.websocket_clients.add(websocket)
        logger.debug("WebSocket client added (total: %d)", len(self.websocket_clients))

    def remove_websocket(self, websocket: WebSocket):
        self.websocket_clients.discard(websocket)
        logger.debug("WebSocket client removed (total: %d)", len(self.websocket_clients))

    async def publish(self, client_token: str, message: Any):
        try:
            self.message_queue.put_nowait((client_token, message))
        except asyncio.QueueFull:
            self._dropped_messages += 1
            if self._dropped_messages % 100 == 1:
                logger.warning(
                    "MessageBus queue full (%d max), dropped %d message(s) total",
                    _MAX_QUEUE_SIZE,
                    self._dropped_messages,
                )

    def publish_sync(self, client_token: str, message: Any):
        try:
            self.message_queue.put_nowait((client_token, message))
        except asyncio.QueueFull:
            self._dropped_messages += 1
            if self._dropped_messages % 100 == 1:
                logger.warning(
                    "MessageBus queue full (sync), dropped %d message(s) total",
                    self._dropped_messages,
                )
        except Exception:
            try:
                asyncio.create_task(self.publish(client_token, message))
            except Exception:
                logger.error("Failed to schedule async publish task")

    def publish_from_thread(self, client_token: str, message: Any) -> None:
        """Thread-safe publish for callers outside the event loop thread."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            # Called on the loop thread — a plain coroutine call is fine.
            asyncio.ensure_future(self.publish(client_token, message))
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            logger.warning(
                "MessageBus: no event loop for thread publish; dropping message"
            )
            return
        asyncio.run_coroutine_threadsafe(self.publish(client_token, message), loop)

    async def start_processing(self):
        self.running = True
        self._loop = asyncio.get_running_loop()
        logger.info("MessageBus processing loop started")
        while self.running:
            try:
                client_token, message = await asyncio.wait_for(
                    self.message_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("MessageBus queue get error: %s", e)
                continue

            try:
                if client_token in self.subscribed_clients:
                    for callback in self.subscribed_clients[client_token]:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(message)
                            else:
                                callback(message)
                        except Exception as cb_err:
                            logger.error("MessageBus callback error: %s", cb_err)

                websocket_message = {
                    "token": client_token,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                }

                dead_sockets = set()
                for websocket in self.websocket_clients:
                    try:
                        await websocket.send_json(websocket_message)
                    except Exception:
                        dead_sockets.add(websocket)

                for dead_socket in dead_sockets:
                    self.remove_websocket(dead_socket)

            except Exception as e:
                logger.error("MessageBus processing error: %s", e)

        logger.info("MessageBus processing loop stopped")

    def stop_processing(self):
        self.running = False


class BasePlugin(ABC):
    def __init__(self):
        self.message_bus = MessageBus()

    async def send_message(self, topic: str, message: Any):
        await self.message_bus.publish(topic, message)

    @abstractmethod
    async def initialize(self):
        pass
