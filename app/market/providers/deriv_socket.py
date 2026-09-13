"""One Deriv connection, many requests.

Deriv has no REST surface, so every history request is a WebSocket call. Those
used to open a connection each: measured from the production box, a single page
cost 1.48s of which almost all was the handshake, and five days of 1m gold bars
is eight pages. That alone put the request over the API's upstream timeout.

Deriv correlates a response to its request with ``req_id``, so one connection
serves any number of concurrent callers. The connection is lazy, per event
loop, and replaced when it drops -- nothing here retries the vendor's own
answer, only the transport.

A per-loop instance rather than one global: the scheduled jobs each run their
own ``asyncio.run()``, and a socket belongs to the loop it was opened on.
Same reason app/market/providers/registry.py keys its locks by loop.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import weakref
from typing import Any

import websockets

logger = logging.getLogger(__name__)

RESPONSE_TIMEOUT_SECONDS = 15.0
CONNECT_TIMEOUT_SECONDS = 10.0


class DerivSocket:
    """A single connection, shared by every caller on one event loop."""

    def __init__(self, url: str):
        self._url = url
        self._ws: Any = None
        self._reader: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._ids = itertools.count(1)
        self._connecting = asyncio.Lock()

    async def request(self, payload: dict, timeout: float = RESPONSE_TIMEOUT_SECONDS) -> dict:
        """Send `payload`, wait for the response Deriv tags with the same id.

        Raises on a transport failure -- the provider above turns that into its
        own "vendor failed" answer, which is a different thing from an empty
        range and must stay that way.
        """
        req_id = next(self._ids)
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future = loop.create_future()
        self._pending[req_id] = waiter
        try:
            ws = await self._ensure()
            await ws.send(json.dumps({**payload, "req_id": req_id}))
            return await asyncio.wait_for(waiter, timeout=timeout)
        finally:
            self._pending.pop(req_id, None)

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            self._reader = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _ensure(self):
        if self._ws is not None and not getattr(self._ws, "closed", False):
            return self._ws
        async with self._connecting:
            # Another caller may have connected while this one queued.
            if self._ws is not None and not getattr(self._ws, "closed", False):
                return self._ws
            self._ws = await websockets.connect(self._url, open_timeout=CONNECT_TIMEOUT_SECONDS)
            self._reader = asyncio.create_task(self._read_forever(self._ws))
            return self._ws

    async def _read_forever(self, ws) -> None:
        try:
            async for raw in ws:
                message = json.loads(raw)
                waiter = self._pending.get(message.get("req_id"))
                if waiter is not None and not waiter.done():
                    waiter.set_result(message)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # The connection died. Everyone waiting on it has to hear about it
            # rather than sit until their own timeout.
            logger.warning("Deriv socket dropped: %s", e)
            self._fail_pending(e)
        finally:
            if self._ws is ws:
                self._ws = None

    def _fail_pending(self, error: BaseException) -> None:
        for waiter in list(self._pending.values()):
            if not waiter.done():
                waiter.set_exception(error)


_by_loop: weakref.WeakKeyDictionary[Any, DerivSocket] = weakref.WeakKeyDictionary()


def socket_for(url: str) -> DerivSocket:
    """The connection for the running loop, opened on first use."""
    loop = asyncio.get_running_loop()
    existing = _by_loop.get(loop)
    if existing is None:
        existing = DerivSocket(url)
        _by_loop[loop] = existing
    return existing
