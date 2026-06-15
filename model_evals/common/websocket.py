"""Msgpack websocket client shared by model evaluation backends."""

from __future__ import annotations

import functools
import logging
import time

import msgpack
import numpy as np

logger = logging.getLogger(__name__)


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class WebsocketClient:
    """Persistent websocket client with auto-reconnect."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._ws = None
        self._metadata = None

    def _connect(self):
        import websockets.sync.client as ws_client

        url = f"ws://{self.host}:{self.port}"
        logger.info("Connecting to %s...", url)
        while True:
            try:
                self._ws = ws_client.connect(
                    url,
                    max_size=None,
                    compression=None,
                    ping_interval=120,
                    ping_timeout=600,
                )
                self._metadata = _unpackb(self._ws.recv())
                logger.info("Connected. Server metadata: %s", self._metadata)
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(2)

    def infer(self, obs: dict) -> dict:
        if self._ws is None:
            self._connect()
        try:
            self._ws.send(_packb(obs))
            raw = self._ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Server error:\n{raw}")
            return _unpackb(raw)
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            self._ws = None
            self._connect()
            self._ws.send(_packb(obs))
            raw = self._ws.recv()
            if isinstance(raw, str):
                raise RuntimeError(f"Server error:\n{raw}")
            return _unpackb(raw)

    def close(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

