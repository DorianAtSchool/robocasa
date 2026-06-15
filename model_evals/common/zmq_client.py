"""Msgpack ZeroMQ client compatible with RLDX-1 PolicyServer."""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import msgpack
import numpy as np

logger = logging.getLogger(__name__)


def _encode_custom_classes(obj):
    if isinstance(obj, np.ndarray):
        output = io.BytesIO()
        np.save(output, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": output.getvalue()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _decode_custom_classes(obj):
    if not isinstance(obj, dict):
        return obj
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def _packb(data: Any) -> bytes:
    return msgpack.packb(data, default=_encode_custom_classes)


def _unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, object_hook=_decode_custom_classes)


class ZmqPolicyClient:
    """Persistent REQ client for RLDX-1's ZeroMQ policy server."""

    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._context = None
        self._socket = None

    def _connect(self):
        import zmq

        if self._context is None:
            self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.connect(f"tcp://{self.host}:{self.port}")
        logger.info("Connected to tcp://%s:%s", self.host, self.port)

    def _call_endpoint(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        import zmq

        if self._socket is None:
            self._connect()
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        try:
            self._socket.send(_packb(request))
            response = _unpackb(self._socket.recv())
        except zmq.error.Again as exc:
            self._reset_socket()
            raise TimeoutError(f"Timed out waiting for RLDX server at {self.host}:{self.port}") from exc
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def infer(self, request: dict) -> Any:
        return self._call_endpoint("get_action", request)

    def reset(self):
        try:
            return self._call_endpoint("reset", {"options": None})
        except Exception:
            logger.debug("RLDX reset endpoint failed", exc_info=True)
            return None

    def _reset_socket(self):
        if self._socket is not None:
            try:
                self._socket.close(linger=0)
            except Exception:
                pass
            self._socket = None

    def close(self):
        self._reset_socket()
        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass
            self._context = None
