"""Torch-serialized ZeroMQ client compatible with Isaac-GR00T service.py."""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _torch_save(data: Any) -> bytes:
    import torch

    buffer = io.BytesIO()
    torch.save(data, buffer)
    return buffer.getvalue()


def _torch_load(data: bytes) -> Any:
    import torch

    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)


class TorchZmqPolicyClient:
    """Persistent REQ client for GR00T's RobotInferenceServer."""

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

    def _call_endpoint(self, endpoint: str, data: dict | None = None):
        import zmq

        if self._socket is None:
            self._connect()
        request = {"endpoint": endpoint, "data": data or {}}
        try:
            self._socket.send(_torch_save(request))
            response = _torch_load(self._socket.recv())
        except zmq.error.Again as exc:
            self._reset_socket()
            raise TimeoutError(f"Timed out waiting for GR00T server at {self.host}:{self.port}") from exc
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def infer(self, request: dict) -> Any:
        return self._call_endpoint("get_action", request)

    def get_modality_config(self) -> Any:
        return self._call_endpoint("get_modality_config")

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
