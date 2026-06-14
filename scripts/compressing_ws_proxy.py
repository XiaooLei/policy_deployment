#!/usr/bin/env python3
"""WebSocket policy proxy that compresses raw image observations before forwarding.

Use when an upstream evaluator sends the original policy_deployment protocol with
raw CHW uint8 numpy arrays, but the downstream OpenPI server accepts JPEG/PNG
bytes in obs['images'] (metadata key: accepts_compressed_images=true).

Flow:
  evaluator/client -> this proxy: raw msgpack-numpy observation (~2.77 MB)
  this proxy -> backend: same observation, but images compressed to JPEG bytes
  backend -> this proxy -> evaluator/client: action response unchanged
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import time
from typing import Any

import numpy as np
from PIL import Image
import websockets
from websockets.asyncio import client as ws_client
from websockets.asyncio import server as ws_server

from server import msgpack_numpy

LOG = logging.getLogger("compressing_ws_proxy")


def _array_to_jpeg_bytes(value: np.ndarray, *, quality: int) -> bytes:
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"image ndarray must be 3-D CHW/HWC, got shape={arr.shape}")

    # policy_deployment uses CHW; tolerate HWC for robustness.
    if arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)

    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def compress_observation(raw_frame: bytes, *, quality: int) -> tuple[bytes, dict[str, Any]]:
    obs = msgpack_numpy.unpackb(raw_frame)
    if not isinstance(obs, dict):
        return raw_frame, {"compressed": False, "reason": "obs_not_dict"}

    images = obs.get("images")
    if not isinstance(images, dict):
        return raw_frame, {"compressed": False, "reason": "images_not_dict"}

    out_images: dict[str, Any] = {}
    raw_image_bytes = 0
    jpeg_bytes = 0
    compressed_count = 0
    shapes: dict[str, tuple[int, ...]] = {}

    for key, value in images.items():
        if isinstance(value, np.ndarray):
            raw_image_bytes += int(value.nbytes)
            shapes[str(key)] = tuple(value.shape)
            encoded = _array_to_jpeg_bytes(value, quality=quality)
            jpeg_bytes += len(encoded)
            out_images[key] = encoded
            compressed_count += 1
        else:
            # Already compressed bytes or some other metadata; pass through.
            out_images[key] = value
            if isinstance(value, (bytes, bytearray, memoryview)):
                jpeg_bytes += len(value)

    if compressed_count == 0:
        return raw_frame, {"compressed": False, "reason": "no_ndarray_images"}

    new_obs = dict(obs)
    new_obs["images"] = out_images
    packed = msgpack_numpy.packb(new_obs)
    return packed, {
        "compressed": True,
        "images": compressed_count,
        "raw_image_bytes": raw_image_bytes,
        "jpeg_bytes": jpeg_bytes,
        "in_frame_bytes": len(raw_frame),
        "out_frame_bytes": len(packed),
        "shapes": shapes,
    }


async def handle_connection(
    downstream: ws_server.ServerConnection,
    *,
    backend_uri: str,
    quality: int,
    backend_open_timeout: float,
) -> None:
    peer = downstream.remote_address
    LOG.info("client connected peer=%s backend=%s", peer, backend_uri)
    async with ws_client.connect(
        backend_uri,
        compression=None,
        max_size=None,
        open_timeout=backend_open_timeout,
        ping_interval=None,
        close_timeout=10,
    ) as backend:
        metadata = await backend.recv()
        await downstream.send(metadata)
        LOG.info("metadata forwarded peer=%s bytes=%s", peer, len(metadata) if hasattr(metadata, "__len__") else None)

        while True:
            try:
                raw = await downstream.recv()
            except websockets.ConnectionClosed:
                LOG.info("client closed peer=%s", peer)
                break

            if isinstance(raw, str):
                # Policy observations should be binary; forward text unchanged.
                await backend.send(raw)
            else:
                t0 = time.monotonic()
                forwarded, stats = compress_observation(raw, quality=quality)
                compress_ms = (time.monotonic() - t0) * 1000
                LOG.info(
                    "obs peer=%s compressed=%s in=%s out=%s raw_images=%s jpeg=%s compress_ms=%.1f detail=%s",
                    peer,
                    stats.get("compressed"),
                    stats.get("in_frame_bytes", len(raw)),
                    stats.get("out_frame_bytes", len(forwarded)),
                    stats.get("raw_image_bytes"),
                    stats.get("jpeg_bytes"),
                    compress_ms,
                    stats,
                )
                await backend.send(forwarded)

            response = await backend.recv()
            await downstream.send(response)
            LOG.info("response forwarded peer=%s bytes=%s", peer, len(response) if hasattr(response, "__len__") else None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8001)
    parser.add_argument("--backend", default="ws://120.55.13.171:8001")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--backend-open-timeout", type=float, default=15.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    async def _main() -> None:
        async with ws_server.serve(
            lambda ws: handle_connection(
                ws,
                backend_uri=args.backend,
                quality=args.jpeg_quality,
                backend_open_timeout=args.backend_open_timeout,
            ),
            args.listen_host,
            args.listen_port,
            compression=None,
            max_size=None,
            ping_interval=None,
            close_timeout=10,
        ):
            LOG.info("listening on %s:%s -> %s jpeg_quality=%s", args.listen_host, args.listen_port, args.backend, args.jpeg_quality)
            await asyncio.Future()

    asyncio.run(_main())


if __name__ == "__main__":
    main()
