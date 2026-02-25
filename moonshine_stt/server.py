#!/usr/bin/env python3
"""
Moonshine Voice STT Server — AssemblyAI-compatible WebSocket API.

A standalone WebSocket server that speaks the AssemblyAI Streaming v3 protocol
but runs Moonshine Voice locally for inference. Any client that works with
AssemblyAI's real-time API can point at this server instead.

Usage:
    python -m moonshine_stt.server                       # defaults: 0.0.0.0:8765, English
    python -m moonshine_stt.server --port 9000           # custom port
    python -m moonshine_stt.server --language en          # language selection
    python -m moonshine_stt.server --model-path /path/to/model --model-arch 2

Protocol (mirrors AssemblyAI Streaming v3):
    Client → Server:
        • Raw PCM16-LE audio bytes (binary frames)
        • {"type": "ForceEndpoint"}          — force end-of-turn
        • {"type": "Terminate"}              — end session

    Server → Client:
        • {"type": "Begin", "id": "...", "expires_at": ...}
        • {"type": "Turn", "turn_order": N, "transcript": "...", "end_of_turn": bool, ...}
        • {"type": "Termination", "audio_duration_seconds": ..., "session_duration_seconds": ...}
"""

import argparse
import asyncio
import json
import struct
import time
import uuid
from typing import Optional

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve
except ModuleNotFoundError:
    raise SystemExit(
        "websockets is required: pip install websockets"
    )

from moonshine_voice import (
    Transcriber,
    TranscriptEventListener,
    LineStarted,
    LineTextChanged,
    LineCompleted,
    ModelArch,
    get_model_for_language,
)


# ---------------------------------------------------------------------------
# Transcript event → AssemblyAI protocol bridge
# ---------------------------------------------------------------------------

class _SessionListener(TranscriptEventListener):
    """Bridges Moonshine events into an async queue of AssemblyAI-shaped messages."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self._queue = queue
        self._loop = loop
        self._turn_order = 0

    def _put(self, msg: dict):
        """Thread-safe enqueue (Moonshine callbacks run on the executor thread)."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, msg)

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        self._turn_order += 1
        self._put({
            "type": "Turn",
            "turn_order": self._turn_order,
            "turn_is_formatted": False,
            "end_of_turn": False,
            "end_of_turn_confidence": 0.0,
            "transcript": event.line.text,
            "words": [],
        })

    def on_line_completed(self, event: LineCompleted) -> None:
        self._turn_order += 1
        self._put({
            "type": "Turn",
            "turn_order": self._turn_order,
            "turn_is_formatted": True,
            "end_of_turn": True,
            "end_of_turn_confidence": 1.0,
            "transcript": event.line.text,
            "words": [],
        })


# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------

async def _handle_connection(
    ws,
    transcriber: Transcriber,
    sample_rate: int,
    update_interval: float,
):
    """Handle a single WebSocket client session."""

    session_id = str(uuid.uuid4())
    session_start = time.time()
    audio_samples_received = 0
    loop = asyncio.get_event_loop()

    # Per-session queue for outbound messages
    out_queue: asyncio.Queue = asyncio.Queue()

    # Create a dedicated stream for this connection
    stream = transcriber.create_stream(update_interval=update_interval)
    listener = _SessionListener(out_queue, loop)
    stream.add_listener(listener)
    stream.start()

    print(f"[{session_id[:8]}] Client connected")

    # Send Begin message
    begin_msg = {
        "type": "Begin",
        "id": session_id,
        "expires_at": int(session_start + 86400),  # 24h from now
    }
    await ws.send(json.dumps(begin_msg))

    # --- Writer task: drain out_queue → client
    async def _writer():
        try:
            while True:
                msg = await out_queue.get()
                if msg is None:
                    break
                await ws.send(json.dumps(msg))
        except websockets.ConnectionClosed:
            pass

    writer_task = asyncio.create_task(_writer())

    # --- Reader: receive audio or control messages from client
    try:
        async for message in ws:
            if isinstance(message, bytes):
                # PCM16-LE audio
                n_samples = len(message) // 2
                if n_samples == 0:
                    continue
                samples = struct.unpack(f"<{n_samples}h", message[:n_samples * 2])
                float_samples = [s / 32768.0 for s in samples]
                audio_samples_received += n_samples

                # Run blocking add_audio in executor
                await loop.run_in_executor(
                    None, stream.add_audio, float_samples, sample_rate
                )

            elif isinstance(message, str):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type", "")

                if msg_type == "ForceEndpoint":
                    # Force a transcription update to finalize current speech
                    await loop.run_in_executor(
                        None,
                        stream.update_transcription,
                        Transcriber.MOONSHINE_FLAG_FORCE_UPDATE,
                    )

                elif msg_type == "Terminate":
                    break

    except websockets.ConnectionClosed:
        pass

    # --- Cleanup: stop stream, send termination, close
    try:
        await loop.run_in_executor(None, stream.stop)
    except Exception as e:
        print(f"[{session_id[:8]}] Stream stop error: {e}")

    # Drain any remaining events
    await asyncio.sleep(0.1)

    session_end = time.time()
    audio_duration = audio_samples_received / sample_rate

    termination_msg = {
        "type": "Termination",
        "audio_duration_seconds": round(audio_duration, 2),
        "session_duration_seconds": round(session_end - session_start, 2),
    }
    try:
        await ws.send(json.dumps(termination_msg))
    except websockets.ConnectionClosed:
        pass

    # Signal writer to stop
    out_queue.put_nowait(None)
    await writer_task

    try:
        stream.close()
    except Exception:
        pass

    print(
        f"[{session_id[:8]}] Disconnected — "
        f"audio={audio_duration:.1f}s, session={session_end - session_start:.1f}s"
    )


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def run_server(
    host: str,
    port: int,
    model_path: Optional[str],
    model_arch: Optional[int],
    language: str,
    sample_rate: int,
    update_interval: float,
):
    """Start the WebSocket server."""

    # Resolve model
    if model_path is None:
        model_path, resolved_arch = get_model_for_language(
            wanted_language=language,
            wanted_model_arch=model_arch,
        )
        if model_arch is None:
            model_arch = resolved_arch
    else:
        if model_arch is None:
            raise ValueError("--model-arch is required when --model-path is specified")
        model_arch = ModelArch(model_arch)

    print(f"Loading Moonshine model: path={model_path}, arch={model_arch}")
    transcriber = Transcriber(
        model_path=model_path,
        model_arch=model_arch,
        update_interval=update_interval,
    )
    print(f"Model loaded successfully")

    async def handler(ws):
        await _handle_connection(ws, transcriber, sample_rate, update_interval)

    print(f"Moonshine STT server listening on ws://{host}:{port}")
    print(f"  sample_rate={sample_rate}, update_interval={update_interval}s")
    print(f"  Protocol: AssemblyAI Streaming v3 compatible")
    print()

    async with ws_serve(handler, host, port):
        await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(
        description="Moonshine Voice STT Server — AssemblyAI-compatible WebSocket API"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--language", type=str, default="en",
        help="Language for auto model selection (default: en)",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Explicit path to model directory (overrides --language)",
    )
    parser.add_argument(
        "--model-arch", type=int, default=None,
        help="Model architecture int (0=tiny, 1=base, 2=tiny-streaming, 3=base-streaming, 4=small-streaming, 5=medium-streaming)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Expected audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--update-interval", type=float, default=0.5,
        help="Seconds between transcription updates (default: 0.5)",
    )
    args = parser.parse_args()

    asyncio.run(run_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        model_arch=args.model_arch,
        language=args.language,
        sample_rate=args.sample_rate,
        update_interval=args.update_interval,
    ))


if __name__ == "__main__":
    main()
