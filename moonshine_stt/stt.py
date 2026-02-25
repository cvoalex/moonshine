#
# Copyright (c) 2025
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moonshine Voice speech-to-text service implementation for pipecat.

This module provides a local, on-device STT service using Moonshine Voice
as a drop-in replacement for cloud-based STT services like AssemblyAI.
"""

import asyncio
import struct
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    from moonshine_voice import (
        Transcriber,
        TranscriptEventListener,
        LineStarted,
        LineTextChanged,
        LineCompleted,
        ModelArch,
        get_model_for_language,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Moonshine STT, you need to `pip install moonshine-voice`.'
    )
    raise Exception(f"Missing module: {e}")


class _FrameCollector(TranscriptEventListener):
    """Collects transcript events and converts them to pipecat frames.

    This listener is called synchronously during Moonshine's add_audio()
    and update_transcription() calls. Frames are collected into a list
    and later yielded by the async run_stt() generator.
    """

    def __init__(self, user_id: str, language: Language):
        self._user_id = user_id
        self._language = language
        self.frames: list[Frame] = []

    def on_line_text_changed(self, event: LineTextChanged) -> None:
        """Emit interim transcription when line text changes."""
        if not event.line.text:
            return
        self.frames.append(
            InterimTranscriptionFrame(
                text=event.line.text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._language,
            )
        )

    def on_line_completed(self, event: LineCompleted) -> None:
        """Emit final transcription when a line is completed."""
        if not event.line.text:
            return
        self.frames.append(
            TranscriptionFrame(
                text=event.line.text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._language,
            )
        )

    def drain(self) -> list[Frame]:
        """Return collected frames and clear the internal buffer."""
        frames = self.frames
        self.frames = []
        return frames


class MoonshineSTTService(STTService):
    """Moonshine Voice local speech-to-text service for pipecat.

    Provides real-time on-device speech transcription using Moonshine Voice.
    This is a drop-in replacement for cloud STT services like AssemblyAI -
    audio is processed locally with no network calls.

    Supports both interim and final transcriptions. The Moonshine streaming
    engine handles VAD, line segmentation, and partial results internally.

    Example usage::

        from moonshine_stt import MoonshineSTTService

        stt = MoonshineSTTService(language="en")

        # Or with explicit model path:
        stt = MoonshineSTTService(
            model_path="/path/to/model",
            model_arch=ModelArch.BASE_STREAMING,
        )
    """

    def __init__(
        self,
        *,
        language: str = "en",
        model_path: Optional[str] = None,
        model_arch: Optional[ModelArch] = None,
        update_interval: float = 0.5,
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize the Moonshine STT service.

        Args:
            language: Language code for model selection (e.g. "en").
                Used to auto-download the appropriate model if model_path
                is not provided. Defaults to "en".
            model_path: Explicit path to the model directory. If None,
                the model is resolved automatically from the language.
            model_arch: Model architecture (e.g. ModelArch.BASE_STREAMING).
                Required if model_path is set. If None and model_path is None,
                resolved automatically from the language.
            update_interval: Seconds between transcription updates during
                streaming. Lower values give faster interim results but
                use more CPU. Defaults to 0.5.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            **kwargs: Additional arguments passed to pipecat STTService
                (e.g. audio_passthrough, stt_ttfb_timeout).
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._language_code = language
        self._language = Language.EN  # pipecat Language enum for frames
        self._model_path = model_path
        self._model_arch = model_arch
        self._update_interval = update_interval

        # Runtime state - initialized in start()
        self._transcriber: Optional[Transcriber] = None
        self._stream = None
        self._collector: Optional[_FrameCollector] = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        """Start the Moonshine STT service.

        Loads the model and creates a transcription stream. Model loading
        runs in a thread executor to avoid blocking the event loop.

        Args:
            frame: The pipecat StartFrame.
        """
        await super().start(frame)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._init_transcriber)
        logger.info(f"Moonshine STT started (model_arch={self._model_arch})")

    def _init_transcriber(self):
        """Initialize the Moonshine transcriber and stream (runs in executor)."""
        # Resolve model if not explicitly provided
        if self._model_path is None:
            self._model_path, self._model_arch = get_model_for_language(
                wanted_language=self._language_code
            )
            logger.debug(
                f"Resolved model: path={self._model_path}, arch={self._model_arch}"
            )

        self._transcriber = Transcriber(
            model_path=self._model_path,
            model_arch=self._model_arch,
            update_interval=self._update_interval,
        )

        self._collector = _FrameCollector(
            user_id=self._user_id,
            language=self._language,
        )

        self._stream = self._transcriber.create_stream(
            update_interval=self._update_interval
        )
        self._stream.add_listener(self._collector)
        self._stream.start()

    async def stop(self, frame: EndFrame):
        """Stop the Moonshine STT service.

        Stops the stream and frees model resources.

        Args:
            frame: The pipecat EndFrame.
        """
        await super().stop(frame)
        await self._cleanup()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Moonshine STT service.

        Args:
            frame: The pipecat CancelFrame.
        """
        await super().cancel(frame)
        await self._cleanup()

    async def _cleanup(self):
        """Release Moonshine resources in a thread executor."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._cleanup_sync)

    def _cleanup_sync(self):
        """Synchronous cleanup of Moonshine resources."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Error stopping Moonshine stream: {e}")
            self._stream = None

        if self._transcriber is not None:
            try:
                self._transcriber.close()
            except Exception as e:
                logger.warning(f"Error closing Moonshine transcriber: {e}")
            self._transcriber = None

        self._collector = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data through Moonshine for speech-to-text.

        Converts PCM16 audio bytes to float samples, feeds them to the
        Moonshine streaming engine, and yields any resulting transcription
        frames (interim or final).

        The actual inference runs in a thread executor so the asyncio
        event loop is not blocked.

        Args:
            audio: Raw 16-bit signed PCM audio bytes (mono, at sample_rate Hz).

        Yields:
            TranscriptionFrame for finalized lines.
            InterimTranscriptionFrame for partial/in-progress lines.
        """
        if self._stream is None or self._collector is None:
            yield None
            return

        # Update user_id in collector (may change between calls)
        self._collector._user_id = self._user_id

        # Convert 16-bit signed PCM to float32 [-1.0, 1.0]
        samples = _pcm16_bytes_to_float_list(audio)

        # Run the blocking Moonshine add_audio in a thread.
        # Events fire synchronously during this call and are
        # collected by _FrameCollector.
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._stream.add_audio, samples, self.sample_rate
        )

        # Yield all frames collected during add_audio
        for frame in self._collector.drain():
            yield frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, handling VAD events for turn management.

        On VADUserStoppedSpeakingFrame, forces a transcription update to
        flush any pending partial results, similar to AssemblyAI's
        ForceEndpoint behavior.

        Args:
            frame: The frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            pass  # Moonshine handles speech start internally
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Force a transcription update to finalize pending text quickly
            if self._stream is not None and self._collector is not None:
                self._collector._user_id = self._user_id
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None,
                    self._stream.update_transcription,
                    Transcriber.MOONSHINE_FLAG_FORCE_UPDATE,
                )
                for frame in self._collector.drain():
                    await self.push_frame(frame)
            await self.start_processing_metrics()


def _pcm16_bytes_to_float_list(audio_bytes: bytes) -> list[float]:
    """Convert 16-bit signed little-endian PCM bytes to a list of floats in [-1.0, 1.0].

    Args:
        audio_bytes: Raw PCM16 audio data.

    Returns:
        List of float samples normalized to [-1.0, 1.0].
    """
    n_samples = len(audio_bytes) // 2
    if n_samples == 0:
        return []
    samples = struct.unpack(f"<{n_samples}h", audio_bytes[: n_samples * 2])
    return [s / 32768.0 for s in samples]
