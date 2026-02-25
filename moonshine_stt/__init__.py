#
# Moonshine Voice STT service for pipecat.
#
# Drop-in replacement for AssemblyAI's real-time STT service,
# using Moonshine Voice for on-device speech-to-text.
#

from .stt import MoonshineSTTService

__all__ = ["MoonshineSTTService"]
