from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MODEL_REVISION = "14a68262fa5fc6d497cb107c18bc5a9308ef21ca"


def normalize_transcript(value: str) -> str:
    return " ".join(str(value).split())


def _extract_text(result: Any) -> str:
    if isinstance(result, str):
        return normalize_transcript(result)
    if isinstance(result, dict):
        for key in ("text", "transcription", "transcript"):
            if key in result:
                return _extract_text(result[key])
    if isinstance(result, (list, tuple)):
        return normalize_transcript(" ".join(_extract_text(item) for item in result))
    raise RuntimeError(f"unsupported ChunkFormer result type: {type(result).__name__}")


def _write_pcm16_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


class ChunkFormerASRProvider:
    """One-job wrapper around the official ChunkFormer Python API."""

    name = "chunkformer"

    def __init__(
        self,
        *,
        model_name: str = "khanhld/chunkformer-ctc-large-vie",
        revision: str | None = DEFAULT_MODEL_REVISION,
        device: str = "cpu",
        allow_download: bool = False,
    ) -> None:
        try:
            from chunkformer import ChunkFormerModel
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "ChunkFormer runtime is unavailable; install the 'asr' extra"
            ) from exc
        try:
            checkpoint = snapshot_download(
                repo_id=model_name,
                revision=revision,
                local_files_only=not allow_download,
            )
        except Exception as exc:
            policy = "download allowed" if allow_download else "local files only"
            raise RuntimeError(
                f"could not resolve {model_name!r} ({policy}); no fallback model is used"
            ) from exc
        self.model_name = model_name
        self.revision = revision or Path(checkpoint).name
        self.device = device
        self._model = ChunkFormerModel.from_pretrained(checkpoint).to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        if sample_rate != 16000:
            raise ValueError("ChunkFormer provider requires 16 kHz audio")
        with tempfile.TemporaryDirectory(prefix="mvr-asr-") as temp_dir:
            audio_path = Path(temp_dir) / "speech.wav"
            _write_pcm16_wav(audio_path, audio, sample_rate)
            result = self._model.endless_decode(
                audio_path=str(audio_path),
                chunk_size=64,
                left_context_size=128,
                right_context_size=128,
                total_batch_duration=14400,
                return_timestamps=False,
            )
        return _extract_text(result)
