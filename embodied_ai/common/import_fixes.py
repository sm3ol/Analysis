"""Import helpers for avoiding local package shadowing."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def preload_external_av() -> None:
    """Prefer the real PyAV package over the local Analysis/av package."""
    analysis_root = Path(__file__).resolve().parents[2]

    existing = sys.modules.get("av")
    if existing is not None:
        current_file = getattr(existing, "__file__", "")
        if current_file:
            try:
                if not _is_within(Path(current_file).resolve(), analysis_root):
                    return
            except OSError:
                return
        sys.modules.pop("av", None)

    original_sys_path = list(sys.path)
    filtered_sys_path: list[str] = []
    for entry in original_sys_path:
        try:
            resolved = Path(entry or ".").resolve()
        except OSError:
            filtered_sys_path.append(entry)
            continue
        if resolved == analysis_root:
            continue
        filtered_sys_path.append(entry)

    sys.path[:] = filtered_sys_path
    try:
        try:
            import av as pyav  # type: ignore
        except ModuleNotFoundError:
            pyav = _build_pyav_stub()
    finally:
        sys.path[:] = original_sys_path

    sys.modules["av"] = pyav


def _build_pyav_stub() -> types.ModuleType:
    """Create a minimal PyAV shim for torchvision imports.

    The embodied analysis flows only need torchvision image transforms. They do
    not use torchvision video IO. A small stub is enough to satisfy the import
    path when PyAV is not installed.
    """

    stub = types.ModuleType("av")

    class _AVError(Exception):
        pass

    class _FFmpegError(_AVError):
        pass

    class _Logging:
        ERROR = 0

        @staticmethod
        def set_level(_level: int) -> None:
            return None

    class _VideoFrame:
        pict_type = None

        @staticmethod
        def from_ndarray(*_args, **_kwargs):
            raise _FFmpegError("PyAV stub does not implement video decoding or encoding.")

    class _AudioFrame:
        @staticmethod
        def from_ndarray(*_args, **_kwargs):
            raise _FFmpegError("PyAV stub does not implement audio decoding or encoding.")

    def _unsupported_open(*_args, **_kwargs):
        raise _FFmpegError("PyAV stub does not implement container IO.")

    stub.logging = _Logging()
    stub.video = types.SimpleNamespace(frame=types.SimpleNamespace(VideoFrame=_VideoFrame))
    stub.VideoFrame = _VideoFrame
    stub.AudioFrame = _AudioFrame
    stub.AVError = _AVError
    stub.FFmpegError = _FFmpegError
    stub.open = _unsupported_open
    return stub
