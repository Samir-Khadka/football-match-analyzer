"""Video source resolution (v1.0).

Accepts:
- local file path → used directly
- http(s) URL to a video file → downloaded
- YouTube / social video URL → best-quality ≤720p via yt-dlp
- live stream URL (YouTube live, HLS, RTSP) → stream URL for live mode

Returns (kind, location) where kind is "file" or "stream".
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def is_likely_stream(url: str) -> bool:
    """Heuristic: live streams and playlist/manifest URLs."""
    low = url.lower()
    return any(k in low for k in (".m3u8", "rtsp://", "/live", "livestream"))


def resolve(url: str, out_dir: Path) -> tuple[str, str]:
    """Resolve a URL to (kind, location).

    kind="file": location is a local path (downloaded).
    kind="stream": location is a stream URL for live processing.
    Raises RuntimeError on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if is_likely_stream(url):
        # try to get a direct stream URL via yt-dlp; fall back to the URL
        # itself (OpenCV can read HLS/RTSP directly)
        stream = _yt_dlp_stream_url(url)
        return ("stream", stream or url)

    # VOD: download
    out_path = out_dir / "source.mp4"
    _yt_dlp_download(url, out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError("download produced no file")
    return ("file", str(out_path))


def _yt_dlp_download(url: str, out_path: Path):
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError("yt-dlp not installed (pip install yt-dlp)")
    # direct file URL? try a plain download first for speed
    if url.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
        try:
            import urllib.request
            urllib.request.urlretrieve(url, out_path)
            return
        except Exception:
            pass  # fall through to yt-dlp
    cmd = [ytdlp, "-f", "best[height<=720]/best", "-o", str(out_path),
           "--no-playlist", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {r.stderr[-300:]}")


def _yt_dlp_stream_url(url: str) -> str | None:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return None
    try:
        r = subprocess.run(
            [ytdlp, "-g", "-f", "best[height<=720]/best", "--no-playlist",
             url],
            capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None
