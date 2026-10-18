"""Ordinary HTTP/HLS video and a generated live test feed. No DRM integration."""
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import urlsplit
import uuid

import imageio_ffmpeg

from analysis_engine import MediaJob, ScanConfig, CHUNK
from core import make_demo

RETAIN = 24  # At most ~144 seconds, plus one file currently being encoded.


class LiveCapture:
    def __init__(self, engine, folder: Path, url: str, config: ScanConfig):
        self.engine, self.folder = engine, folder
        folder.mkdir(parents=True, exist_ok=True)
        source = urlsplit(url)
        if url != "demo" and (source.scheme not in ("http", "https") or not source.hostname or source.username or source.password):
            raise ValueError("Use an ordinary HTTP or HTTPS video/HLS URL without embedded credentials")
        title = "Generated live buffer test · no scene accuracy claim" if url == "demo" else f"Live · {source.hostname}"
        self.job = MediaJob(uuid.uuid4().hex, title, folder / "source.mp4", 0, True,
                            config=config, active=True, live=True, capture_running=True)
        self.stopped = threading.Event()
        self.process = None
        args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        if url == "demo":
            make_demo(folder / "demo.mp4")
            args += ["-re", "-stream_loop", "-1", "-i", str(folder / "demo.mp4")]
        else:
            args += ["-protocol_whitelist", "http,https,tcp,tls,crypto", "-rw_timeout", "15000000", "-re", "-i", url]
        args += ["-map", "0:v:0", "-map", "0:a:0?", "-sn", "-dn",
                 "-vf", "scale=w=1280:h=720:force_original_aspect_ratio=decrease:force_divisible_by=2",
                 "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-crf", "23",
                 "-profile:v", "main", "-level:v", "3.1", "-pix_fmt", "yuv420p", "-r", "30",
                 "-g", "180", "-keyint_min", "180", "-sc_threshold", "0", "-bf", "0",
                 "-maxrate", "4000k", "-bufsize", "8000k", "-c:a", "aac", "-b:a", "128k", "-ar", "48000",
                 "-f", "hls", "-hls_time", "6", "-hls_list_size", str(RETAIN),
                 "-hls_segment_type", "fmp4", "-hls_fmp4_init_filename", "init.mp4",
                 "-hls_flags", "temp_file+independent_segments", "-hls_segment_filename", str(folder / "part-%08d.m4s"),
                 str(folder / "capture.m3u8")]
        self.args = args
        with engine.condition:
            # One input capture at a time; callers stop the previous stream first.
            engine.jobs[self.job.id] = self.job
        self.thread = threading.Thread(target=self.run, name="live-capture", daemon=True)
        self.thread.start()

    def run(self):
        log_path = self.folder / "capture.log"
        try:
            with log_path.open("wb") as log:
                self.process = subprocess.Popen(self.args, cwd=self.folder, stdout=subprocess.DEVNULL, stderr=log,
                                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                while not self.stopped.wait(.3):
                    if self.job.error:
                        break
                    playlist = self.folder / "capture.m3u8"
                    if playlist.exists():
                        text = playlist.read_text(encoding="utf-8")
                        entries = re.findall(r"#EXTINF:([\d.]+),\s*\n(part-(\d+)\.m4s)", text)
                        with self.engine.condition:
                            changed = False
                            for duration, filename, index in entries:
                                index = int(index)
                                path = self.folder / filename
                                if index not in self.job.segments and path.exists():
                                    self.job.segments[index] = path
                                    self.job.duration = max(self.job.duration, index * CHUNK + float(duration))
                                    changed = True
                            if self.job.segments:
                                newest = max(self.job.segments)
                                oldest = max(0, newest - RETAIN + 1)
                                for index in list(self.job.segments):
                                    if index < oldest:
                                        path = self.job.segments.pop(index)
                                        path.unlink(missing_ok=True)
                                        self.job.chunks.pop(index, None)
                                self.job.retained_start = oldest * CHUNK
                                if self.job.position + .2 < self.job.retained_start:
                                    raise RuntimeError("The live buffer filled while playback was behind. Restart the stream to continue; no unreviewed content was skipped forward.")
                            if changed:
                                self.job.updated += 1
                                self.engine.condition.notify_all()
                    if self.process.poll() is not None:
                        if self.process.returncode and not self.stopped.is_set():
                            raise RuntimeError("The video stream could not be read or disconnected. Check its URL and try again.")
                        break
                    if time.monotonic() - self.job.last_seen > 30:
                        self.stopped.set()
        except Exception as error:
            with self.engine.condition:
                self.job.error = str(error)[:700]
                self.job.status = "error"
                self.job.active = False
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            with self.engine.condition:
                self.job.capture_running = False
                self.job.updated += 1
                self.engine.condition.notify_all()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=6)

    def manifest(self):
        with self.engine.condition:
            return {"segments": [{"index": i, "start": i * CHUNK, "end": min(self.job.duration, (i + 1) * CHUNK)}
                                  for i in sorted(self.job.segments) if i in self.job.chunks],
                    "running": self.job.capture_running, "retained_start": self.job.retained_start,
                    "mime": 'video/mp4; codecs="avc1.4d401f, mp4a.40.2"'}
