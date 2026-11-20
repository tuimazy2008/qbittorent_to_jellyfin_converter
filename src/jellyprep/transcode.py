"""Video re-encoding and audio processing with ffmpeg.

Hardware path (VAAPI, Intel/AMD): decode + scale + encode on the GPU. HDR is tone-mapped on the CPU
(zscale + hable) after the GPU has already scaled the picture down: GPU tone mapping on older Intel
iGPUs (tonemap_vaapi) crushes shadows, and libplacebo/Vulkan is broken there.
"""
from __future__ import annotations

import functools
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from . import media

VOICE_TITLE = "Голос громче"
QUALITY = {"small": 4000, "standard": 6000, "high": 9000}   # kbit/s for 1920x1080
VOICE = {
    "light":  {"center": 1.2, "dyn": "dynaudnorm=f=250:g=15:p=0.9:m=5"},
    "medium": {"center": 1.5, "dyn": "dynaudnorm=f=150:g=11:p=0.9:m=10"},
    "strong": {"center": 2.0, "dyn": "dynaudnorm=f=100:g=7:p=0.9:m=20:s=12"},
}
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv")
HW_DECODE = {"h264", "hevc", "mpeg2video", "vc1", "vp8", "vp9", "av1"}
AUDIO_DEFAULT_BPS = {"dts": 1_509_000, "truehd": 3_000_000, "ac3": 448_000, "eac3": 640_000, "aac": 192_000,
                     "flac": 1_200_000, "mp3": 192_000, "opus": 128_000, "vorbis": 160_000,
                     "pcm_s16le": 1_536_000, "pcm_s24le": 2_304_000}
LANG2 = {"ru": "rus", "en": "eng", "uk": "ukr", "de": "ger", "fr": "fre", "es": "spa", "it": "ita",
         "ja": "jpn", "ko": "kor", "zh": "chi", "be": "bel", "kk": "kaz", "pl": "pol"}


def norm_lang(code: str | None) -> str:
    code = (code or "").strip().lower()
    return LANG2.get(code, code) if code not in ("", "und", "unk", "unknown") else ""


# --- probe summary ---------------------------------------------------------------
def _bps(stream: dict) -> int | None:
    if str(stream.get("bit_rate", "")).isdigit():
        return int(stream["bit_rate"])
    for k, v in (stream.get("tags") or {}).items():
        if k.upper() in ("BPS", "BPS-ENG") and str(v).isdigit():
            return int(v)
    return None


def summarize(probe: dict) -> dict:
    """Compact, JSON-friendly description of a file from ffprobe output."""
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    duration = float(fmt.get("duration") or 0)
    size = int(fmt.get("size") or 0)
    info: dict = {"duration": duration, "size": size, "video": None, "audio": [], "subs": []}
    for st in streams:
        tags = st.get("tags") or {}
        disp = st.get("disposition") or {}
        kind = st.get("codec_type")
        if kind == "video" and not disp.get("attached_pic") and info["video"] is None:
            info["video"] = {
                "index": st["index"], "codec": st.get("codec_name", ""),
                "width": int(st.get("width") or 0), "height": int(st.get("height") or 0),
                "pix_fmt": st.get("pix_fmt", ""), "bps": _bps(st),
                "hdr": st.get("color_transfer") in ("smpte2084", "arib-std-b67"),
                "interlaced": st.get("field_order") in ("tt", "bb", "tb", "bt"),
            }
        elif kind == "audio":
            info["audio"].append({"index": st["index"], "codec": st.get("codec_name", ""),
                                  "channels": int(st.get("channels") or 2), "lang": norm_lang(tags.get("language")),
                                  "title": tags.get("title", ""), "default": bool(disp.get("default")),
                                  "bps": _bps(st)})
        elif kind == "subtitle":
            info["subs"].append({"index": st["index"], "codec": st.get("codec_name", ""),
                                 "lang": norm_lang(tags.get("language")), "title": tags.get("title", ""),
                                 "forced": bool(disp.get("forced"))})
    v = info["video"]
    if v and not v["bps"] and duration:
        total = int(fmt.get("bit_rate") or 0) or int(size * 8 / duration)
        audio = sum(a["bps"] or AUDIO_DEFAULT_BPS.get(a["codec"], 256_000) for a in info["audio"])
        v["bps"] = max(total - audio, 0) or None
        v["bps_estimated"] = True
    return info


def probe(path: str | Path) -> dict:
    return summarize(media.ffprobe(path))


def describe(info: dict) -> str:
    v = info.get("video") or {}
    parts = []
    if v:
        parts.append(f'{v["codec"]} {v["width"]}×{v["height"]}'
                     + (f' {v["bps"] / 1e6:.1f} Мбит/с' if v.get("bps") else "")
                     + (" HDR" if v.get("hdr") else "") + (" чересстрочное" if v.get("interlaced") else ""))
    if info.get("audio"):
        parts.append("звук: " + ", ".join(f'{a["lang"] or "?"} {a["codec"]} {a["channels"]}ch' for a in info["audio"]))
    if info.get("subs"):
        parts.append("субтитры: " + ", ".join(a["lang"] or "?" for a in info["subs"]))
    return " · ".join(parts)


# --- decisions ---------------------------------------------------------------------
def video_decision(info: dict, s: dict) -> tuple[bool, list[str]]:
    v = info.get("video")
    mode = s.get("video_mode", "compat")
    if not v or mode == "off":
        return False, []
    reasons = []
    if v["width"] > 1920 or v["height"] > 1088:
        reasons.append(f'больше 1080p ({v["width"]}×{v["height"]})')
    if v["hdr"]:
        reasons.append("HDR → обычный цвет")
    if mode == "compat":
        if v["codec"] != "h264":
            reasons.append(f'кодек {v["codec"]} играет не везде')
        elif "10" in v["pix_fmt"]:
            reasons.append("10-битный H.264 играет не везде")
        if v.get("bps") and v["bps"] > float(s.get("video_max_mbps", 12)) * 1e6:
            reasons.append(f'битрейт {v["bps"] / 1e6:.1f} Мбит/с > {float(s["video_max_mbps"]):g}')
        if v["interlaced"]:
            reasons.append("чересстрочное")
    return bool(reasons), reasons


def select_tracks(info: dict, s: dict) -> dict:
    keep_langs = {norm_lang(x) for x in s.get("keep_langs", [])}
    keep_unknown = s.get("keep_unknown_lang", True)

    def wanted(t):
        if not keep_langs:          # no languages chosen = no filtering
            return True
        return t["lang"] in keep_langs or (not t["lang"] and keep_unknown)

    audio = info.get("audio", [])
    keep = [a for a in audio if wanted(a)] or list(audio)
    per_lang = int(s.get("max_tracks_per_lang") or 0)
    if per_lang > 0:
        seen: dict[str, int] = {}
        limited = []
        for a in keep:
            if seen.get(a["lang"], 0) < per_lang or a["default"]:
                limited.append(a)
                seen[a["lang"]] = seen.get(a["lang"], 0) + 1
        keep = limited
    subs_keep = [t for t in info.get("subs", []) if wanted(t)]
    voice = None
    if s.get("voice_track") and keep and not any(a["title"] == VOICE_TITLE for a in audio):
        for lang in [norm_lang(x) for x in s.get("voice_langs", [])] + [None]:
            cands = [a for a in keep if lang is None or a["lang"] == lang]
            if cands:
                voice = next((a for a in cands if a["default"]), cands[0])
                break
    return {"keep": keep, "drop": [a for a in audio if a not in keep], "voice": voice,
            "subs_keep": subs_keep, "subs_drop": [t for t in info.get("subs", []) if t not in subs_keep]}


def processing_reasons(sel: dict) -> list[str]:
    r = []
    if sel["voice"]:
        r.append(f'добавить «{VOICE_TITLE}» ({sel["voice"]["lang"] or "?"})')
    if sel["drop"]:
        r.append("убрать дорожки: " + ", ".join(a["lang"] or "?" for a in sel["drop"]))
    if sel["subs_drop"]:
        r.append("убрать субтитры: " + ", ".join(t["lang"] or "?" for t in sel["subs_drop"]))
    return r


def target_size(w: int, h: int) -> tuple[int, int]:
    f = min(1920 / w, 1080 / h, 1.0) if w and h else 1.0
    return int(w * f) // 2 * 2, int(h * f) // 2 * 2


def target_kbps(w: int, h: int, quality: str, src_bps: int | None) -> int:
    k = QUALITY.get(quality, QUALITY["standard"]) * (w * h) / (1920 * 1080)
    k = max(k, 1200)
    if src_bps:
        k = min(k, src_bps * 0.85 / 1000)
    return int(max(k, 800))


def estimate_bytes(info: dict, s: dict, transcode: bool) -> int:
    if not transcode or not info.get("video") or not info.get("duration"):
        return int(info.get("size") or 0)
    v = info["video"]
    w, h = target_size(v["width"], v["height"])
    video = target_kbps(w, h, s.get("video_quality", "standard"), v.get("bps")) * 1000
    audio = sum(a["bps"] or AUDIO_DEFAULT_BPS.get(a["codec"], 256_000) for a in select_tracks(info, s)["keep"])
    return int((video + audio + 192_000) * info["duration"] / 8)


def voice_filter(channels: int, strength: str) -> str:
    p = VOICE.get(strength, VOICE["medium"])
    c = p["center"]
    if channels >= 3:
        mix = (f"aformat=channel_layouts=5.1,"
               f"pan=stereo|FL={c}*FC+0.7*FL+0.5*BL+0.3*LFE|FR={c}*FC+0.7*FR+0.5*BR+0.3*LFE")
    else:
        mix = "aformat=channel_layouts=stereo"
    return f"{mix},{p['dyn']},loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000"


# --- command -------------------------------------------------------------------------
def build_command(src: str, dst: str, info: dict, s: dict, *, transcode: bool,
                  hw: str = "vaapi", hw_decode: bool = True) -> list[str]:
    v = info["video"]
    sel = select_tracks(info, s)
    dev = s.get("hw_device", "/dev/dri/renderD128")
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
    if transcode:
        # filters (scaling, HDR tone mapping on the CPU) are single-threaded by default
        cmd += ["-filter_threads", str(os.cpu_count() or 4)]
    if transcode and hw == "vaapi":
        cmd += ["-init_hw_device", f"vaapi=va:{dev}", "-filter_hw_device", "va"]
        if hw_decode:
            cmd += ["-hwaccel", "vaapi", "-hwaccel_device", "va", "-hwaccel_output_format", "vaapi"]
    cmd += ["-i", src, "-map", f"0:{v['index']}"]
    if sel["voice"]:
        cmd += ["-map", f"0:{sel['voice']['index']}"]
    for a in sel["keep"]:
        cmd += ["-map", f"0:{a['index']}"]
    for t in sel["subs_keep"]:
        cmd += ["-map", f"0:{t['index']}"]
    cmd += ["-map", "0:t?"]

    if not transcode:
        cmd += ["-c:v", "copy"]
    else:
        w, h = target_size(v["width"], v["height"])
        kbps = target_kbps(w, h, s.get("video_quality", "standard"), v.get("bps"))
        cmd += ["-vf", video_filter(v, w, h, hw, hw_decode)]
        if hw == "vaapi":
            cmd += ["-c:v", "h264_vaapi", "-profile:v", "high", "-rc_mode", "VBR",
                    "-b:v", f"{kbps}k", "-maxrate", f"{int(kbps * 1.6)}k", "-bufsize", f"{kbps * 2}k"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "21", "-profile:v", "high", "-pix_fmt", "yuv420p",
                    "-maxrate", f"{int(kbps * 1.6)}k", "-bufsize", f"{kbps * 2}k"]
        if v["hdr"]:
            cmd += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]
        cmd += _clear_stats("v:0")

    out = 0
    if sel["voice"]:
        lang = sel["voice"]["lang"] or "und"
        cmd += [f"-filter:a:{out}", voice_filter(sel["voice"]["channels"], s.get("voice_strength", "medium")),
                f"-c:a:{out}", "aac", f"-b:a:{out}", "192k",
                f"-metadata:s:a:{out}", f"title={VOICE_TITLE}", f"-metadata:s:a:{out}", f"language={lang}",
                f"-disposition:a:{out}", "default", *_clear_stats(f"a:{out}")]
        out += 1
    has_default = any(a["default"] for a in sel["keep"])
    for i, a in enumerate(sel["keep"]):
        cmd += [f"-c:a:{out}", "copy"]
        if sel["voice"]:
            cmd += [f"-disposition:a:{out}", "0"]
        elif not has_default and i == 0:
            cmd += [f"-disposition:a:{out}", "default"]
        out += 1
    for j, t in enumerate(sel["subs_keep"]):
        cmd += [f"-c:s:{j}", "srt" if t["codec"] == "mov_text" else "copy"]
    cmd += ["-c:t", "copy", "-map_metadata", "0", "-map_chapters", "0",
            "-max_muxing_queue_size", "4096", "-f", "matroska", dst]
    return cmd


STATS_TAGS = ("BPS", "NUMBER_OF_BYTES", "NUMBER_OF_FRAMES", "DURATION")


def _clear_stats(spec: str) -> list[str]:
    """mkvmerge statistics tags are copied from the source and would describe the old stream."""
    out = []
    for tag in STATS_TAGS:
        for name in (tag, f"{tag}-eng"):
            out += [f"-metadata:s:{spec}", f"{name}="]
    return out


def video_filter(v: dict, w: int, h: int, hw: str, hw_decode: bool) -> str:
    hdr, inter = v["hdr"], v["interlaced"]
    if hw == "vaapi" and hw_decode:
        parts = ["deinterlace_vaapi"] if inter else []
        if not hdr:
            return ",".join(parts + [f"scale_vaapi=w={w}:h={h}:format=nv12"])
        return ",".join(parts + [f"scale_vaapi=w={w}:h={h}:format=p010", "hwdownload", "format=p010le",
                                 TONEMAP, "format=nv12", "hwupload"])
    parts = ["bwdif"] if inter else []
    if (w, h) != (v["width"], v["height"]):
        parts.append(f"scale={w}:{h}:flags=lanczos")
    if hdr:
        parts.append(TONEMAP)
    parts += ["format=nv12", "hwupload"] if hw == "vaapi" else ["format=yuv420p"]
    return ",".join(parts)


# --- hardware ------------------------------------------------------------------------
@functools.lru_cache(maxsize=4)
def _vaapi_works(dev: str) -> tuple[bool, str]:
    if not os.path.exists(dev):
        return False, f"нет устройства {dev} (в контейнер не передан /dev/dri)"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-init_hw_device", f"vaapi=va:{dev}",
           "-filter_hw_device", "va", "-f", "lavfi", "-i", "testsrc2=duration=1:size=1280x720:rate=25",
           "-vf", "format=nv12,hwupload", "-c:v", "h264_vaapi", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, str(e)
    if r.returncode != 0:
        return False, r.stderr.strip()[-300:] or "ошибка ffmpeg"
    return True, "аппаратное кодирование H.264 работает"


def hardware(s: dict) -> tuple[str, str]:
    """Return ("vaapi" | "software", explanation)."""
    if s.get("hw_accel") == "software":
        return "software", "выбрано кодирование на процессоре"
    ok, msg = _vaapi_works(s.get("hw_device", "/dev/dri/renderD128"))
    if ok:
        return "vaapi", msg
    return "software", f"видеокарта недоступна ({msg}), будет кодирование на процессоре — это медленно"


# --- run -----------------------------------------------------------------------------
class FFmpegError(RuntimeError):
    pass


def run(cmd: list[str], duration: float, progress: Callable[[float, str], None] | None = None) -> None:
    with tempfile.TemporaryFile(mode="w+") as err:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        speed = ""
        assert p.stdout is not None
        for line in p.stdout:
            key, _, val = line.strip().partition("=")
            if key == "speed":
                speed = val
            elif key in ("out_time_us", "out_time_ms") and val.isdigit() and duration and progress:
                progress(min(int(val) / 1e6 / duration, 1.0), speed)
        rc = p.wait()
        if rc != 0:
            err.seek(0)
            raise FFmpegError(f"ffmpeg завершился с кодом {rc}: {err.read().strip()[-600:]}")


def process(src: str, dst: str, info: dict, s: dict, *, transcode: bool,
            progress: Callable[[float, str], None] | None = None) -> dict:
    """Write dst from src (re-encode video or only rework audio/subtitles). Safe for src == dst."""
    hw, _ = hardware(s) if transcode else ("software", "")
    v = info["video"]
    can_hw_decode = v["codec"] in HW_DECODE and not (v["codec"] == "h264" and "10" in v["pix_fmt"])
    attempts = [("vaapi", True), ("vaapi", False)] if hw == "vaapi" else [("software", False)]
    if hw == "vaapi" and not can_hw_decode:
        attempts = [("vaapi", False)]
    if not transcode:
        attempts = [("software", False)]
    partial = Path(dst).with_name(Path(dst).stem + ".partial.mkv")
    started, errors = time.time(), []
    try:
        for hw_kind, hw_decode in attempts:
            cmd = build_command(src, str(partial), info, s, transcode=transcode, hw=hw_kind, hw_decode=hw_decode)
            try:
                run(cmd, info.get("duration") or 0, progress)
                break
            except FFmpegError as e:
                errors.append(str(e))
        else:
            raise FFmpegError(" | ".join(errors))
        d_src, d_dst = info.get("duration") or 0, media.duration(partial)
        if d_src and abs(d_src - d_dst) > 2:
            raise FFmpegError(f"длительность не совпадает: {d_src:.0f} c → {d_dst:.0f} c")
        os.replace(partial, dst)
    finally:
        partial.unlink(missing_ok=True)
    return {"hw": attempts[0][0] if not errors else "fallback", "seconds": round(time.time() - started),
            "size": os.path.getsize(dst)}
