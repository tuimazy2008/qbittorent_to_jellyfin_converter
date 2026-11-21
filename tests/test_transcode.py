import shutil
import subprocess

import pytest

from jellyprep import transcode
from conftest import fake_info

S = {"video_mode": "compat", "video_max_mbps": 12, "video_quality": "standard", "voice_track": True,
     "voice_strength": "medium", "voice_langs": ["rus", "eng"], "keep_langs": ["rus", "eng"],
     "keep_unknown_lang": True, "hw_device": "/dev/dri/renderD128"}


@pytest.mark.parametrize("info, mode, expect", [
    (fake_info(3840, 2160), "compat", True),
    (fake_info(hdr=True), "compat", True),
    (fake_info(codec="hevc"), "compat", True),
    (fake_info(codec="hevc"), "balanced", False),
    (fake_info(mbps=18), "compat", True),
    (fake_info(mbps=18), "balanced", False),
    (fake_info(mbps=8), "compat", False),
    (fake_info(1920, 1088), "compat", False),
    (fake_info(3840, 2160), "off", False),
])
def test_video_decision(info, mode, expect):
    assert transcode.video_decision(info, {**S, "video_mode": mode})[0] is expect


def test_track_selection():
    info = fake_info(audio=(("eng", 6), ("rus", 6), ("ukr", 2), ("", 2)), subs=("rus", "ukr", "eng"))
    sel = transcode.select_tracks(info, S)
    assert [a["lang"] for a in sel["keep"]] == ["eng", "rus", ""]
    assert [a["lang"] for a in sel["drop"]] == ["ukr"]
    assert sel["voice"]["lang"] == "rus"          # language priority beats file order
    assert [t["lang"] for t in sel["subs_keep"]] == ["rus", "eng"]
    reasons = transcode.processing_reasons(sel)
    assert any("Голос громче" in r for r in reasons) and any("ukr" in r for r in reasons)


def test_track_selection_keeps_everything_when_no_language_matches():
    sel = transcode.select_tracks(fake_info(audio=(("jpn", 2), ("ger", 2))), S)
    assert len(sel["keep"]) == 2 and sel["voice"]["lang"] == "jpn"


def test_already_processed_file_needs_nothing():
    info = fake_info(audio=(("rus", 2), ("rus", 6)))
    info["audio"][0]["title"] = transcode.VOICE_TITLE
    assert transcode.processing_reasons(transcode.select_tracks(info, S)) == []


@pytest.mark.parametrize("w, h, out", [((3840, 2160), None, (1920, 1080)), ((3840, 1606), None, (1920, 802)),
                                      ((1920, 1080), None, (1920, 1080)), ((688, 544), None, (688, 544))])
def test_target_size(w, h, out):
    assert transcode.target_size(*w) == out


def test_target_bitrate_never_above_source():
    assert transcode.target_kbps(1920, 1080, "standard", None) == 6000
    assert transcode.target_kbps(1920, 1080, "standard", 4_000_000) == 3400
    assert transcode.target_kbps(1280, 720, "standard", None) < 6000


def test_command_vaapi_sdr():
    cmd = transcode.build_command("in.mkv", "out.mkv", fake_info(3840, 2160), S, transcode=True)
    vf = cmd[cmd.index("-vf") + 1]
    assert "-hwaccel" in cmd and vf == "scale_vaapi=w=1920:h=1080:format=nv12"
    assert cmd[cmd.index("-c:v") + 1] == "h264_vaapi"


def test_command_vaapi_hdr_tonemaps_on_cpu_after_gpu_scaling():
    cmd = transcode.build_command("in.mkv", "out.mkv", fake_info(3840, 2160, codec="hevc", hdr=True), S, transcode=True)
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("scale_vaapi=w=1920:h=1080:format=p010,hwdownload")
    assert "tonemap=tonemap=hable" in vf and vf.endswith("hwupload")
    assert "-color_trc" in cmd


def test_command_software_and_audio_only():
    sw = transcode.build_command("in.mkv", "out.mkv", fake_info(3840, 2160), S, transcode=True, hw="software")
    assert sw[sw.index("-c:v") + 1] == "libx264" and "-hwaccel" not in sw
    proc = transcode.build_command("in.mkv", "out.mkv", fake_info(audio=(("rus", 6), ("ukr", 2))), S, transcode=False)
    assert proc[proc.index("-c:v") + 1] == "copy" and "-init_hw_device" not in proc
    maps = [proc[i + 1] for i, x in enumerate(proc) if x == "-map"]
    assert maps == ["0:0", "0:1", "0:1", "0:t?"]   # video, voice (from rus), rus original; ukr dropped
    f = proc[proc.index("-filter:a:0") + 1]
    assert "pan=stereo" in f and "loudnorm" in f
    assert proc[proc.index("-disposition:a:0") + 1] == "default"
    assert proc[proc.index("-disposition:a:1") + 1] == "0"


def test_voice_filter_stereo_source():
    f = transcode.voice_filter(2, "strong")
    assert "pan" not in f and "dynaudnorm=f=100" in f


def test_summarize_estimates_video_bitrate_from_total():
    probe = {"format": {"duration": "100", "size": str(125_000_000), "bit_rate": "10000000"},
             "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264", "width": 1920, "height": 1080,
                          "pix_fmt": "yuv420p"},
                         {"index": 1, "codec_type": "audio", "codec_name": "dts", "channels": 6,
                          "tags": {"language": "ru", "BPS": "1509000"}},
                         {"index": 2, "codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}}]}
    info = transcode.summarize(probe)
    assert info["video"]["bps"] == 10_000_000 - 1_509_000
    assert info["audio"][0]["lang"] == "rus"


# --- real ffmpeg -------------------------------------------------------------------
needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def make_clip(path, w=2560, h=1440, seconds=4):
    surround = "|".join(["0.3*sin(440*2*PI*t)", "0.3*sin(440*2*PI*t)", "0.5*sin(300*2*PI*t)", "0",
                         "0.2*sin(500*2*PI*t)", "0.2*sin(500*2*PI*t)"])
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=25:duration={seconds}",
        "-f", "lavfi", "-i", f"aevalsrc='{surround}':c=5.1:d={seconds}",
        "-f", "lavfi", "-i", f"sine=frequency=600:duration={seconds}",
        "-map", "0", "-map", "1", "-map", "2",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a:0", "ac3", "-c:a:1", "aac",
        "-metadata:s:a:0", "language=rus", "-metadata:s:a:1", "language=ukr", str(path)], check=True)


def streams(path):
    info = transcode.probe(path)
    return info, [(a["codec"], a["channels"], a["lang"], a["title"], a["default"]) for a in info["audio"]]


@needs_ffmpeg
def test_real_transcode_software(tmp_path):
    src, dst = tmp_path / "in.mkv", tmp_path / "out.mkv"
    make_clip(src)
    info = transcode.probe(src)
    res = transcode.process(str(src), str(dst), info, {**S, "hw_accel": "software"}, transcode=True)
    out, audio = streams(dst)
    assert (out["video"]["codec"], out["video"]["width"], out["video"]["height"]) == ("h264", 1920, 1080)
    assert audio == [("aac", 2, "rus", transcode.VOICE_TITLE, True), ("ac3", 6, "rus", "", False)]
    assert abs(out["duration"] - info["duration"]) < 0.5
    assert res["size"] > 0 and not (tmp_path / "out.partial.mkv").exists()


@needs_ffmpeg
def test_real_audio_only_in_place(tmp_path):
    src = tmp_path / "movie.mkv"
    make_clip(src, 640, 360)
    info = transcode.probe(src)
    transcode.process(str(src), str(src), info, S, transcode=False)   # src == dst: rewrite in place
    out, audio = streams(src)
    assert out["video"]["width"] == 640
    assert [a[3] for a in audio] == [transcode.VOICE_TITLE, ""]


@needs_ffmpeg
def test_real_transcode_vaapi_if_available(tmp_path):
    kind, msg = transcode.hardware({**S, "hw_accel": "vaapi"})
    if kind != "vaapi":
        pytest.skip(f"no VAAPI here: {msg}")
    src, dst = tmp_path / "in.mkv", tmp_path / "out.mkv"
    make_clip(src, 3840, 2160)
    res = transcode.process(str(src), str(dst), transcode.probe(src), {**S, "hw_accel": "vaapi"}, transcode=True)
    out, _ = streams(dst)
    assert (out["video"]["width"], out["video"]["height"]) == (1920, 1080)
    assert res["hw"] == "vaapi"


def test_max_tracks_per_language_keeps_default():
    info = fake_info(audio=(("rus", 6), ("rus", 6), ("rus", 2), ("eng", 6), ("eng", 2)))
    info["audio"][2]["default"] = True
    info["audio"][0]["default"] = False
    sel = transcode.select_tracks(info, {**S, "max_tracks_per_lang": 1})
    assert [(a["lang"], a["index"]) for a in sel["keep"]] == [("rus", 1), ("rus", 3), ("eng", 4)]


def test_transcode_command_threads_and_stale_stats():
    cmd = transcode.build_command("in.mkv", "out.mkv", fake_info(3840, 2160), S, transcode=True)
    assert "-filter_threads" in cmd
    assert "-metadata:s:v:0" in cmd and "BPS=" in cmd and "BPS-eng=" in cmd
    i = cmd.index("-disposition:a:0")
    assert "-metadata:s:a:0" in cmd[i:]


@needs_ffmpeg
def test_real_output_has_no_stale_bitrate_tags(tmp_path):
    src, dst = tmp_path / "in.mkv", tmp_path / "out.mkv"
    make_clip(src, 2560, 1440)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0", "-c", "copy",
                    "-metadata:s:v:0", "BPS=99000000", str(tmp_path / "tagged.mkv")], check=True)
    info = transcode.probe(tmp_path / "tagged.mkv")
    assert info["video"]["bps"] == 99_000_000
    transcode.process(str(tmp_path / "tagged.mkv"), str(dst), info, {**S, "hw_accel": "software"}, transcode=True)
    assert transcode.probe(dst)["video"]["bps"] < 20_000_000


def test_empty_language_list_keeps_everything():
    info = fake_info(audio=(("rus", 6), ("ukr", 2), ("", 2)), subs=("rus", "ukr"))
    sel = transcode.select_tracks(info, {**S, "keep_langs": [], "voice_track": False})
    assert len(sel["keep"]) == 3 and len(sel["subs_keep"]) == 2
    assert transcode.processing_reasons(sel) == []


@needs_ffmpeg
def test_inplace_rerun_skips_finished_files(tmp_path):
    from jellyprep import executor
    f = tmp_path / "Show S01E01.mkv"
    make_clip(f, 640, 360)
    info = transcode.probe(f)
    transcode.process(str(f), str(f), info, S, transcode=False)        # first run adds the voice track
    plan = {"target": str(tmp_path), "actions": [{"src": str(f), "dst": str(f), "action": "process", "inplace": True,
                                                   "type": "video", "size": f.stat().st_size, "info": info}]}
    res = executor.execute(plan, log=lambda m: None, settings=S)
    assert res[0]["method"] == "exists"                                  # second run (after a restart) skips it


def test_size_check_catches_half_downloaded_sources():
    info = fake_info(mbps=4, duration=1473)          # ~740 MB of video
    with pytest.raises(transcode.FFmpegError, match="недокачан"):
        transcode.check_size(info, S, False, 7_000_000)
    transcode.check_size(info, S, False, 760_000_000)  # normal: video copied + voice track
    with pytest.raises(transcode.FFmpegError, match="повреждён"):
        transcode.check_size(fake_info(3840, 2160, mbps=20), S, True, 5_000_000)
    transcode.check_size(fake_info(3840, 2160, mbps=20), S, True, 5_000_000_000)


@needs_ffmpeg
def test_real_truncated_source_keeps_original(tmp_path):
    src = tmp_path / "ep.mkv"
    make_clip(src, 640, 360, seconds=4)
    info = transcode.probe(src)
    info["video"]["bps"] = 50_000_000                # pretend the header promised much more video
    before = src.read_bytes()
    with pytest.raises(transcode.FFmpegError):
        transcode.process(str(src), str(src), info, S, transcode=False)
    assert src.read_bytes() == before and not (tmp_path / "ep.partial.mkv").exists()
