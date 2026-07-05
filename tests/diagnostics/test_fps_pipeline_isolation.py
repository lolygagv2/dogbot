#!/usr/bin/env python3
"""Work order §1A FPS/pipeline isolation benchmarks (2026-06-21 diagnostics).

Runs each pipeline stage in isolation and reports the §1C table.
STOP treatbot.service FIRST — these tests need exclusive camera + Hailo.

    sudo systemctl stop treatbot
    env_new/bin/python tests/diagnostics/test_fps_pipeline_isolation.py [--duration 60] [--tests 1,2,3,4,5,6]
    sudo systemctl start treatbot

TEST 1: pure Hailo inference (dummy 640x640)      — raw accelerator ceiling
TEST 2: camera capture only, 640x480              — camera+ISP cost
TEST 3: camera capture only, 1920x1080            — resolution scaling cost
TEST 4: camera 640 + Hailo                        — real vision pipeline, no encode
TEST 5: dual-stream camera + Hailo + VP8 720p enc — what runs today (encode tax)
TEST 6: camera 1080 + cv2 resize to 640 + Hailo   — downscale-in-Python cost
"""
import argparse
import subprocess
import time

import numpy as np

MODEL = "/home/morgan/dogbot/ai/models/dogdetector_14.hef"
RESULTS = {}


def cpu_sample_start():
    import psutil
    psutil.cpu_percent(percpu=True)  # prime
    return psutil


def cpu_sample_end(psutil):
    per_core = psutil.cpu_percent(percpu=True)
    return per_core


def throttled():
    try:
        out = subprocess.run(['vcgencmd', 'get_throttled'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out
    except Exception:
        return "unknown"


class HailoRunner:
    """Minimal InferVStreams wrapper (per work order TEST 1 script)."""

    def __init__(self):
        # Mirrors core/ai_controller_3stage_fixed.py:272-302 exactly
        import hailo_platform as hpf
        self.hef = hpf.HEF(MODEL)
        self.target = hpf.VDevice()
        params = hpf.ConfigureParams.create_from_hef(
            hef=self.hef, interface=hpf.HailoStreamInterface.PCIe)
        self.network_group = self.target.configure(self.hef, params)[0]
        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.input_shape = self.input_info.shape
        self.ng_params = self.network_group.create_params()
        self.input_params = hpf.InputVStreamParams.make_from_network_group(
            self.network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        self.output_params = hpf.OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)
        self._InferVStreams = hpf.InferVStreams

    def pipeline(self):
        from contextlib import contextmanager

        @contextmanager
        def _pipe():
            with self.network_group.activate(self.ng_params):
                with self._InferVStreams(self.network_group, self.input_params,
                                         self.output_params) as p:
                    yield p
        return _pipe()

    def infer(self, pipeline, frame):
        return pipeline.infer({self.input_info.name: np.expand_dims(frame, axis=0)})

    def close(self):
        self.target.release()


def make_camera(main_size, lores=None):
    from picamera2 import Picamera2
    cam = Picamera2()
    kwargs = {"main": {"size": main_size, "format": "RGB888"}}
    if lores:
        kwargs["lores"] = {"size": lores, "format": "RGB888"}
    cam.configure(cam.create_video_configuration(**kwargs))
    cam.start()
    time.sleep(2)  # let auto-exposure settle
    return cam


def record(name, frames, elapsed, per_core):
    fps = frames / elapsed if elapsed else 0
    RESULTS[name] = (fps, per_core)
    print(f"{name}: {fps:.1f} FPS over {elapsed:.1f}s | CPU/core: "
          f"{[f'{c:.0f}' for c in per_core]}")


def test1(duration):
    print("\n=== TEST 1: pure Hailo inference (no camera/stream) ===")
    h = HailoRunner()
    print(f"Model input shape: {h.input_shape}")
    dummy = np.random.randint(0, 255, size=h.input_shape, dtype=np.uint8)
    ps = cpu_sample_start()
    frames, start = 0, time.time()
    with h.pipeline() as pipe:
        while time.time() - start < duration:
            h.infer(pipe, dummy)
            frames += 1
    record("TEST1 pure Hailo", frames, time.time() - start, cpu_sample_end(ps))
    h.close()


def _camera_only(name, size, duration):
    cam = make_camera(size)
    ps = cpu_sample_start()
    frames, start = 0, time.time()
    while time.time() - start < duration:
        cam.capture_array()
        frames += 1
    record(name, frames, time.time() - start, cpu_sample_end(ps))
    cam.stop()
    cam.close()


def test2(duration):
    print("\n=== TEST 2: camera only 640x480 ===")
    _camera_only("TEST2 camera 640x480", (640, 480), duration)


def test3(duration):
    print("\n=== TEST 3: camera only 1920x1080 ===")
    _camera_only("TEST3 camera 1080p", (1920, 1080), duration)


def test4(duration):
    print("\n=== TEST 4: camera 640 + Hailo (no stream) ===")
    h = HailoRunner()
    cam = make_camera((640, 640))
    ps = cpu_sample_start()
    frames, start = 0, time.time()
    with h.pipeline() as pipe:
        while time.time() - start < duration:
            frame = cam.capture_array()
            h.infer(pipe, frame)
            frames += 1
    record("TEST4 camera+Hailo", frames, time.time() - start, cpu_sample_end(ps))
    cam.stop()
    cam.close()
    h.close()


def test5(duration):
    """Today's real pipeline shape: dual-stream capture, lores->Hailo,
    main 720p -> VP8 encode (aiortc uses VP8; PyAV mirrors that encoder)."""
    print("\n=== TEST 5: dual-stream + Hailo + VP8 720p encode ===")
    import av
    h = HailoRunner()
    cam = make_camera((1280, 720), lores=(640, 640))

    codec = av.CodecContext.create('libvpx', 'w')
    codec.width, codec.height = 1280, 720
    codec.pix_fmt = 'yuv420p'
    codec.bit_rate = 1_500_000  # matches webrtc.py max_bitrate
    codec.framerate = 15
    codec.open()

    ps = cpu_sample_start()
    frames, start = 0, time.time()
    with h.pipeline() as pipe:
        while time.time() - start < duration:
            (lores, main), _md = cam.capture_arrays(["lores", "main"])
            h.infer(pipe, lores)
            vf = av.VideoFrame.from_ndarray(main, format='rgb24')
            vf.pts = frames
            codec.encode(vf)
            frames += 1
    record("TEST5 dual+Hailo+VP8", frames, time.time() - start, cpu_sample_end(ps))
    cam.stop()
    cam.close()
    h.close()


def test6(duration):
    print("\n=== TEST 6: camera 1080 + resize to 640 + Hailo ===")
    import cv2
    h = HailoRunner()
    cam = make_camera((1920, 1080))
    ps = cpu_sample_start()
    frames, start = 0, time.time()
    with h.pipeline() as pipe:
        while time.time() - start < duration:
            frame = cam.capture_array()
            small = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_AREA)
            h.infer(pipe, small)
            frames += 1
    record("TEST6 1080+resize+Hailo", frames, time.time() - start, cpu_sample_end(ps))
    cam.stop()
    cam.close()
    h.close()


TESTS = {1: test1, 2: test2, 3: test3, 4: test4, 5: test5, 6: test6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=int, default=60)
    ap.add_argument('--tests', default='1,2,3,4,5,6')
    args = ap.parse_args()

    wanted = [int(t) for t in args.tests.split(',')]
    print(f"Throttle status before: {throttled()}")
    for t in wanted:
        try:
            TESTS[t](args.duration)
        except Exception as e:
            print(f"TEST {t} FAILED: {e}")
        time.sleep(2)  # cool-down / device release between tests
    print(f"\nThrottle status after: {throttled()}")

    print("\n===== §1C REPORT TABLE =====")
    for name, (fps, per_core) in RESULTS.items():
        print(f"{name:28s} {fps:7.1f} FPS   CPU/core: {[f'{c:.0f}%' for c in per_core]}")


if __name__ == '__main__':
    main()
