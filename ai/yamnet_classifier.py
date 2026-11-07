#!/usr/bin/env python3
"""
ai/yamnet_classifier.py - YAMNet (AudioSet, 521 classes) ambient-sound tagging on CPU.

Model: ai/models/yamnet.tflite (tfhub google/yamnet classification lite,
input float32 [15600] = 0.975 s @ 16 kHz mono, output [1, 521] scores) +
ai/models/yamnet_class_map.csv. Runtime: ai_edge_litert (+~23 MB RSS; the
same runtime the bark classifier prefers — never import full tensorflow in
the robot process, see requirements.txt / tb4 2026-10-24 lag RCA).

Used by the relay `listen` tool (2026-11-07). A clip longer than one window
is scored in 50 %-overlap windows and the scores averaged.
"""
import csv
import logging
import threading
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger('YAMNet')

MODEL_PATH = Path('/home/morgan/dogbot/ai/models/yamnet.tflite')
CLASS_MAP_PATH = Path('/home/morgan/dogbot/ai/models/yamnet_class_map.csv')
SAMPLE_RATE = 16000
WINDOW = 15600          # samples per model input (0.975 s)
HOP = WINDOW // 2


def read_wav_mono16k(path: str) -> np.ndarray:
    """WAV -> float32 mono in [-1, 1] at 16 kHz (resampled with linear
    interpolation if needed; the PTT path already records 16 kHz)."""
    with wave.open(str(path), 'rb') as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"unsupported sample width {width}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE and len(x):
        t_new = np.arange(0, len(x), rate / SAMPLE_RATE)
        x = np.interp(t_new, np.arange(len(x)), x).astype(np.float32)
    return x


def mean_dbfs(x: np.ndarray) -> float:
    """RMS level of the clip in dBFS (0 dB = full scale, silence ~ -90)."""
    if x.size == 0:
        return -90.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return round(20.0 * np.log10(max(rms, 1e-9)), 1)


class YAMNetClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, class_map_path: Path = CLASS_MAP_PATH,
                 num_threads: int = 2):
        self.model_path = Path(model_path)
        self.class_map_path = Path(class_map_path)
        self.num_threads = num_threads
        self._lock = threading.Lock()
        self._it = None
        self._in = None
        self._out = None
        self.labels: List[str] = []

    def available(self) -> bool:
        return self.model_path.exists() and self.class_map_path.exists()

    def _load(self) -> None:
        if self._it is not None:
            return
        from ai_edge_litert import interpreter as tfl
        with open(self.class_map_path, newline='') as f:
            self.labels = [r['display_name'] for r in csv.DictReader(f)]
        it = tfl.Interpreter(model_path=str(self.model_path), num_threads=self.num_threads)
        it.allocate_tensors()
        self._in = it.get_input_details()[0]
        outs = it.get_output_details()
        self._out = next((d for d in outs if int(d['shape'][-1]) == len(self.labels)), outs[0])
        self._it = it
        logger.info(f"YAMNet loaded: {self.model_path.name}, {len(self.labels)} classes, "
                    f"input {self._in['shape'].tolist()}")

    def classify_samples(self, x: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """x: float32 mono 16 kHz. Returns top_k [{label, confidence}] (mean
        score over windows). Never raises on empty input."""
        with self._lock:
            self._load()
            if x.size == 0:
                return []
            if len(x) < WINDOW:
                x = np.pad(x, (0, WINDOW - len(x)))
            scores = []
            for s in range(0, len(x) - WINDOW + 1, HOP):
                seg = np.ascontiguousarray(x[s:s + WINDOW], dtype=np.float32)
                self._it.set_tensor(self._in['index'], seg)
                self._it.invoke()
                scores.append(self._it.get_tensor(self._out['index']).reshape(-1, len(self.labels)).mean(0))
            m = np.mean(scores, axis=0)
        top = np.argsort(m)[::-1][:top_k]
        return [{'label': self.labels[i], 'confidence': round(float(m[i]), 3)} for i in top]

    def classify_wav(self, path: str, top_k: int = 5) -> Dict[str, Any]:
        x = read_wav_mono16k(path)
        return {'labels': self.classify_samples(x, top_k=top_k), 'mean_db': mean_dbfs(x),
                'seconds': round(len(x) / SAMPLE_RATE, 2)}


_instance: Optional[YAMNetClassifier] = None
_instance_lock = threading.Lock()


def get_yamnet() -> YAMNetClassifier:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = YAMNetClassifier()
        return _instance
