#!/usr/bin/env python3
import sys
import time
import numpy as np
sys.path.append('/home/morgan/dogbot')

from ai.bark_classifier import BarkClassifier
from audio.bark_buffer import BarkAudioBuffer

print("Initializing bark detection...")
classifier = BarkClassifier(
    model_path='/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
    emotion_mapping_path='/home/morgan/dogbot/ai/models/emotion_mapping.json'
)

buffer = BarkAudioBuffer(sample_rate=48000, chunk_duration=3.0)
print("Starting audio recording...")
buffer.start_recording()

print("\n🎤 LISTENING FOR BARKS - Play sounds now!\n")

try:
    for i in range(20):  # Listen for 10 seconds
        audio = buffer.get_audio_chunk(timeout=0.5)
        if audio is not None:
            energy = np.mean(np.abs(audio))
            print(f"Audio chunk {i}: energy={energy:.4f}")
            if energy > 0.01:
                result = classifier.predict(audio)
                print(f"  -> {result['emotion']} ({result['confidence']:.2%})")
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    buffer.stop_recording()
    print("Done")
