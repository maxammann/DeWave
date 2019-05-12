import glob
import librosa
import os
import numpy as np
from .constant import *
import argparse
import math

def audio_clip(data_dir, N, low, high, duration, output_dir):
    speakers_dirs = [os.path.join(data_dir, i) for i in os.listdir(data_dir)\
                        if os.path.isdir(os.path.join(data_dir, i))]

    print("Speaker count: %s" % len(speakers_dirs))
    for i in range(len(speakers_dirs)):
        print("Clipping from speaker %d: %s" % (i, len(speakers_dirs)))
        speaker_dir = speakers_dirs[i]
        audio_files = glob.glob(os.path.join(speaker_dir, "**/*.sph"), recursive=True)
        audio_files.extend(glob.glob(os.path.join(speaker_dir, "**/*.wav"), recursive=True))
        audio_files.extend(glob.glob(os.path.join(speaker_dir, "**/*.m4a"), recursive=True))

        p = os.path.join(output_dir, str(i))
        if not os.path.exists(p):
            os.makedirs(p)
        
        for audio_file in audio_files:
            y, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
            for j in range(N):
                k = int(np.random.uniform(low,  min(high, (len(y) - duration*SAMPLING_RATE) / SAMPLING_RATE), size=1))
                librosa.output.write_wav(os.path.join(p, str(j)) + ".wav", 
                y[k*SAMPLING_RATE : math.floor((k+duration)*SAMPLING_RATE)], SAMPLING_RATE)
