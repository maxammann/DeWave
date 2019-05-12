import glob
import librosa
import os
import numpy as np
from .constant import *
import argparse

def audio_clip(data_dir, N, low, high, duration, output_dir):
    speakers = glob.glob(os.path.join(data_dir, "**/*.sph"), recursive=True)
    speakers.extend(glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True))
    speakers.extend(glob.glob(os.path.join(data_dir, "**/*.m4a"), recursive=True))
    for i in range(len(speakers)):
        p = os.path.join(output_dir, str(i))
        if not os.path.exists(p):
            os.makedirs(p)
        y, _ = librosa.load(speakers[i], sr=SAMPLING_RATE)
        for j in range(N):
            k = int(np.random.randint(low,  min(high, (len(y) - duration*SAMPLING_RATE) / SAMPLING_RATE), size=1))
            librosa.output.write_wav(os.path.join(p, str(j)) + ".wav", 
              y[k*SAMPLING_RATE : (k+duration)*SAMPLING_RATE], SAMPLING_RATE)
