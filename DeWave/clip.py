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
    for speaker_dir in speakers_dirs:
        speaker = os.path.basename(speaker_dir)
        print("Speaker:\t%s" % speaker)
        print("Source:\t%s" % speaker_dir)
        
        audio_files = glob.glob(os.path.join(speaker_dir, "**/*.sph"), recursive=True)
        audio_files.extend(glob.glob(os.path.join(speaker_dir, "**/*.wav"), recursive=True))
        audio_files.extend(glob.glob(os.path.join(speaker_dir, "**/*.m4a"), recursive=True))

        print("Source files:\t%d" % len(audio_files))

        p = os.path.join(output_dir, speaker)
        if not os.path.exists(p):
            os.makedirs(p)
        
        for j in range(N):
            audio_file = np.random.choice(audio_files)
            print("\t%s" % audio_file)
            y, _ = librosa.load(audio_file, sr=SAMPLING_RATE)

            if len(y) > duration * SAMPLING_RATE:
                # Select randomly
                k = int(np.random.uniform(low,  min(high, (len(y) - duration*SAMPLING_RATE) / SAMPLING_RATE), size=1))
                utterance = y[k*SAMPLING_RATE : math.floor((k+duration)*SAMPLING_RATE)]
            else:
                # Use whole
                utterance = y
            
            librosa.output.write_wav(os.path.join(p, str(j)) + ".wav", utterance, SAMPLING_RATE)
