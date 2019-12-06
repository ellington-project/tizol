#!/usr/bin/env python3

import os
import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

in_f = sys.argv[1]
iterations = int(sys.argv[2])

t0 = time.clock()

for i in range(0, iterations):
    samplerate = 44100
    lowf = 0  # 64
    highf = 1024  # 320

    # fix the sample rate, otherwise we'll never be able to see if caches are correct...
    (y, sr) = librosa.load(in_f, sr=samplerate, res_type='kaiser_best')

    S = librosa.stft(y)
    M = librosa.core.magphase(S)[0]
    # Compute amplitude to db, and cut off the high/low frequencies
    spect = librosa.amplitude_to_db(M, ref=np.max)[lowf:highf, :]

    (h, w) = spect.shape
    fig = plt.figure(figsize=(w / 100, h / 100))
    ax = plt.subplot(111)
    ax.set_frame_on(False)
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    # Perform some magnitue-to-frequency calculations, and write the result to the figure
    librosa.display.specshow(spect, y_axis='linear', sr=samplerate)

    plt.close(fig)

    tn = time.clock()
    print("Current progress: {n} / {t}s / {m}s".format(n=i,
                                                       t=(tn - t0),
                                                       m=((tn - t0) /
                                                          (i + 1))))

t1 = time.clock()

elapsed = t1 - t0
mean = elapsed / iterations
print("Elapsed time: {el}s, Mean {me}s".format(el=elapsed, me=mean))