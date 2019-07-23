#!/usr/bin/env python3

import os
import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

filename = "One O'Clock Jump - Metronome All Star Band.mp3"

output = "reference.png"

samplerate = 44100
lowf = 0  # 64
highf = 1024  # 320

# fix the sample rate, otherwise we'll never be able to see if caches are correct...
(y, sr) = librosa.load(filename, sr=samplerate, res_type='kaiser_best')

print("Loaded samples")

print("Dumped samples")

S = librosa.stft(y)
M = librosa.core.magphase(S)[0]
# Compute amplitude to db, and cut off the high/low frequencies
spect = librosa.amplitude_to_db(M, ref=np.max)[lowf:highf, :]

print("Audio of shape: " + str(y.shape))
print("Sample rate: " + str(sr))
print("Audio of time: " + str(y.shape[0] / sr))
print("Spectrogram of shape: " + str(spect.shape))

(h, w) = spect.shape
fig = plt.figure(figsize=(w / 100, h / 100))
ax = plt.subplot(111)
ax.set_frame_on(False)
plt.axis('off')
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

# Perform some magnitue-to-frequency calculations, and write the result to the figure
librosa.display.specshow(spect, y_axis='linear', sr=samplerate)

# Save the figure, and close it
fig.savefig(output, dpi=100, bbox_inches='tight', pad_inches=0.0)
plt.close(fig)

# reshape, and then save the data
# spect_cm = spect.reshape(w, h, order='F')
spect_cm = spect.transpose()

# sample_str = ",".join(map(lambda l: ",".join(map(str, l)), spect_cm.tolist()))
# with open("python_samples.txt", "w") as f:
#     f.writelines(sample_str)