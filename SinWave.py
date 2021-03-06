
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
import math

#sin
st_1 = 0.05
time = np.arange(0, 10, st_1)
signal_1 = np.sin(time*np.pi*2)
plt.plot(time, signal_1)
plt.title("Sine")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()

f_max_1 = (1/st_1)/2
p = 20*np.log10(np.abs(np.fft.rfft(signal_1)))
delta_f_1 = f_max_1/(len(p)-1)
f = np.linspace(0, f_max_1, len(p))
k = -285 + 255 * (math.sqrt(np.pi/2)*(signal.unit_impulse(np.shape(f), idx = int(1/delta_f_1))))
plt.plot(f, p, '-', f, k, '-.')
plt.title("Spectrum of a sin wave")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()
