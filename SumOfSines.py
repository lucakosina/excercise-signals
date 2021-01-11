
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
import math

#sum of sines
st_2 = 0.1
time = np.arange(0, 10, st_2)
sin_one = np.sin(time*np.pi*2)
sin_two = np.sin(time*np.pi*2/5)
signal_2 = sin_one + sin_two
plt.plot(time, signal_2)
plt.title("Sum of sines")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()

f_max_2 = (1/st_2)/2
p2 = 20*np.log10(np.abs(np.fft.rfft(signal_2)))
delta_f_2 = f_max_2/(len(p2)-1)
f2 = np.linspace(0, f_max_2, len(p2))
k2 = -285 + 315 * (signal.unit_impulse(np.shape(f2), idx = int(1/delta_f_2))) 
k2_2 = -285 + 315 * (signal.unit_impulse(np.shape(f2), idx = int((1/5)/delta_f_2)))
plt.plot(f2, p2, '-', f2, k2, '-.', f2, k2_2, '-.')
plt.title("Spectrum of a sum of sines")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()
