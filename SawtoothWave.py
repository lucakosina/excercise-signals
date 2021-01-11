
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal


#sawtooth
st_4 = 0.025
time = np.arange(0, 10, st_4)
signal_4 = signal.sawtooth(2*np.pi*time)
plt.plot(time, signal_4)
plt.title("Sawtooth")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()


f_max_4 = (1/st_4)/2
p4 = 20*np.log10(np.abs(np.fft.rfft(signal_4)-5))
f4 = np.linspace(0, f_max_4, len(p4))
k4 = 14 + 28*np.sinc(f4*5-5) 
plt.plot(f4, p4, '-', f4, k4, '-.')
plt.title("Spectrum of a sawtooth signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()

