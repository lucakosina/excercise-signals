import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
import math

#square
st_3 = 0.001
time = np.arange(0, 10, st_3)
signal_3 = signal.square(time)
plt.plot(time, signal_3)
plt.title("Square")
plt.show()

f_max_3 = (1/st_3)/2
p3 = 20*np.log10(np.abs(np.fft.rfft(signal_3)))
f3 = np.linspace(0.0001, f_max_3, len(p3))
frequency = f3 * (1/st_3) / 25001
k3 = 120 * np.sinc(0.5*frequency + 0.56) + 5
#k3 = 1000 * np.sinc(0.2*f3+4) + 23
plt.plot(f3, p3, '-', f3, k3, '.-')
plt.title("Spectrum of a square signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()


#def func(x, a, b, c):
#    return a * np.sinc(b*x+c) + d
#popt, pcov = sp.optimize.curve_fit(func, f3, p3, maxfev = 500)
#print(popt)

