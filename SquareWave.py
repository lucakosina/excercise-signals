import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal

#square
st_3 = 0.25
time = np.arange(0, 10, st_3)
signal_3 = signal.square(time)
plt.plot(time, signal_3)
plt.title("Square")
plt.show()

f_max_3 = (1/st_3)/2
p3 = 20*np.log10(np.abs(np.fft.rfft(signal_3)))
f3 = np.linspace(0, f_max_3, len(p3))
k3 = 2.5895 + 6.1585e+02*np.exp(-f3) * np.cos(-8.8052e-03*f3 - 7.8218)
plt.plot(f3, p3, '-', f3, k3, '-.')
plt.title("Spectrum of a square signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()


def func(x, a, b, c, d):
    return a * np.exp(-x) * np.cos(b*x+c) + d
popt, pcov = sp.optimize.curve_fit(func, f3, p3, maxfev = 5000)
print(popt)