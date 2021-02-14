import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
import math

#square
st_3 = 0.001
time = np.arange(0, 500, st_3)
signal_3 = signal.square(time)
plt.plot(time, signal_3)
plt.title("Square")
plt.show()

f_max_3 = (1/st_3)/2
p3 = 20*np.log10(np.abs(np.fft.rfft(signal_3)))
f3 = np.linspace(0.0001, f_max_3, len(p3))
k3 = 3.5288 * (np.log(f3*0.2e-04)/np.log(0.5)) + 10
#7e+01 * np.exp(-f3*5e-02) + 4e+01
#-308 * np.sinc(0.09*f3 + 1.01) + 5
#k3 = 6.2233 + 6.8e+01*np.exp(-f3*10e-02) * np.cos(-15*f3 - 2.1871e+02)
plt.plot(f3, p3, '-', f3, k3, '.-')
plt.title("Spectrum of a square signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()


def func(x, a, b, c):
    f3 = np.linspace(0.0001, f_max_3, len(p3))
    return a * (np.log(f3*b)/np.log(0.5)) + c
#np.exp(-x*b) + c
#np.sinc(b*x+c) + d
#np.exp(-x*e) * np.cos(b*x+c) + d
popt, pcov = sp.optimize.curve_fit(func, f3, p3, maxfev = 500)
print(popt)

