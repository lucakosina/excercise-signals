
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
p4 = 20*np.log10(np.abs(np.fft.rfft(signal_4)-1))
f4 = np.linspace(0, f_max_4, len(p4))
k4 = 2.53209 + 1.53636*np.sinc(f4*0.75085-1.53636) 
plt.plot(f4, p4, '-', f4, k4, '-.')
plt.title("Spectrum of a sawtooth signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()

def func(x, a, b, c, d):
    return a * np.sinc(b*x-c) + d 
popt, pcov = sp.optimize.curve_fit(func, f4, p4)
print(popt)

