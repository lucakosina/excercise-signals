
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
delta_f_4 = f_max_4/(len(p4)-1)
k4 = np.arange(0, len(f4), 1)
def delta(k4):
        if k4 != int():
            return 1
        else:
            return 0
#k4 = 42.1127*signal.unit_impulse(np.shape(f4), idx = int(1/delta_f_4)) 
plt.plot(f4, p4, '-', f4, k4, '-.')
plt.title("Spectrum of a sawtooth signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()

def func(x, a):
    return a * signal.unit_impulse(np.shape(f4), idx = int(1/delta_f_4)) 
popt, pcov = sp.optimize.curve_fit(func, f4, p4)
print(popt)
