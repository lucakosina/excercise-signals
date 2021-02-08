
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
k4 = [0] * 201
k4[0] = 1 * 21
k4[10] = 1 * 42
k4[20] = 1 * 37
k4[30] = 1 * 33
k4[40] = 1 * 30
k4[50] = 1 * 28
k4[60] = 1 * 27
k4[70] = 1 * 26
k4[80] =1 * 25
k4[90] = 1 * 24.5
k4[100] = 1 * 23.5
k4[110] =  1 * 23
#x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
#for elem in x:
#    k4[elem] = 1

#k4 = 42.1127*signal.unit_impulse(np.shape(f4), idx = int(1/delta_f_4)) 
plt.plot(f4, p4, '-', f4, k4, '-.')
plt.title("Spectrum of a sawtooth signal")
plt.ylabel("Spectral Intensity / Decibel")
plt.xlabel("Frequency")
plt.show()


