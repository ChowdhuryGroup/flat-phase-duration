import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

data = np.loadtxt('AfterChirpmirr_2ndStage.txt', skiprows=14)
wavelengths = data[:,0]
meas_frequencies = 2.*np.pi*3.e8/(wavelengths*1.e-9)
frequencies = np.linspace(meas_frequencies[-1], meas_frequencies[0], len(meas_frequencies), endpoint=True)
counts = data[:,1]

interpolator = interpolate.interp1d(meas_frequencies, counts)
interp_counts = interpolator(frequencies)

interp_counts -= interp_counts[0]
interp_counts[interp_counts<0] = 0.

i = np.argmax(interp_counts)
df = frequencies[1]-frequencies[0]
expanded_frequencies = np.linspace(frequencies[0], frequencies[0]+df*i*2, i*2, endpoint=True)
interpolator = interpolate.interp1d(frequencies, interp_counts, bounds_error=False, fill_value=0.)
expanded_counts = interpolator(expanded_frequencies)
spectral_envelope_amplitude = expanded_counts**0.5
shifted_frequencies = expanded_frequencies-expanded_frequencies[i]

envelope = np.abs(np.fft.fftshift(np.fft.ifft(spectral_envelope_amplitude)))
times = np.fft.fftshift(np.fft.fftfreq(len(spectral_envelope_amplitude), df))

peak_index = np.argmax(envelope)
for j in range(int(len(envelope)/5)):
    if envelope[peak_index+j] < envelope[peak_index]/2.:
        high_half_index = peak_index+j
        break
for j in range(int(len(envelope)/5)):
    if envelope[peak_index-j] < envelope[peak_index]/2.:
        low_half_index = peak_index-j
        break


duration = times[high_half_index] - times[low_half_index]

fig, axes = plt.subplots(1,2)

axes[0].plot(wavelengths, counts)
axes[0].set_xlabel('wavelengths/nm')
axes[0].set_ylabel('counts')
axes[1].plot(times, envelope)
axes[1].plot(times, np.ones_like(envelope)*(envelope[high_half_index]+envelope[high_half_index-1])/2., '--')
axes[1].set_title(str(duration*1.e15)[:5]+'fs FWHM Duration')
plt.show()
