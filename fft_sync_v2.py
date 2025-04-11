import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, freqz

# === 1. Load modulated audio ===
sample_rate, modulated_signal = wavfile.read("modulated_noisy_audio.wav")

# Normalize if int16
if modulated_signal.dtype == np.int16:
    modulated_signal = modulated_signal.astype(np.float32) / 32767

n = len(modulated_signal)
time = np.arange(n) / sample_rate

# === 2. Estimate Carrier Frequency using FFT ===
window = np.hanning(n)
fft_spectrum = np.abs(np.fft.fft(modulated_signal * window))
freqs = np.fft.fftfreq(n, d=1/sample_rate)
positive_freqs = freqs[:n // 2]
positive_spectrum = fft_spectrum[:n // 2]

# Find two peaks
peak_indices = positive_spectrum.argsort()[-2:]
f1, f2 = sorted(positive_freqs[peak_indices])
estimated_fc = (f1 + f2) / 2

print(f"Estimated Carrier Frequency (mean of peaks): {estimated_fc:.2f} Hz")

# Plot FFT
plt.figure(figsize=(10, 4))
plt.plot(positive_freqs, positive_spectrum)
plt.axvline(estimated_fc, color='r', linestyle='--', label=f'Estimated Fc ≈ {estimated_fc:.1f} Hz')
plt.title("FFT Spectrum of Modulated Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 3. Bandpass Filter ===
def bandpass_filter(signal, fc, bandwidth, fs, order=6):
    nyq = 0.5 * fs
    low = (fc - bandwidth / 2) / nyq
    high = (fc + bandwidth / 2) / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

bandwidth = 1200 # Adjust based on your message bandwidth
filtered_signal = bandpass_filter(modulated_signal, estimated_fc, bandwidth, sample_rate)

# Plot filtered signal
plt.figure(figsize=(10, 3))
plt.plot(time[:5000], filtered_signal[:5000])
plt.title("Bandpass Filtered Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 4. Synchronous Demodulation ===
cos_wave = np.cos(2 * np.pi * estimated_fc * time)
demodulated = filtered_signal * cos_wave

# Plot demodulated signal
plt.figure(figsize=(10, 3))
plt.plot(time[:5000], demodulated[:5000])
plt.title("After Multiplying with Cosine (Pre-Lowpass)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 5. Lowpass Filter (to extract baseband) ===
def lowpass_filter(signal, cutoff, fs, order=6):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='low')
    return filtfilt(b, a, signal)

cutoff_lp = 1000  # Adjust based on your message's pitch
demodulated_audio = lowpass_filter(demodulated, cutoff_lp, sample_rate)

#A highpass to remove noise bw 0 - 200 ish hz lets see
def highpass_filter(signal, cutoff, fs, order=4):
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq
    b, a = butter(order, norm_cutoff, btype='high')
    return filtfilt(b, a, signal)

demodulated_audio = highpass_filter(demodulated_audio, cutoff=250, fs=sample_rate)


# === 6. DC Block and Normalize ===
demodulated_audio -= np.mean(demodulated_audio)
demodulated_audio = demodulated_audio / np.max(np.abs(demodulated_audio))

# === 7. Save Output ===
wavfile.write("recovered_sync_final.wav", sample_rate, (demodulated_audio * 32767).astype(np.int16))

# Plot final signal
plt.figure(figsize=(10, 3))
plt.plot(time[:5000], demodulated_audio[:5000])
plt.title("Recovered Audio After Lowpass Filter")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 8. NEW: FFT of Demodulated Audio ===
n_demod = len(demodulated_audio)
window_demod = np.hanning(n_demod)
fft_demod = np.abs(np.fft.fft(demodulated_audio * window_demod))
freqs_demod = np.fft.fftfreq(n_demod, d=1/sample_rate)
positive_freqs_demod = freqs_demod[:n_demod // 2]
positive_spectrum_demod = fft_demod[:n_demod // 2]

plt.figure(figsize=(10, 4))
plt.plot(positive_freqs_demod, positive_spectrum_demod)
plt.title("FFT of Final Demodulated Audio")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.tight_layout()
plt.show()

print("✅ Final audio saved as 'recovered_sync_final.wav'")
