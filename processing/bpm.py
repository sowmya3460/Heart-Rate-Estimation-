import numpy as np


def calculate_bpm(signal, fps=30):

    if signal is None:
        return 0

    if len(signal) < fps * 5:
        return 0

    signal = np.array(signal)

    signal = signal - np.mean(signal)

    fft = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        d=1/fps
    )

    bpm_freqs = freqs * 60

    valid_idx = np.where(
        (bpm_freqs >= 55) &
        (bpm_freqs <= 110)
    )

    bpm_freqs = bpm_freqs[valid_idx]

    fft = np.abs(fft[valid_idx])

    if len(fft) == 0:
        return 0

    peak_idx = np.argmax(fft)

    bpm = bpm_freqs[peak_idx]

    return int(bpm)