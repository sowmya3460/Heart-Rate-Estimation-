from scipy.signal import butter, filtfilt
import numpy as np


def bandpass_filter(signal,
                    low=0.8,
                    high=3.0,
                    fs=30,
                    order=3):

    # EMPTY CHECK

    if signal is None:
        return []

    signal = np.array(signal)

    # VERY IMPORTANT FIX

    if len(signal) < 30:
        return signal.tolist()

    nyquist = 0.5 * fs

    low = low / nyquist
    high = high / nyquist

    b, a = butter(
        order,
        [low, high],
        btype='band'
    )

    try:

        filtered = filtfilt(
            b,
            a,
            signal
        )

        return filtered.tolist()

    except:

        return signal.tolist()