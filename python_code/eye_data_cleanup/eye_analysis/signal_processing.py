"""Signal processing utilities for eye tracking data."""

import numpy as np
from scipy.signal import butter, filtfilt


def apply_butterworth_filter(
    *,
    data: np.ndarray,
    cutoff: float = 5.0,
    fs: float = 30.0,
    order: int = 4
) -> np.ndarray:
    """Apply Butterworth lowpass filter to data.

    Args:
        data: Input data array (n_samples,)
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order

    Returns:
        Filtered data array
    """
    # Remove NaN values for filtering
    valid_mask: np.ndarray = ~np.isnan(data)
    if not np.any(valid_mask):
        return data

    # Design filter
    nyquist: float = 0.5 * fs
    normal_cutoff: float = cutoff / nyquist
    b, a = butter(N=order, Wn=normal_cutoff, btype='low', analog=False)

    # Apply filter only to valid data
    filtered_data: np.ndarray = data.copy()
    if np.sum(valid_mask) > order * 3:  # Need enough points for filtering
        filtered_data[valid_mask] = filtfilt(b=b, a=a, x=data[valid_mask])

    return filtered_data


def remove_nan_values(
    *,
    x_data: np.ndarray,
    y_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaN values from paired X/Y data.

    Args:
        x_data: X coordinate array
        y_data: Y coordinate array

    Returns:
        Tuple of (x_valid, y_valid) with NaN values removed
    """
    valid_mask: np.ndarray = ~(np.isnan(x_data) | np.isnan(y_data))
    return x_data[valid_mask], y_data[valid_mask]
