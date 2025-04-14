import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter


# Extract pre-stimulus data
def extract_tapered_psd(epochs, tmin = -1.0, tmax = 0.0, pad_duration = 1.0):
    pre_stim_data = epochs.copy().crop(tmin=tmin, tmax=tmax).get_data()
    n_epochs, n_channels, n_times = pre_stim_data.shape
    sfreq = epochs.info['sfreq']

    pad_duration = pad_duration  # padding duration in seconds
    n_pad = int(pad_duration * sfreq)
    window = np.hanning(n_times).reshape(1, 1, n_times)
    tapered_data = pre_stim_data * window
    padded_data = np.pad(tapered_data, pad_width=((0, 0), (0, 0), (0, n_pad)), mode='constant')

    # Compute FFT
    freqs = np.fft.rfftfreq(n_times + n_pad, d=1/sfreq)
    fft_data = np.fft.rfft(padded_data, axis=-1)
    psd = np.abs(fft_data)**2
    psd /= ((n_times + n_pad) * np.sum(window[0, 0, :]**2))  # Normalize by duration and window energy
    psd[..., 1:-1] *= 2  # Correct for one-sided FFT

    maxfreq_idx = np.argmin(np.abs(freqs - 30))
    psd = psd[:,:,:maxfreq_idx+1]
    freqs = freqs[:maxfreq_idx+1]
    return psd, freqs


# Function to extract a mask based on the CDF of the power spectrum
def extract_mask(powspect,threshold = .90, plot_cdf = False):
    # Normalize the power spectrum along the time and frequency dimensions
    powspect = np.abs(powspect)
    normalized_powspect = powspect / np.max(powspect, keepdims=True)
    flattened_powspect = normalized_powspect.flatten()
    kmf = KaplanMeierFitter()
    kmf.fit(flattened_powspect, event_observed=np.ones_like(flattened_powspect))

    cdf = 1 - kmf.survival_function_.values
    times = kmf.survival_function_.index.values
    
    # Retrieve the empirical CDF
    cdf_values = 1 - kmf.survival_function_.reindex(flattened_powspect, method="nearest").values.flatten()
    cdf_matrix = cdf_values.reshape(normalized_powspect.shape)

    # Define thresholds based on the reshaped CDF matrices
    threshold = threshold * (cdf_matrix.max() - cdf_matrix.min()) + cdf_matrix.min()
    if plot_cdf:
        plt.plot(times, cdf)
        plt.axhline(y = threshold, color = 'r', linestyle = '--')
        plt.show()
    # Create masks for filtering
    mask = cdf_matrix > threshold
    return mask


# Funciton to perform the MLR
def perform_mlr(data, times, fs, regressor_data, Stim_onset_sec = 0, Win_for_peak_sec=.15,
                with_intercept=True, Half_win_around_peak_sec=0.06, Artifact_offset_sec=0):
    
    """"
    Returns
    -------
    results : dict
        Dictionary containing the following keys:
          - 'fitted_responses': Fitted responses (n_trials × window_length).
          - 'peaks': Single-trial peak values (1D array).
          - 'latencies': Single-trial latency indices (1D array).
          - 'original_peak': ERP peak (scalar) from the regressor signal.
          - 'original_latency': ERP latency (scalar) from the regressor signal.
          - 'coefficients': Regression coefficients (n_trials × n_regressors).
          - 'residuals': Residuals (1D array).
          - 'neg_coeff_trials': List of trial indices with a negative coefficient (for the relevant regressor).
          - 'Win_start': Start index of the data window.
          - 'Win_end': End index of the data window.
    """

    Regressor_ERP = regressor_data.mean(axis=0)
    Peakwin_start_samp = np.argmin(np.abs(times - (Stim_onset_sec+Artifact_offset_sec)))
    Peakwin_end_samp = Peakwin_start_samp + int(Win_for_peak_sec*fs)
    peak_latency_samp = np.argmin(Regressor_ERP[Peakwin_start_samp:Peakwin_end_samp]) + Peakwin_start_samp
    peak_amplitude = np.min(Regressor_ERP[Peakwin_start_samp:Peakwin_end_samp])
    peak_latency_sec = times[peak_latency_samp]

    # Calculate the subject specific ERP peak and latency
    Subject_ERP = data.mean(axis=0)
    Subject_ERP_peak_latency_samp = np.argmin(Subject_ERP[Peakwin_start_samp:Peakwin_end_samp]) + Peakwin_start_samp
    Subject_ERP_peak_latency_sec = times[Subject_ERP_peak_latency_samp]
    Subject_ERP_peak_amplitude = np.min(Subject_ERP[Peakwin_start_samp:Peakwin_end_samp])

    # Slice the data for the selected time window
    data_windowed = data[:, Peakwin_start_samp:Peakwin_end_samp]
    n_timepoints = data_windowed.shape[1]

    # Build the regressors
    if with_intercept:
        regressors = np.zeros((n_timepoints, 3))
        regressors[:, 0] = 1  # intercept column
        regressors[:, 1] = Regressor_ERP[Peakwin_start_samp:Peakwin_end_samp]
        regressors[:, 2] = np.concatenate(([0], np.diff(regressors[:, 1])))
    else:
        regressors = np.zeros((n_timepoints, 2))
        regressors[:, 0] = Regressor_ERP[Peakwin_start_samp:Peakwin_end_samp]
        regressors[:, 1] = np.concatenate(([0], np.diff(regressors[:, 0])))

    # Run MLR for each trial
    n_trials = data_windowed.shape[0]
    n_regressors = regressors.shape[1]
    beta = np.zeros((n_trials, n_regressors))
    residuals = np.zeros((n_trials, 1))
    for trial in range(n_trials):
        y = data_windowed[trial, :]
        beta[trial, :], residuals[trial], _, _ = np.linalg.lstsq(regressors, y, rcond=None)
    fitted_responses = np.dot(beta, regressors.T)
    if fitted_responses.shape != data_windowed.shape:
        raise ValueError("Fitted responses shape mismatch.")

    # Define a window of interest around the ERP latency (±peak_window_sec)
    win_ext = int(Half_win_around_peak_sec * fs)
    temp_peak_latency_samp = np.argmin(Regressor_ERP[Peakwin_start_samp:Peakwin_end_samp])
    win_of_interest = [max(0, temp_peak_latency_samp - win_ext), min(n_timepoints, temp_peak_latency_samp + win_ext)]

    # Initialize arrays for trial-by-trial peak and latency measures.
    peaks = np.zeros((n_trials, 1))
    latencies = np.zeros((n_trials, 1))
    neg_coeff_trials = []
    # Determine which regression coefficient to inspect based on with_intercept.
    coeff_idx = 1 if with_intercept else 0
    for trial in range(n_trials):
        if beta[trial, coeff_idx] < 0:
            neg_coeff_trials.append(trial)
            # If the coefficient is negative, take the maximum in the window.
            peaks[trial, 0] = np.max(fitted_responses[trial, win_of_interest[0]:win_of_interest[1]])
            latencies[trial, 0] = np.argmax(fitted_responses[trial, win_of_interest[0]:win_of_interest[1]]) + win_of_interest[0]
        else:
            peaks[trial, 0] = np.min(fitted_responses[trial, win_of_interest[0]:win_of_interest[1]])
            latencies[trial, 0] = np.argmin(fitted_responses[trial, win_of_interest[0]:win_of_interest[1]]) + win_of_interest[0]

    # Adjust latencies to be relative to the original data timeline
    latencies += Peakwin_start_samp

    results = {
        'fitted_responses': fitted_responses,
        'peaks': peaks.flatten(),
        'latencies': latencies.flatten(),
        'original_peak': Subject_ERP_peak_amplitude,
        'original_latency': Subject_ERP_peak_latency_samp,
        'coefficients': beta,
        'residuals': residuals.flatten(),
        'neg_coeff_trials': neg_coeff_trials,
        'Win_start': Peakwin_start_samp,
        'Win_end': Peakwin_end_samp
    }
    return results




