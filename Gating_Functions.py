import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from scipy.signal import stft, istft
import mne
from fooof.sim import gen_group_power_spectra, gen_power_spectrum
from fooof import FOOOF
from scipy.optimize import minimize_scalar
import numpy as np
import os
import scipy.signal as signal


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



def extract_optimal_mask(Input_signal, t, fs, winsize = .25, S1_onset=0, 
                         S2_onset=.5, extract_mask_func = extract_mask, 
                         threshold_drop=0.99, plot_signal = False):

    S1_onset = np.argmin(np.abs(t - S1_onset))
    S2_onset = np.argmin(np.abs(t - S2_onset))
    window_size = int(winsize * fs)

    # Extract peaks and latencies
    S1 = np.min(Input_signal[S1_onset:S1_onset + window_size])
    S2 = np.min(Input_signal[S2_onset:S2_onset + window_size])

    S1_latency = np.argmin(Input_signal[S1_onset:S1_onset + window_size]) + S1_onset
    S2_latency = np.argmin(Input_signal[S2_onset:S2_onset + window_size]) + S2_onset

    if plot_signal:
        plt.plot(t, Input_signal)
        plt.plot(t[S1_latency], S1, 'o', label = 'S1')
        plt.plot(t[S2_latency], S2, 'o', label = 'S2')
        plt.axvline(x = t[S1_onset], color = 'r', linestyle = '--')
        plt.axvline(x = t[S1_onset + window_size], color = 'r', linestyle = '--')
        plt.axvline(x = t[S2_onset], color = 'g', linestyle = '--')
        plt.axvline(x = t[S2_onset + window_size], color = 'g', linestyle = '--')
        plt.legend()
        plt.show()

    f, time_tfr, powspect = stft(Input_signal, fs=fs, nperseg=100, noverlap=99, boundary='zeros')
    time_tfr = np.linspace(-.5, 1.5, powspect.shape[1])
    
    baseline_window = int(np.argmin(np.abs(time_tfr - (-.5))))
    baseline_data = powspect[:,baseline_window:baseline_window + int(fs*.4)].mean(axis=1, keepdims=True)

    # powspect -= baseline_data

    Relative_S1, Relative_S2 = [], []
    Threshold_values = np.linspace(.2, 1.00, 100)

    for thresh in Threshold_values:
        mask = extract_mask_func(powspect, threshold=thresh, plot_cdf=False)

        _, signal_filtered = istft(powspect * mask, fs=fs, nperseg=100, noverlap=99, boundary='zeros')
        t_reconstructed = np.linspace(t[0], t[-1], len(signal_filtered))
        assert signal_filtered.shape == Input_signal.shape

        base_start = np.argmin(np.abs(t - (-.5)))
        base_end = np.argmin(np.abs(t - (0)))
        signal_filtered -= np.mean(signal_filtered[base_start:base_end])

        # Get relative amplitude
        Relative_S1.append(signal_filtered[S1_latency] / S1)
        Relative_S2.append(signal_filtered[S2_latency] / S2)

    below_thresh_S1 = np.where(np.array(Relative_S1) < threshold_drop * Relative_S1[0])[0]
    below_thresh_S2 = np.where(np.array(Relative_S2) < threshold_drop * Relative_S2[0])[0]

    if below_thresh_S1.size > 0:
        Threshold_S1 = Threshold_values[below_thresh_S1[0]]
    else:
        Threshold_S1 = Threshold_values[-1] 

    if below_thresh_S2.size > 0:
        Threshold_S2 = Threshold_values[below_thresh_S2[0]]
    else:
        Threshold_S2 = Threshold_values[-1]

    Final_Threshold = np.min([Threshold_S1, Threshold_S2])

    mask = extract_mask_func(powspect, threshold=Final_Threshold, plot_cdf=False)

    return mask, Final_Threshold, Relative_S1, Relative_S2, powspect


# Funciton to perform the MLR
def perform_mlr(data, times, fs, regressor_data, Stimulus = 1, config = None):
    if config is None:
        from Config import perform_mlr_config
        config = perform_mlr_config
    else:
        config = config

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
    if Stimulus == 1:
        Stim_onset_sec = config['Stim_onset_sec_1']
    elif Stimulus == 2:
        Stim_onset_sec = config['Stim_onset_sec_2']
    Win_for_peak_sec = config['Win_for_peak_sec']
    with_intercept = config['with_intercept']
    Half_win_around_peak_sec = config['Half_win_around_peak_sec']
    Artifact_offset_sec = config['Artifact_offset_sec']
    

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




# Simulation functions
def Get_Fooof_Parameters(index, Path, config = None):
    if config is None:
        from Config import Get_Foof_config
        config = Get_Foof_config
    tmin = config['tmin']
    tmax = config['tmax']
    ERP_files = [f for f in os.listdir(Path) if f.endswith('.set') and not f.startswith('.')]
    name = ERP_files[index]
    if 'GFAS' in name:
         channel2xtract = config['chan_gfas']
    elif 'GFAR' in name:
         channel2xtract = config['chan_gfar']
    epochs = mne.read_epochs_eeglab(Path + ERP_files[index])
    epochs = epochs.crop(tmin = tmin, tmax = tmax)
    evoked = epochs.average().get_data(picks = channel2xtract).squeeze()
    emperical_snr = get_snr(evoked, epochs.times)

    # Fit the foof parameters to the background noise
    aperiodic_params = []
    periodic_params = []
    spectra, freqs = extract_tapered_psd(epochs, tmin=tmin, tmax=tmax, pad_duration=0)
    fm = FOOOF(peak_width_limits=[1, 15], max_n_peaks=3, aperiodic_mode='fixed')
    freq_range = [1, 30]
    for i in range(spectra.shape[0]):
        fm.fit(freqs, spectra[i,channel2xtract,:], freq_range)
        aperiodic_params.append(fm.aperiodic_params_)
        periodic_params.append(fm.peak_params_)
    return aperiodic_params, periodic_params, emperical_snr, epochs


def Simulate_Background_Oscillations(n_trials, n_times, aperiodic_params, periodic_params, config = None):
    if config is None:
        from Config import Simulate_Background_config
        config = Simulate_Background_config
    Sim_sig_Background = np.zeros((n_trials, n_times)) 
    sfreq = config['sfreq']
    noise_level = config['noise_level']
    freq_range = config['freq_range']
    print(len(aperiodic_params))

    for ind in range(n_trials):

        freqs, powers, sim_params = gen_power_spectrum(
            [1, sfreq // 2], aperiodic_params[ind], periodic_params[ind].flatten().tolist(), 
            nlv=noise_level, freq_res=sfreq / n_times, return_params=True
        )

        random_phases = np.random.uniform(0, 2 * np.pi, len(powers))
        magnitudes = np.sqrt(powers)
        complex_spectrum = magnitudes * np.exp(1j * random_phases)
        
        full_spectrum = np.concatenate([complex_spectrum, complex_spectrum[-2:0:-1].conj()]) 

        Temp_signal = np.fft.irfft(full_spectrum, n=n_times)  # Ensure length matches n_times
        Sim_sig_Background[ind, :] = Temp_signal
        
    return Sim_sig_Background


def Simulate_Signal_Sine_Wave(n_trials, n_times, times, config = None):
    if config is None:
        from Config import signal_sine_wave_config
        config = signal_sine_wave_config
    SimulatedData = np.zeros((n_trials, n_times))

    for i in range(n_trials):
        # Extract parameters
        lat1 = config["Ideal_latency_1"] + config["Latency_jitter1_scaling"] * np.random.randn()
        lat2 = config["Ideal_latency_2"] + config["Latency_jitter2_scaling"] * np.random.randn()
        freq_vec = np.ones(n_times) * config["Response_Frequency"]

        # Frequency Gaussian
        fgauss1 = np.where(
            times < lat1,
            -.9 * np.exp(-((times - lat1)**2) / (2 * config["sigma_rise_fgaussian"]**2)) + 1,
            -.9 * np.exp(-((times - lat1)**2) / (2 * config["sigma_fall_fgaussian"]**2)) + 1
        )
        fgauss2 = np.where(
            times < lat2,
            -.9 * np.exp(-((times - lat2)**2) / (2 * config["sigma_rise_fgaussian"]**2)) + 1,
            -.9 * np.exp(-((times - lat2)**2) / (2 * config["sigma_fall_fgaussian"]**2)) + 1
        )
        freq_vec1 = freq_vec * fgauss1
        freq_vec2 = freq_vec * fgauss2

        # Amplitude Gaussian taper
        asym1 = np.where(
            times < lat1,
            np.exp(-((times - lat1)**2) / (2 * config["sigma_rise"]**2)),
            np.exp(-((times - lat1)**2) / (2 * config["sigma_fall"]**2))
        )
        asym2 = np.where(
            times < lat2,
            np.exp(-((times - lat2)**2) / (2 * config["sigma_rise"]**2)),
            np.exp(-((times - lat2)**2) / (2 * config["sigma_fall"]**2))
        )

        # Sinusoidal responses
        phase1 = np.random.uniform(*config["phase_range1"])
        phase2 = np.random.uniform(*config["phase_range2"])
        R1 = config["Response_Amplitude_scaling1"] * np.sin(2 * np.pi * freq_vec1 * times + phase1) * asym1
        R2 = config["Response_Amplitude_scaling2"] * np.sin(2 * np.pi * freq_vec2 * times + phase2) * asym2

        SimulatedData[i, :] = R1 + R2

    jitter = np.random.uniform(*config["jitter_scaling_range"], size=(SimulatedData.shape[0], 1))
    SimulatedData *= jitter

    return SimulatedData


def Simulate_Signal_from_Gaussians(n_trials, n_times, times, config = None):
    if config is None:
        from Config import Signal_gaussian_config
        config = Signal_gaussian_config
    # Unpack config
    A1 = -1
    A2 = A1 * config["Gating_ratio"]
    latency_P1 = config["latency_N1"] + 0.15
    latency_P2 = config["latency_N2"] + 0.15

    simulated_trials = np.zeros((n_trials, n_times))
    
    for trial in range(n_trials):
        # Jittered latencies
        trial_latency_N1 = config["latency_N1"] + np.random.randn() * config["latency_jitter"]
        trial_latency_N2 = config["latency_N2"] + np.random.randn() * config["latency_jitter"]
        trial_latency_P1 = latency_P1 + np.random.randn() * config["latency_jitter"]
        trial_latency_P2 = latency_P2 + np.random.randn() * config["latency_jitter"]

        # Jittered amplitudes
        trial_amplitude_N1 = A1 + np.random.randn() * config["amplitude_jitter"]
        trial_amplitude_N2 = A2 + np.random.randn() * config["amplitude_jitter"]
        trial_amplitude_P1 = -A1 + np.random.randn() * config["amplitude_jitter"]
        trial_amplitude_P2 = -A2 + np.random.randn() * config["amplitude_jitter"]

        # ERP components
        erp_negative1 = trial_amplitude_N1 * np.exp(-((times - trial_latency_N1) ** 2) / (2 * config["sigma1"] ** 2))
        erp_positive1 = trial_amplitude_P1 * np.exp(-((times - trial_latency_P1) ** 2) / (2 * config["sigma2"] ** 2))

        erp_negative2 = trial_amplitude_N2 * np.exp(-((times - trial_latency_N2) ** 2) / (2 * config["sigma1"] ** 2))
        erp_positive2 = trial_amplitude_P2 * np.exp(-((times - trial_latency_P2) ** 2) / (2 * config["sigma2"] ** 2))

        # Combine all
        simulated_trials[trial, :] = erp_negative1 + erp_positive1 + erp_negative2 + erp_positive2

    # Scale to desired N1 amplitude
    simulated_erp = simulated_trials.mean(axis=0)
    min_peak = np.min(simulated_erp[(times > 0) & (times < 0.25)])
    scaling_factor = config["N1_Amplitude"] / min_peak
    simulated_trials *= scaling_factor

    return simulated_trials


def Combine_and_Scale_Signal(Simulated_Signal, Sim_sig_Background, times, sfreq, emperical_snr, config = None):
    if config is None:
        from Config import combine_config
        config = combine_config

    SNR_mode = config["SNR_mode"]
    Gaussian_noise_level = config["Gaussian_noise_level"]
    desired_snr_db = emperical_snr if SNR_mode == 'Emperical' else config["desired_snr_db"]

    # Normalize background
    Sim_sig_Background = (Sim_sig_Background - np.mean(Sim_sig_Background, axis=1, keepdims=True)) / \
                         np.std(Sim_sig_Background, axis=1, keepdims=True)

    # Func for SNR difference
    def snr_difference(alpha):
        background_ERP = Sim_sig_Background.mean(axis=0)
        Signal_ERP = Simulated_Signal.mean(axis=0)
        Full_ERP = (alpha * background_ERP) + Signal_ERP
        return (get_snr(Full_ERP, times) - desired_snr_db) ** 2

    # Optimize SNR scaling
    result = minimize_scalar(snr_difference, bounds=(0.01, 100), method='bounded')
    alpha_opt = result.x
    Scaled_background = alpha_opt * Sim_sig_Background
    final_signal = Simulated_Signal + Scaled_background

    # Add Gaussian noise
    Gaussian_noise_std = Gaussian_noise_level * np.std(final_signal, axis=1, keepdims=True)
    Gaussian_noise = np.random.normal(0, Gaussian_noise_std, size=final_signal.shape)
    final_signal += Gaussian_noise

    # Bandpass filtering
    nyq_freq = sfreq / 2
    low = config["filter_low"] / nyq_freq
    high = config["filter_high"] / nyq_freq
    b, a = signal.butter(config["filter_order"], [low, high], btype='band')
    final_signal = signal.filtfilt(b, a, final_signal)

    # Baseline correction
    final_signal = final_signal - final_signal[:, (times < 0)].mean(axis=1, keepdims=True)

    # Final SNR
    snr_empirical_db = get_snr(final_signal.mean(axis=0), times, config['get_snr_config'])

    return final_signal, snr_empirical_db


def Simulate_Datasets(Path, times, Simulate_data_config = None):
    if Simulate_data_config is None:
        from Config import Simulate_data_config
        Simulate_data_config = Simulate_data_config
        
    Gating_ratio = Simulate_data_config["Gating_ratio"]
    Gaussian_noise_level = Simulate_data_config["Gaussian_noise_level"]
    n_datasets = Simulate_data_config["n_datasets"]
    Grand_Average_ERP = np.zeros(len(times))
    All_Simulated_Data = {}
    Extracted = {}
    index = 0
    for ind in range(100):
        print(f'The index is {index}')
        try:
            aperiodic_params, periodic_params, emperical_snr, epochs = Get_Fooof_Parameters(index, Path, config = Get_Foof_config)
        except:
            continue
        epochs = epochs.crop(tmin = -.5, tmax = 1.5)
        info = epochs.info
        sfreq = epochs.info['sfreq']  
        tmin, tmax = -0.5, 1.5
        times = epochs.times
        n_times = len(times)
        n_trials = len(aperiodic_params)
        if index not in Extracted:
            Extracted[index] = {} 

        Sim_sig_Background = Simulate_Background_Oscillations(n_trials, n_times, aperiodic_params, 
                                                      periodic_params, config = Simulate_Background_config)

        Signal_gaussian_config['Gating_ratio'] = Gating_ratio
        Simulated_Signal = Simulate_Signal_from_Gaussians(n_trials, n_times, times, config=Signal_gaussian_config)

        time_beg_n1 = 0
        time_end_n1 = 0.4
        time_beg_n2 = 0.5
        time_end_n2 = 0.9
        epsilon = 1e-10
        # Store the amplitudes of each trial
        N1_amplitudes, N1_latencies = extract_single_trial_peaks(Simulated_Signal, times, stim_onset = 0, sfreq = sfreq)
        Extracted[index]['Pure_N1_amplitudes'] = N1_amplitudes
        Extracted[index]['Pure_N1_latencies'] = N1_latencies

        N2_amplitudes, N2_latencies = extract_single_trial_peaks(Simulated_Signal, times, stim_onset = .5, sfreq = sfreq)
        Extracted[index]['Pure_N2_amplitudes'] = N2_amplitudes
        Extracted[index]['Pure_N2_latencies'] = N2_latencies
        Extracted[index]['Pure_Simulated_gating'] = Extracted[index]['Pure_N2_amplitudes'] / Extracted[index]['Pure_N1_amplitudes']
        
        # Compute ERP (average across trials)
        erp = Simulated_Signal.mean(axis=0)
        N1_erp_amp, N1_erp_latency = extract_ERP_peak(erp, times, stim_onset=0, sfreq=sfreq)
        Extracted[index]['Pure_N1_ERP_amplitude'] = N1_erp_amp
        Extracted[index]['Pure_N1_ERP_latency'] = N1_erp_latency

        N2_erp_amp, N2_erp_latency = extract_ERP_peak(erp, times, stim_onset=0.5, sfreq=sfreq)
        Extracted[index]['Pure_N2_ERP_amplitude'] = N2_erp_amp
        Extracted[index]['Pure_N2_ERP_latency'] = N2_erp_latency
        Extracted[index]['Pure_Simulated_ERP_gating'] = N2_erp_amp / (N1_erp_amp + epsilon)

                
        combine_config['Gaussian_noise_level'] = Gaussian_noise_level

        Final_Signal, Final_SNR = Combine_and_Scale_Signal(Simulated_Signal, Sim_sig_Background, times, 
                                                        sfreq, emperical_snr, config=combine_config)
                

        All_Simulated_Data[index] = Final_Signal
        Grand_Average_ERP += Final_Signal.mean(axis=0)

        # Save the final SNR
        Extracted[index]['Emperical_SNR'] = Final_SNR

        impure_erp = Final_Signal.mean(axis=0)
        imp_N1_amp, imp_N1_latency = extract_ERP_peak(impure_erp, times, stim_onset=0, sfreq=sfreq)
        Extracted[index]['Impure_N1_ERP_amplitude'] = imp_N1_amp
        Extracted[index]['Impure_N1_ERP_latency'] = imp_N1_latency
        imp_N2_amp, imp_N2_latency = extract_ERP_peak(impure_erp, times, stim_onset=0.5, sfreq=sfreq)
        Extracted[index]['Impure_N2_ERP_amplitude'] = imp_N2_amp
        Extracted[index]['Impure_N2_ERP_latency'] = imp_N2_latency
        Extracted[index]['Impure_ERP_gating'] = imp_N2_amp / (imp_N1_amp + epsilon)
                
        # Extract single-trial N1 peaks
        Impure_N1_amplitudes, Impure_N1_latencies = extract_single_trial_peaks(Final_Signal, times, stim_onset=0, sfreq=sfreq)
        Extracted[index]['Impure_N1_single_trial_amplitudes'] = Impure_N1_amplitudes
        Extracted[index]['Impure_N1_single_trial_latencies'] = Impure_N1_latencies
        # Extract single-trial N2 peaks
        Impure_N2_amplitudes, Impure_N2_latencies = extract_single_trial_peaks(Final_Signal, times, stim_onset=0.5, sfreq=sfreq)
        Extracted[index]['Impure_N2_single_trial_amplitudes'] = Impure_N2_amplitudes
        Extracted[index]['Impure_N2_single_trial_latencies'] = Impure_N2_latencies
        Extracted[index]['Impure_single_trial_gating'] = Impure_N2_amplitudes / (Impure_N1_amplitudes + epsilon)

        Simulated_TFR_MNE = compute_MNE_tfr(Final_Signal, sfreq = sfreq, tmin = -.5, freqs=np.linspace(0.1, 35, 35), baseline=(None, -0.2), ch_name='EEG 001')

        # # Extract the gating based on the MNE TFR using the delta band
        delta_gating = extract_tfr_band_gating(Simulated_TFR_MNE, band_name = 'delta', t1 = [time_beg_n1, time_end_n1], t2 = [time_beg_n2, time_end_n2], picks='EEG 001', epsilon=1e-10)
        Extracted[index].update(delta_gating)

        # # Extract the gating based on the MNE TFR using the theta band
        theta_gating = extract_tfr_band_gating(Simulated_TFR_MNE, band_name = 'theta', t1 = [time_beg_n1, time_end_n1], t2 = [time_beg_n2, time_end_n2], picks='EEG 001', epsilon=1e-10)
        Extracted[index].update(theta_gating)

        index += 1
        if index == n_datasets:
            break

    Grand_Average_ERP /= index
    return All_Simulated_Data, Extracted, Grand_Average_ERP


def Apply_WF_and_MLR(All_Simulated_Data, times, sfreq, mask, plot_sample = False, Intercept = True):
    from Config import perform_mlr_config
    perform_mlr_config['with_intercept'] = Intercept
    Extracted_WF_MLR = {}

    # Loop over the simulated datasets using the keys
    for key in All_Simulated_Data.keys():
        if key not in Extracted_WF_MLR:
            Extracted_WF_MLR[key] = {}
        Final_Signal = All_Simulated_Data[key]
        f, TFR_time, Simulated_TFR = stft(Final_Signal, fs=sfreq, nperseg=100, noverlap=99, boundary='zeros')
        TFR_time = np.linspace(-.5, 1.5, Simulated_TFR.shape[2])
        Zxx_filtered = Simulated_TFR * mask
        _, Final_signal_filtered = istft(Zxx_filtered, fs=sfreq, nperseg=100, noverlap=99, boundary='zeros')
        assert Final_signal_filtered.shape == Final_Signal.shape, 'The shape of the filtered signal is not the same as the original signal'
        base_start = np.argmin(np.abs(times - (-.5)))
        base_end = np.argmin(np.abs(times - (0)))
        Final_signal_filtered -= np.mean(Final_signal_filtered[base_start:base_end])

        Extracted_WF_MLR[key]['Filtered_Signal'] = Final_signal_filtered

        MLR_results_S1 = perform_mlr(Final_signal_filtered, times, sfreq, Final_signal_filtered, Stimulus = 1, config = None)

        Extracted_WF_MLR[key]['MLR_Results_S1'] = MLR_results_S1

        if plot_sample:
            trial = np.random.randint(0, Final_Signal.shape[0])
            plt.plot(times, Final_Signal.mean(axis=0))
            plt.plot(times[MLR_results_S1['Win_start']:MLR_results_S1['Win_end']], MLR_results_S1['fitted_responses'].mean(axis=0))
            plt.plot(times[int(MLR_results_S1['latencies'].mean())], MLR_results_S1['peaks'].mean(), 'o', label='Fitted Response')
            plt.plot(times[int(MLR_results_S1['original_latency'])], MLR_results_S1['original_peak'], 'o', label='Original Response')
            plt.axvline(x = 0, color = 'r', label = 'Stimulus Onset')
            plt.legend()
            plt.show()

        MLR_results_S2 = perform_mlr(Final_signal_filtered, times, sfreq, Final_signal_filtered, Stimulus = 2, config = None)
        
        Extracted_WF_MLR[key]['MLR_Results_S2'] = MLR_results_S2

        if plot_sample:
            trial = np.random.randint(0, Final_Signal.shape[0])
            plt.plot(times, Final_Signal.mean(axis=0))
            plt.plot(times[MLR_results_S2['Win_start']:MLR_results_S2['Win_end']], MLR_results_S2['fitted_responses'].mean(axis=0))
            plt.plot(times[int(MLR_results_S2['latencies'].mean())], MLR_results_S2['peaks'].mean(), 'o', label='Fitted Response')
            plt.plot(times[int(MLR_results_S2['original_latency'])], MLR_results_S2['original_peak'], 'o', label='Original Response')
            plt.axvline(x = 0, color = 'r', label = 'Stimulus Onset')
            plt.legend()
            plt.show()

        # Calculate the gating based on the MLR results
        time_beg_n1 = 0
        time_end_n1 = 0.4
        time_beg_n2 = 0.5
        time_end_n2 = 0.9
        epsilon = 1e-10
        MLR_results_gating = MLR_results_S2['peaks'] / (MLR_results_S1['peaks'] + epsilon)
        Extracted_WF_MLR[key]['MLR_single_trial_gating'] = MLR_results_gating
        Extracted_WF_MLR[key]['MLR_ERP_gating'] = MLR_results_S2['original_peak'] / (MLR_results_S1['original_peak'] + epsilon)

        Simulated_TFR_MNE = compute_MNE_tfr(Final_signal_filtered, sfreq = 250, tmin = -.5, freqs=np.linspace(0.1, 35, 35), baseline=(None, -0.2), ch_name='EEG 001')

        delta_results = extract_tfr_band_gating(Simulated_TFR_MNE, band_name = 'delta', t1 = [time_beg_n1, time_end_n1], t2 = [time_beg_n2, time_end_n2], picks='EEG 001', epsilon=1e-10)
        Extracted_WF_MLR[key].update(delta_results)

        theta_results = extract_tfr_band_gating(Simulated_TFR_MNE, band_name = 'theta',  t1 = [time_beg_n1, time_end_n1], t2 = [time_beg_n2, time_end_n2], picks='EEG 001', epsilon=1e-10)
        Extracted_WF_MLR[key].update(theta_results)

    return Extracted_WF_MLR





# Helper functions for feature extraction
def extract_single_trial_peaks(signal, times, stim_onset = 0, sfreq = 250, window_ms=400):
    """
    Extracts the amplitude and latency of single-trial ERP peaks (e.g., N1, N2) 
    in a specified window following stimulus onset.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_trials, n_times). The EEG signal for each trial.
    times : np.ndarray
        Time vector corresponding to signal shape.
    stim_onset : float
        Stimulus onset time in seconds (e.g., 0 or 0.5).
    window_ms : float
        Size of the search window in milliseconds (default: 400).

    Returns
    -------
    peak_amplitudes : np.ndarray
        1D array of shape (n_trials,) with peak amplitudes per trial.
    peak_latencies : np.ndarray
        1D array of shape (n_trials,) with peak latencies per trial (in seconds).
    """
    assert signal.ndim == 2, "Expected 2D array for signal"

    # Convert window to seconds
    window_sec = window_ms / sfreq
    time_mask = (times >= stim_onset) & (times <= stim_onset + window_sec)

    # Check the mask is valid
    if not np.any(time_mask):
        raise ValueError("No time points found in the specified window.")

    # Apply mask to signal
    masked_signal = signal[:, time_mask]

    # Extract peaks
    peak_amplitudes = masked_signal.min(axis=1)
    peak_indices = np.argmin(masked_signal, axis=1)
    

    # Extract corresponding latencies
    time_window = times[time_mask]
    peak_latencies = time_window[peak_indices]

    return peak_amplitudes, peak_latencies

def get_snr(data, times, Config_get_snr = None):
    if Config_get_snr is None:
        from Config import Config_get_snr
        Config_get_snr = Config_get_snr

    time_beg = Config_get_snr['time_beg']
    time_end = Config_get_snr['time_end']
    signal_power = np.var(data[(times > time_beg) & (times < time_end)])
    noise_power = np.var(data[(times < 0)])  # Variance across trials & time
    snr_db = 10 * np.log10(signal_power / noise_power) # Compute SNR for each channel
    return snr_db

def extract_ERP_peak(signal, times, stim_onset = 0, sfreq = 250, window_ms=400):
    """
    Extracts the amplitude and latency of ERP peaks 
    in a specified window following stimulus onset.

    Parameters
    ----------
    signal : np.ndarray
        Shape (n_trials, n_times). The EEG signal for each trial.
    times : np.ndarray
        Time vector corresponding to signal shape.
    stim_onset : float
        Stimulus onset time in seconds (e.g., 0 or 0.5).
    window_ms : float
        Size of the search window in milliseconds (default: 400).

    Returns
    -------
    peak_amplitude : np.ndarray
        Scalar peak amplitude for the ERP.
    peak_latencies : np.ndarray
        Scalar latency of the peak (in seconds).
    """
    window_sec = window_ms / sfreq
    time_mask = (times >= stim_onset) & (times <= stim_onset + window_sec)

    if not np.any(time_mask):
        raise ValueError("No time points found in the specified window.")

    masked_signal = signal[time_mask]

    peak_amplitude = masked_signal.min()
    peak_index = np.argmin(masked_signal)
    
    time_window = times[time_mask]
    peak_latency = time_window[peak_index]

    return peak_amplitude, peak_latency

def compute_MNE_tfr(data, sfreq = 250, tmin = -.5, 
                    freqs=np.linspace(0.1, 35, 35), 
                    baseline=(None, -0.2), ch_name='EEG 001'):
    """
    Computes the Morlet wavelet time-frequency representation (TFR) with baseline correction.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_trials, n_times) EEG data.
    sfreq : float
        Sampling frequency in Hz.
    tmin : float
        Start time of epochs (relative to event onset).
    freqs : array-like
        Frequencies of interest for Morlet wavelets (default: 0.1 to 35 Hz).
    baseline : tuple
        Baseline window in seconds (e.g., (-0.2, 0)).
    ch_name : str
        Name of the EEG channel.

    Returns
    -------
    tfr : mne.time_frequency.EpochsTFR
        The computed and baseline-corrected TFR object.
    """
    data_mne = data[:, np.newaxis, :]
    info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(data_mne, info=info, tmin=tmin)

    n_cycles = freqs / 2.0
    tfr = epochs.compute_tfr(method='morlet', freqs=freqs, n_cycles=n_cycles)
    tfr.apply_baseline(baseline=baseline, mode='ratio')

    return tfr

def extract_tfr_band_gating(TFR, band_name, t1 = [0, .4], t2 = [.5, .9], picks='EEG 001', epsilon=1e-10):
    """
    Extract single-trial and average-based gating values for a frequency band from MNE TFR data.

    Parameters
    ----------
    TFR : mne.EpochsTFR
        The TFR object (e.g., computed with Morlet wavelets).
    band_name : str
        Frequency band name: 'delta' or 'theta'.
    t1, t2 : tuple of float
        Time windows for S1 and S2 (e.g., (0, 0.4) and (0.5, 0.9)).
    tmin, tmax : float
        Total time window (not used directly here, kept for compatibility).
    picks : str
        Channel name to extract.
    epsilon : float
        Small constant to avoid division by zero.

    Returns
    -------
    results : dict
        Dictionary with extracted TFR metrics.
    """
    if band_name.upper() == 'DELTA':
        fmin, fmax = 0.2, 5
    elif band_name.upper() == 'THETA':
        fmin, fmax = 4, 8
    else:
        raise ValueError(f"Unsupported band name: {band_name}. Use 'delta' or 'theta'.")

    results = {}

    # Single-trial TFR
    results[f'MNE_TFR_S1_{band_name}'] = TFR.get_data(fmin=fmin, fmax=fmax, tmin=t1[0], tmax=t1[1], picks=picks).mean(axis=(-1, -2))
    results[f'MNE_TFR_S2_{band_name}'] = TFR.get_data(fmin=fmin, fmax=fmax, tmin=t2[0], tmax=t2[1], picks=picks).mean(axis=(-1, -2))
    results[f'MNE_TFR_gating_{band_name}'] = results[f'MNE_TFR_S2_{band_name}'] / (results[f'MNE_TFR_S1_{band_name}'] + epsilon)

    # Average TFR
    TFR_avg = TFR.average()
    results[f'MNE_TFR_S1_average_{band_name}'] = TFR_avg.get_data(fmin=fmin, fmax=fmax, tmin=t1[0], tmax=t1[1], picks=picks).mean(axis=(-1, -2)).item()
    results[f'MNE_TFR_S2_average_{band_name}'] = TFR_avg.get_data(fmin=fmin, fmax=fmax, tmin=t2[0], tmax=t2[1], picks=picks).mean(axis=(-1, -2)).item()
    results[f'MNE_average_TFR_gating_{band_name}'] = results[f'MNE_TFR_S2_average_{band_name}'] / (results[f'MNE_TFR_S1_average_{band_name}'] + epsilon)

    return results









# For plotting MLR results
def plot_MLR_results(p, Extracted_WF_MLR, times, All_Simulated_Data):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax = axs[0]
    ax.plot(times, All_Simulated_Data[p].mean(axis=0), 'k', linewidth = .5)
    ax.plot(times[Extracted_WF_MLR[p]['MLR_Results_S1']['Win_start']:Extracted_WF_MLR[p]['MLR_Results_S1']['Win_end']], Extracted_WF_MLR[p]['MLR_Results_S1']['fitted_responses'].mean(axis=0), 'r', linewidth = 2, label = 'Fitted Peak')
    ax.plot(times[Extracted_WF_MLR[p]['MLR_Results_S2']['Win_start']:Extracted_WF_MLR[p]['MLR_Results_S2']['Win_end']], Extracted_WF_MLR[p]['MLR_Results_S2']['fitted_responses'].mean(axis=0), 'g', linewidth = 2, label = 'Fitted Peak')
    ax.plot(times[int(Extracted_WF_MLR[p]['MLR_Results_S1']['original_latency'])], Extracted_WF_MLR[p]['MLR_Results_S1']['original_peak'], 'ko', label='Empirical N1')
    ax.plot(times[int(Extracted_WF_MLR[p]['MLR_Results_S2']['original_latency'])], Extracted_WF_MLR[p]['MLR_Results_S2']['original_peak'], 'ko', label='Emperical N2')
    ax.plot(times[int(Extracted_WF_MLR[p]['MLR_Results_S1']['latencies'].mean())], Extracted_WF_MLR[p]['MLR_Results_S1']['peaks'].mean(), 'ro', label='Average of Fitted N1')
    ax.plot(times[int(Extracted_WF_MLR[p]['MLR_Results_S2']['latencies'].mean())], Extracted_WF_MLR[p]['MLR_Results_S2']['peaks'].mean(), 'ro', label='Average of Fitted N2')
    ax.axvline(x = 0, color = 'k', linewidth = .5)
    ax.set_title(f'Participant {p} ERP', fontsize = 16)
    ax.legend()

    trial = np.random.randint(0, All_Simulated_Data[p].shape[0])
    ax = axs[1]
    ax.plot(times, All_Simulated_Data[p][trial, :], 'k', linewidth = .5)
    ax.plot(times[Extracted_WF_MLR[p]['MLR_Results_S1']['Win_start']:Extracted_WF_MLR[p]['MLR_Results_S1']['Win_end']], Extracted_WF_MLR[p]['MLR_Results_S1']['fitted_responses'][trial, :], 'r', linewidth = 2)
    ax.plot(times[Extracted_WF_MLR[p]['MLR_Results_S2']['Win_start']:Extracted_WF_MLR[p]['MLR_Results_S2']['Win_end']], Extracted_WF_MLR[p]['MLR_Results_S2']['fitted_responses'][trial, :], 'g', linewidth = 2)
    ax.plot(times[int(Extracted_WF_MLR[p]['MLR_Results_S1']['latencies'][trial])], Extracted_WF_MLR[p]['MLR_Results_S1']['peaks'][trial], 'ko', label='Fitted N1')
    ax.plot(times[int(Extracted_WF_MLR[p]['MLR_Results_S2']['latencies'][trial])], Extracted_WF_MLR[p]['MLR_Results_S2']['peaks'][trial], 'ko', label='Fitted N2')
    ax.axvline(x = 0, color = 'k', linewidth = .5)
    ax.set_title(f'Trial {trial}', fontsize = 16)
    ax.legend()






# Extract all measures of gating from the simulated data
def extract_gating_measures(participant_id, extracted_metrics: dict, extracted_wf_mlr: dict):
    all_keys = list(extracted_metrics.keys())
    all_keys_wf = list(extracted_wf_mlr.keys())

    # Define gating-related keys
    trial_gating_keys = [key for key in all_keys if "gating" in key and "ERP" not in key and "average" not in key]
    erp_gating_keys = [key for key in all_keys if "gating" in key and ("ERP" in key or "average" in key)]

    trial_gating_keys_wf = [key + '_WF' for key in all_keys_wf if "gating" in key and "ERP" not in key and "average" not in key]
    erp_gating_keys_wf = [key + '_WF' for key in all_keys_wf if "gating" in key and ("ERP" in key or "average" in key)]

    # Extract trial-level data
    trial_data = {}
    for key in trial_gating_keys:
        trial_data[key] = extracted_metrics[key]
    for key in trial_gating_keys_wf:
        trial_data[key] = extracted_wf_mlr[key[:-3]]

    for key, val in trial_data.items():
        if isinstance(val, np.ndarray) and val.ndim > 1:
            trial_data[key] = val.flatten()

    trial_df = pd.DataFrame(trial_data)
    trial_df["Dataset"] = participant_id

    # Extract ERP-level data
    erp_data = {}
    for key in erp_gating_keys:
        erp_data[key] = extracted_metrics[key]
    for key in erp_gating_keys_wf:
        erp_data[key] = extracted_wf_mlr[key[:-3]]
    erp_data["Dataset"] = participant_id

    erp_df = pd.DataFrame([erp_data])

    return trial_df, erp_df


# Extract amplitudes and power from the simulated data
def extract_amplitudes_and_power(participant_id, extracted_metrics: dict, extracted_wf_mlr: dict):
    all_keys = list(extracted_metrics.keys())
    all_keys_wf = list(extracted_wf_mlr.keys())

    # Trial-level and ERP-level amplitude keys
    amplitude_trial_keys = [key for key in all_keys if "amplitude" in key and "ERP" not in key]
    amplitude_erp_keys = [key for key in all_keys if "amplitude" in key and "ERP" in key]

    # Trial-level and ERP-level power keys
    power_trial_keys = [key for key in all_keys if "MNE_TFR_S" in key and "average" not in key]
    power_erp_keys = [key for key in all_keys if "MNE_TFR_S" in key and "average" in key]

    Power_Trial_keys_wf = [key + '_WF' for key in all_keys_wf if "MNE_TFR_S" in key and "average" not in key]
    Power_ERP_keys_wf = [key + '_WF' for key in all_keys_wf if "MNE_TFR_S" in key and "average" in key]

    # Extract trial-level data
    trial_data = {}
    for key in amplitude_trial_keys + power_trial_keys:
        trial_data[key] = extracted_metrics[key]
    for key in ['MLR_Results_S1', 'MLR_Results_S2']:
        trial_data[f'{key}_peaks'] = extracted_wf_mlr[key]['peaks']
    for key in Power_Trial_keys_wf:
        trial_data[key] = extracted_wf_mlr[key[:-3]]
    
    # Flatten anything that's more than 1D
    for key, val in trial_data.items():
        if isinstance(val, np.ndarray) and val.ndim > 1:
            trial_data[key] = val.flatten()

    trial_df = pd.DataFrame(trial_data)
    trial_df["Dataset"] = participant_id

    # Extract ERP-level data
    erp_data = {}
    for key in amplitude_erp_keys + power_erp_keys:
        erp_data[key] = extracted_metrics[key]
    for key in Power_ERP_keys_wf:
        erp_data[key] = extracted_wf_mlr[key[:-3]]
    erp_data['S1_ERP_peak_MLR'] = extracted_wf_mlr['MLR_Results_S1']['original_peak']
    erp_data['S2_ERP_peak_MLR'] = extracted_wf_mlr['MLR_Results_S2']['original_peak']
    erp_data["Dataset"] = participant_id

    erp_df = pd.DataFrame([erp_data])

    return trial_df, erp_df
