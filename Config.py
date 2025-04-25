# This file contains the configuration for for all the functions in the project
import numpy as np

Config_get_snr = {'time_beg': 0, 'time_end': .8}

Get_Foof_config = {'tmin': -.5, 'tmax': 1.5, 'chan_gfar': 128, 'chan_gfas': 104}

Simulate_Background_config = {'noise_level': 0.2, 'freq_range': [1, 125], 'sfreq': 250 }

perform_mlr_config = {
    'Stim_onset_sec_1': 0, 
    'Stim_onset_sec_2': 0.5,
    'Win_for_peak_sec':.25,
    'with_intercept':True, 
    'Half_win_around_peak_sec':0.08, 
    'Artifact_offset_sec':0
    }

signal_sine_wave_config = {
    "Response_Amplitude_scaling1": 5,
    "Response_Amplitude_scaling2": 2.5,
    "Ideal_latency_1": 0.15,
    "Ideal_latency_2": 0.6,
    "Latency_jitter1_scaling": 0.01,
    "Latency_jitter2_scaling": 0.01,
    "Response_Frequency": 3,
    "sigma_rise": 0.015,
    "sigma_fall": 0.15,
    "sigma_rise_fgaussian": 0.005,
    "sigma_fall_fgaussian": 0.2,
    "phase_range1": [np.pi * 1.2, np.pi * 1.6],
    "phase_range2": [np.pi * 1.0, np.pi * 1.4],
    "jitter_scaling_range": [0.9, 1.1],
}

Signal_gaussian_config = {
    "Gating_ratio": 0.5,
    "latency_N1": 0.15,
    "latency_N2": 0.65,
    "sigma1": 0.03,
    "sigma2": 0.07,
    "latency_jitter": 0.015,
    "amplitude_jitter": 0.1,
    "N1_Amplitude": -5
}

combine_config = {
    "SNR_mode": "Emperical",  # or "Custom" - if custom, the specified snr will be used for the simulation
    "Gaussian_noise_level": .5,
    "desired_snr_db": 6,
    "filter_low": 0.3,
    "filter_high": 35,
    "filter_order": 4,
    "get_snr_config": Config_get_snr
    }


Simulate_data_config = {
    "Gating_ratio": 0.3,
    "Gaussian_noise_level": 3,
    "n_datasets": 50
}
