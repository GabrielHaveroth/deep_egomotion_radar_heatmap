import numpy as np 
from scipy.fft import fft
import math
# returns sample i for a blackman window of size n
# param[in] i: sample index
# param[in] n: size of the window
# return: value of sample i of a blackman window of size n
def blackman(i, n)
  a0 = 0.42
  a1 = 0.5
  a2 = 0.08
  return (a0 - a1 * math.cos((2.0 * math.pi * i) / n)
    + a2 * math.cos((4.0 * math.pi * i) / n))


# applies antenna coupling calibration
# param[in] adc_samples: raw complex adc samples for a frame of radar data
# param[in] coupling_calib: antenna coupling calibration returned 
#                           from dataset_loaders.get_cascade_params()\
# return: post-range-fft radar data with antenna coupling calibration applied
def apply_coupling_calibration(adc_samples, coupling_calib):

  num_tx = coupling_calib['num_tx']
  num_rx = coupling_calib['num_rx']
  num_adc_samples_per_chirp = adc_samples.shape[3]
  num_range_bins = coupling_calib['num_range_bins']
  num_doppler_bins = coupling_calib['num_doppler_bins']

  if num_tx != adc_samples.shape[0]:
    print('provided calibration data dimension 0 does not match data frame dimension 0')
    return None
  if num_rx != adc_samples.shape[1]:
    print('provided calibration data dimension 1 does not match data frame dimension 1')
    return None
  if num_range_bins != num_adc_samples_per_chirp // 2:
    print('provided calibration data dimension 2 does not match data frame dimension 3')
    return None

  # calculate range window
  range_window =  np.zeros(num_adc_samples_per_chirp)
  for i in range(num_adc_samples_per_chirp):
    range_window[i] = blackman(double(i), double(num_adc_samples_per_chirp))

  # apply range window
  adc_samples *= range_window.reshape(1,1,1,-1)

  # run range fft
  range_fft = fft(adc_data, n=num_adc_samples_per_chirp,axis=-1)

  # discard negative component of range fft output
  range_fft = range_fft[:,:,:,1:num_range_bins]

  # apply coupling calibration
  return range_fft - coupling_calib['data'].reshape(num_tx, num_rx, 1, num_range_bins)


# applies phase calibration 
# param[in] adc_samples: raw complex adc samples for a frame of radar data
# param[in] phase_calib: phase calibration returned from dataset_loaders.get_cascade_params()
# return: adc samples with phase calibration applied
def apply_phase_calibration(adc_samples, phase_calib):

  num_tx = phase_calib['num_tx']
  num_rx = phase_calib['num_rx']
  phase_cal_mat = phase_calib['cal_matrix']

  # create phase calibration matrix
  phase_cal_mat = phase_cal_mat[0,0] / phase_cal_mat

  # apply phase calibration
  phase_calibrated_adc_samples = adc_samples * phase_cal_mat.reshape(num_tx, num_rx, 1, 1)

  return phase_calibrated_adc_samples


# applies frequency calibration
# param[in] adc_samples: raw complex adc samples for a frame of radar data
# param[in] freq_calib: frequency calibration returned from dataset_loaders.get_cascade_params()
# param[in] wave_config: waveform config returned from dataset_loaders.get_cascade_params()
# return: adc samples with frequency calibration applied
def apply_frequency_calibration(adc_samples, freq_calib, wave_config):

  num_tx = freq_calib['num_tx']
  num_rx = freq_calib['num_rx']
  cal_frequency_slope = freq_calib['frequency_slope']
  cal_sampling_rate = freq_calib['sampling_rate']
  frequency_slope = wave_config['frequency_slope']
  sampling_rate = wave_config['adc_sample_frequency']
  num_adc_samples_per_chirp = wave_config['num_adc_samples_per_chirp']
  freq_cal_mat = freq_calib['cal_matrix']

  # create frequency calibration matrix
  delta_p = freq_cal_mat - freq_cal_mat[0,0]
  freq_calib = (2.0 * math.pi * delta_p 
                * (frequency_slope / cal_frequency_slope)
                * (cal_sampling_rate / sampling_rate))
  freq_calib /= num_adc_samples_per_chirp
  freq_calib_vec = np.expand_dims(freq_calib, -1) * np.arange(num_adc_samples_per_chirp)
  freq_calib_vec = np.exp(-1j * freq_calib_vec)

  # apply frequency calibration
  freq_calibrated_adc_samples = freq_calib_vec.reshape(num_tx,num_rx,1,num_adc_samples_per_chirp) * adc_samples

  return freq_calibrated_adc_samples
