"""Optimized SAR image generation using batched FFT convolution.

Replaces per-bin Python loops with vectorized GPU operations:
  - Matched-filter cube: vectorized broadcast cp.exp()
  - Convolution: batched FFT2 + element-wise multiply + IFFT2
  - Sum-bins fast path: frequency-domain accumulation with single IFFT

Usage:
    from polysight.dsp.sar import sar_eval, calc_sar_fromfile
"""

import numpy as np
import scipy.constants
import cupy as cp
import cupyx.scipy.fft as cufft
import json
import os
import copy
import time

from polysight.dsp.zoom_fft import gen_zoom_fft_cupy

C = scipy.constants.speed_of_light
PI = scipy.constants.pi


def sar_bp_fft(fftData, sar_configs, sum_bins=False, chunk_size=0):
    """Back-projection SAR using batched FFT convolution.

    Parameters
    ----------
    fftData : cupy.ndarray, shape [numY, numX, num_range_bins]
        Range-compressed data for one RX channel.
    sar_configs : dict
        SAR configuration dictionary.
    sum_bins : bool
        If True, coherently sum over range bins.
    chunk_size : int
        Process this many bins at a time (0 = all at once).

    Returns
    -------
    cupy.ndarray
        Complex SAR image(s).
    """
    chirp_configs = sar_configs['chirp_configs']
    alignment_configs = sar_configs['alignment_configs']

    tx_loc = np.asarray(sar_configs['tx_loc'])
    rx_loc = np.asarray(sar_configs['rx_loc'])
    tarplane_loc = np.asarray(sar_configs['tarplane_loc'])

    dx = sar_configs['dx']
    dy = sar_configs['dy']
    numX = sar_configs['numX']
    numY = sar_configs['numY']
    output_size_x = sar_configs['output_size_x']
    output_size_y = sar_configs['output_size_y']
    nFFTspaceX = output_size_x + numX - 1
    nFFTspaceY = output_size_y + numY - 1

    num_bin = sar_configs['num_fft_bin']
    idx_offset = sar_configs['idx_bin_offset']
    flag_mask = sar_configs['masked_matched_filter']

    # Frequency parameters
    f0 = (chirp_configs['f0']
          + (chirp_configs['adc_start_time']
             + alignment_configs['idx_adc_start'] / chirp_configs['Fs'])
          * chirp_configs['slope'])
    f_ref = f0 + alignment_configs['ref_idx_aligned'] / alignment_configs['fs_zoom'] * chirp_configs['Fs']
    df = chirp_configs['Fs'] / alignment_configs['fs_zoom'] / alignment_configs['nfft_scale']
    dr = df * C / chirp_configs['slope']

    # Geometry (CPU, float64 for precision)
    dl_rxtx = np.sqrt(np.sum((tx_loc - rx_loc) ** 2))
    x = dx * (np.arange(nFFTspaceX) - nFFTspaceX / 2)
    y = dy * (np.arange(nFFTspaceY) - nFFTspaceY / 2)
    x = x.reshape(1, -1)
    y = y.reshape(-1, 1)

    x_tar = x + tarplane_loc[0]
    y_tar = y + tarplane_loc[1]
    z_tar = tarplane_loc[2]

    Rl_tx = np.sqrt((x_tar - tx_loc[0])**2 + (y_tar - tx_loc[1])**2 + (z_tar - tx_loc[2])**2)
    Rl_rx = np.sqrt((x_tar - rx_loc[0])**2 + (y_tar - rx_loc[1])**2 + (z_tar - rx_loc[2])**2)
    Rl = Rl_tx + Rl_rx

    Rl_idx = np.round((Rl - dl_rxtx) / dr).astype(int)
    min_idx = int(np.min(Rl_idx))

    if num_bin == -1:
        num_bin = int(np.max(Rl_idx) - min_idx + 1)

    # Move to GPU
    Rl_gpu = cp.asarray(Rl, dtype=cp.float32)

    # Vectorized matched-filter cube: [num_bin, nFFTspaceY, nFFTspaceX]
    idx_all = cp.arange(min_idx + idx_offset, min_idx + idx_offset + num_bin, dtype=cp.float32)
    k_all = (2 * PI * (f_ref + idx_all * df) / C).astype(cp.float32)
    matchedFilter_cube = cp.exp(-1j * k_all[:, None, None] * Rl_gpu[None, :, :])

    if flag_mask:
        Rl_idx_gpu = cp.asarray(Rl_idx, dtype=cp.int32)
        idx_all_int = cp.arange(min_idx + idx_offset, min_idx + idx_offset + num_bin, dtype=cp.int32)
        mask = (Rl_idx_gpu[None, :, :] == idx_all_int[:, None, None])
        matchedFilter_cube *= mask
        del mask

    # Prepare signal batch: [num_bin, numY, numX]
    bin_start = min_idx + idx_offset
    bin_end = bin_start + num_bin
    signal_batch = fftData[:, :, bin_start:bin_end]
    signal_batch = cp.moveaxis(signal_batch, -1, 0)

    # Batched FFT convolution
    fft_y = nFFTspaceY + numY - 1
    fft_x = nFFTspaceX + numX - 1
    fft_shape = (fft_y, fft_x)

    if sum_bins:
        sarImage = _fft_conv_sum(matchedFilter_cube, signal_batch, fft_shape, num_bin, chunk_size)
        sarImage = sarImage[numY - 1: numY - 1 + output_size_y,
                            numX - 1: numX - 1 + output_size_x]
    else:
        sarImage = _fft_conv_batch(matchedFilter_cube, signal_batch, fft_shape,
                                    num_bin, chunk_size,
                                    numY, numX, output_size_y, output_size_x)

    del matchedFilter_cube, signal_batch
    cp.get_default_memory_pool().free_all_blocks()
    return sarImage


def _fft_conv_sum(mf_cube, sig_batch, fft_shape, num_bin, chunk_size):
    """Sum_i conv(mf[i], sig[i]) via FFT accumulation."""
    if chunk_size <= 0 or chunk_size >= num_bin:
        H = cufft.fft2(mf_cube, s=fft_shape)
        X = cufft.fft2(sig_batch, s=fft_shape)
        accum = (H * X).sum(axis=0)
        del H, X
        return cufft.ifft2(accum)
    else:
        accum = cp.zeros(fft_shape, dtype=cp.complex64)
        for i0 in range(0, num_bin, chunk_size):
            i1 = min(i0 + chunk_size, num_bin)
            H = cufft.fft2(mf_cube[i0:i1], s=fft_shape)
            X = cufft.fft2(sig_batch[i0:i1], s=fft_shape)
            accum += (H * X).sum(axis=0)
            del H, X
        return cufft.ifft2(accum)


def _fft_conv_batch(mf_cube, sig_batch, fft_shape, num_bin, chunk_size,
                     numY, numX, output_size_y, output_size_x):
    """Per-bin FFT convolution, returns [num_bin, oy, ox]."""
    sarImage = cp.zeros([num_bin, output_size_y, output_size_x], dtype=cp.complex64)
    if chunk_size <= 0 or chunk_size >= num_bin:
        H = cufft.fft2(mf_cube, s=fft_shape)
        X = cufft.fft2(sig_batch, s=fft_shape)
        full = cufft.ifft2(H * X)
        sarImage[:] = full[:, numY - 1: numY - 1 + output_size_y,
                             numX - 1: numX - 1 + output_size_x]
        del H, X, full
    else:
        for i0 in range(0, num_bin, chunk_size):
            i1 = min(i0 + chunk_size, num_bin)
            H = cufft.fft2(mf_cube[i0:i1], s=fft_shape)
            X = cufft.fft2(sig_batch[i0:i1], s=fft_shape)
            full = cufft.ifft2(H * X)
            sarImage[i0:i1] = full[:, numY - 1: numY - 1 + output_size_y,
                                      numX - 1: numX - 1 + output_size_x]
            del H, X, full
    return sarImage


def sar_eval(sar_config, radar_name, sum_bins=False, chunk_size=0, base_dir=None):
    """Generate SAR images for all 4 RX channels of a given radar.

    Returns
    -------
    numpy.ndarray
        sum_bins=False: shape [4, num_bin, output_size_y, output_size_x]
        sum_bins=True:  shape [4, output_size_y, output_size_x]
    """
    radar_name_to_idx = {x: i for i, x in enumerate(['LS', 'RH', 'RV'])}
    radar_idx = radar_name_to_idx[radar_name]

    _cfg = copy.deepcopy(sar_config)
    cfg_key = f'{radar_name}_config'
    if cfg_key not in _cfg:
        cfg_key = 'RH_config'
    radar_cfg = _cfg[cfg_key]
    alignment_cfg = radar_cfg['alignment_configs']
    _cfg['alignment_configs'] = alignment_cfg
    alignment_idx = ((alignment_cfg['ref_idx_aligned'] - alignment_cfg['idx_fft_start_zoom'])
                     * alignment_cfg['nfft_scale'])
    _cfg['tarplane_loc'] = np.array(radar_cfg['sar_focuse_center'])
    ant_locs = np.array(_cfg['ant_locs'])
    _cfg['tx_loc'] = ant_locs[_cfg['tx_idx']]
    _cfg['masked_matched_filter'] = radar_cfg['masked_matched_filter']
    _cfg['num_fft_bin'] = radar_cfg['num_fft_bin']
    _cfg['idx_bin_offset'] = radar_cfg['idx_bin_offset']

    # Load and preprocess ADC data
    file_dir = _cfg['fileDir']
    if base_dir is not None:
        file_dir = os.path.join(base_dir, file_dir)
    fileName = os.path.join(file_dir, f'{radar_name}_synced.npy')
    adcData = np.load(fileName)
    adcData = adcData[:, :, :, alignment_cfg['idx_adc_start']:alignment_cfg['idx_adc_end']]
    if radar_cfg['remove_avesig']:
        adcData -= np.mean(adcData, axis=(1, 2), keepdims=True)

    # Zoom FFT (GPU)
    transform = gen_zoom_fft_cupy(alignment_cfg)
    fftData = transform(cp.asarray(adcData, dtype=cp.complex64)).astype(cp.complex64)
    del adcData
    fftData = cp.nan_to_num(fftData, nan=0)
    fftData = fftData[:, :, :, alignment_idx:]

    # SAR back-projection per channel
    sarImage_list = []
    for ch in range(4):
        rx_loc = ant_locs[7 * radar_idx + ch]
        _cfg['rx_loc'] = rx_loc
        sarImage = sar_bp_fft(fftData[ch], _cfg, sum_bins=sum_bins, chunk_size=chunk_size)
        sarImage_list.append(np.flip(sarImage.get(), axis=[-1, -2]))

    del fftData
    cp.get_default_memory_pool().free_all_blocks()

    if sum_bins:
        return np.array(sarImage_list)
    min_bin_num = min(x.shape[0] for x in sarImage_list)
    return np.array([x[:min_bin_num] for x in sarImage_list])


def calc_sar_fromfile(collection_name, radar_list, sum_bins=False, chunk_size=0,
                      labels_dir=None, base_dir=None):
    """Generate SAR images from a labeled collection JSON.

    Parameters
    ----------
    collection_name : str
    radar_list : list of str
    sum_bins : bool
    chunk_size : int
    labels_dir : str
    base_dir : str
        Root directory to resolve relative fileDir paths in the JSON config.

    Returns
    -------
    label_dict, sar_list
    """
    save_folder = os.path.join(labels_dir, collection_name)
    config_path = os.path.join(save_folder, f'{collection_name}.json')

    with open(config_path, 'r') as f:
        label_dict = json.load(f)

    sar_list = []
    for radar_name in radar_list:
        t0 = time.time()
        sar = sar_eval(label_dict['sar_config'], radar_name,
                       sum_bins=sum_bins, chunk_size=chunk_size,
                       base_dir=base_dir)
        if sum_bins:
            sar = np.mean(sar, axis=0)
        else:
            sar = np.mean(sar, axis=(0, 1))
        sar_list.append(sar)
        print(f'  {radar_name}: {time.time() - t0:.2f}s, shape {sar.shape}')

    return label_dict, sar_list
