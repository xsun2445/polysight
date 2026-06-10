"""Digital signal processing utilities for radar data.

Includes steering vectors, angle-of-arrival estimation (MUSIC), chirp
alignment/averaging, synchronization, and raw ADC data reading.
"""

import os
import numpy as np
import scipy
import scipy.signal


def gen_angle_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate steering vectors for AoA estimation.

    Args:
        ang_est_range: Angular span in degrees.
        ang_est_resolution: Resolution in degrees.
        num_ant: Number of Vrx antennas.

    Returns:
        [num_vec, steering_vectors]: Count and array of shape (num_vec, num_ant).
    """
    num_vec = int(round(2 * ang_est_range / ang_est_resolution + 1))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            steering_vectors[kk, jj] = np.cos(mag) + 1j * np.sin(mag)
    return [num_vec, steering_vectors]


def single_angle_steering_vec(ang, num_ant):
    """Generate a single steering vector for a given angle."""
    steering_vec = np.zeros((1, num_ant), dtype='complex64')
    for jj in range(num_ant):
        mag = -1 * np.pi * jj * np.sin(ang * np.pi / 180)
        steering_vec[0, :] = np.cos(mag) + 1j * np.sin(mag)
    return steering_vec


def cov_matrix(x):
    """Spatial covariance matrix (Rxx). Rows = Vrx axis."""
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")
    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    return np.divide(Rxx, num_adc_samples)


def _noise_subspace(covariance, num_sources):
    """Extract noise subspace from covariance matrix."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix should be a 2D square matrix.")
    if num_sources >= covariance.shape[0]:
        raise ValueError("number of sources should be less than number of receivers.")
    _, v = np.linalg.eigh(covariance)
    return v[:, :-num_sources]


def music_1d(steering_vec, rx_chirps, num_sources):
    """1D MUSIC algorithm for angle-of-arrival estimation on ULA."""
    num_antennas = rx_chirps.shape[0]
    assert num_antennas == steering_vec.shape[1]
    if num_antennas < num_sources:
        raise ValueError("number of sources should not exceed number of receivers")
    R = cov_matrix(rx_chirps)
    noise_subspace = _noise_subspace(R, num_sources)
    v = noise_subspace.T.conj() @ steering_vec.T
    return np.reciprocal(np.sum(v * v.conj(), axis=0).real)


def gen_range_steering_vec(expected_resolution_scale, len_sig,
                            freq_start_offset_scale=0, freq_end_offset_scale=1):
    """Generate steering matrix for range MUSIC."""
    freq_start_s = int(expected_resolution_scale * len_sig * freq_start_offset_scale)
    freq_end_s = int(expected_resolution_scale * len_sig * freq_end_offset_scale)
    num_freq = int(expected_resolution_scale * len_sig)
    steering_para = (np.arange(0, len_sig).reshape(1, -1)
                     * np.arange(freq_start_s, freq_end_s).reshape(-1, 1) / num_freq)
    return np.exp(1j * 2 * np.pi * steering_para)


# --- CuPy-accelerated variants ---

def cov_matrix_cupy(x):
    """Spatial covariance matrix using CuPy."""
    import cupy as cp
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")
    _, num_adc_samples = x.shape
    Rxx = x @ cp.conjugate(x.T)
    return cp.divide(Rxx, num_adc_samples)


def _noise_subspace_cupy(covariance, num_sources):
    """Noise subspace extraction using CuPy."""
    import cupy as cp
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance matrix should be a 2D square matrix.")
    if num_sources >= covariance.shape[0]:
        raise ValueError("number of sources should be less than number of receivers.")
    _, v = cp.linalg.eigh(covariance)
    return v[:, :-num_sources]


def music_1d_cupy(steering_vec, rx_chirps, num_sources):
    """1D MUSIC algorithm using CuPy."""
    import cupy as cp
    num_antennas = rx_chirps.shape[0]
    assert num_antennas == steering_vec.shape[1]
    if num_antennas < num_sources:
        raise ValueError("number of sources should not exceed number of receivers")
    R = cov_matrix_cupy(rx_chirps)
    noise_subspace = _noise_subspace_cupy(R, num_sources)
    v = noise_subspace.T.conj() @ steering_vec.T
    return cp.reciprocal(cp.sum(v * v.conj(), axis=0).real)


# --- Chirp alignment ---

def getAveChirp(mat, upscale, domean=True):
    """Average chirps of bistatic signal by frequency/phase alignment."""
    ref = mat[0, :]
    es = 0.1
    ref[np.abs(ref) < es] = es
    divfft = np.fft.fftshift(np.fft.fft(mat / ref, mat.shape[1] * upscale, -1), -1)
    maxlocs = np.argmax(np.abs(divfft), -1, keepdims=True)
    maxvals = np.take_along_axis(divfft, maxlocs, -1)
    df = (maxlocs - mat.shape[1] * upscale / 2) / (mat.shape[1] * upscale)
    angle = np.angle(maxvals)
    res = mat * np.exp(-1j * (2 * np.pi * df * np.arange(mat.shape[1]) + angle))
    if domean:
        res = np.mean(res, 0)
    return res


def getAveChirpAllChannel(mat_allchannel, upscale, sync_ch=0, ref_idx=None, domean=True):
    """Multi-channel chirp averaging using sync channel reference.

    Args:
        mat_allchannel: shape [num_ch, num_chirp, num_adc]
        upscale: FFT upscale factor
        sync_ch: channel used for synchronization
        ref_idx: reference chirp index (auto-selected if None)
        domean: if True, return mean over chirps

    Returns:
        ndarray: shape [num_ch, num_adc] if domean, else [num_ch, num_chirp, num_adc]
    """
    mat = mat_allchannel[sync_ch]
    if ref_idx is None:
        ref_idx = np.argmax(np.sum(np.abs(mat), axis=1))
    ref = mat[ref_idx, :]
    es = 0.1
    ref[np.abs(ref) < es] = es
    divfft = np.fft.fftshift(np.fft.fft(mat / ref, mat.shape[1] * upscale, -1), -1)
    maxlocs = np.argmax(np.abs(divfft), -1, keepdims=True)
    maxvals = np.take_along_axis(divfft, maxlocs, -1)
    df = (maxlocs - mat.shape[1] * upscale / 2) / (mat.shape[1] * upscale)
    angle = np.angle(maxvals)
    res = np.zeros_like(mat_allchannel)
    for ch in range(mat_allchannel.shape[0]):
        res[ch] = mat_allchannel[ch] * np.exp(-1j * (2 * np.pi * df * np.arange(mat.shape[1]) + angle))
    if domean:
        res = np.mean(res, 1)
    return res


# --- Synchronization ---

def find_sync(sig, w_l=10, w_r=10, height=1.5e4, distance=9, target_freq=None, minval=10):
    """Find sync peak in FFT domain and return filtered time-domain signal."""
    tempfft = np.fft.fft(sig)
    locs, _ = scipy.signal.find_peaks(np.abs(tempfft), height=height, distance=distance)
    tempfft[:locs[0] - w_l] = 0
    tempfft[locs[0] + w_r:] = 0
    res = np.fft.ifft(tempfft)
    if minval > 0:
        res[np.abs(res) < minval] = minval
    if target_freq is not None and target_freq >= 0:
        df = (locs[0] - target_freq) / np.max(sig.shape)
        da = np.angle(tempfft[locs[0]])
        res = move_offset(res, df, da)
    return res


def find_offset(sig, ref, upscale_factor=128, w_l=10, w_r=10, height=1.5e4, distance=9):
    """Find frequency and phase offset between signal and reference."""
    num_adc = len(sig)
    tempfft = np.fft.fft(sig)
    locs, _ = scipy.signal.find_peaks(np.abs(tempfft), height=height, distance=distance)
    if len(locs) < 1:
        return [0, 0]
    tempfft[:locs[0] - w_l] = 0
    tempfft[locs[0] + w_r:] = 0
    div_fft = np.fft.fftshift(np.fft.fft(np.fft.ifft(tempfft) / ref, num_adc * upscale_factor))
    argmax = np.argmax(np.abs(div_fft))
    da = np.angle(div_fft[argmax])
    df = (argmax - num_adc * upscale_factor / 2) / (num_adc * upscale_factor)
    return df, da


def move_offset(sig, df, da):
    """Apply frequency/phase offset correction to signal."""
    return sig * np.exp(-1j * (2 * np.pi * df * np.arange(np.max(sig.shape)) + da))


def sync_ref(sig, ref, upscale_factor=128, w_l=10, w_r=10, height=1.5e4, distance=9):
    """Synchronize a signal to a reference sync peak."""
    [df, da] = find_offset(sig, ref, upscale_factor=upscale_factor,
                           w_l=w_l, w_r=w_r, height=height, distance=distance)
    return move_offset(sig, df, da)


def find_offset_norm(sig, ref, upscale_factor=128, w_l=10, w_r=10, height=1.5e4, distance=9):
    """Find offset with amplitude normalization."""
    num_adc = len(sig)
    tempfft = np.fft.fft(sig)
    locs, _ = scipy.signal.find_peaks(np.abs(tempfft), height=height,
                                       prominence=height / 3, distance=distance)
    if len(locs) < 1:
        return [0, 0, 1]
    tempfft[:locs[0] - w_l] = 0
    tempfft[locs[0] + w_r:] = 0
    div_fft = np.fft.fftshift(np.fft.fft(np.fft.ifft(tempfft) / ref, num_adc * upscale_factor))
    argmax = np.argmax(np.abs(div_fft))
    da = np.angle(div_fft[argmax])
    df = (argmax - num_adc * upscale_factor / 2) / (num_adc * upscale_factor)
    norm = (np.max(np.abs(np.fft.fft(ref, 256 * 5)))
            / np.max(np.abs(np.fft.fft(np.fft.ifft(tempfft), 256 * 5))))
    return [df, da, norm]


def sync_ref_norm(sig, ref, upscale_factor=128, w_l=10, w_r=10, height=1.5e4, distance=9):
    """Synchronize with amplitude normalization."""
    [df, da, norm] = find_offset_norm(sig, ref, upscale_factor=upscale_factor,
                                       w_l=w_l, w_r=w_r, height=height, distance=distance)
    res = move_offset(sig, df, da)
    return res, norm


# --- Raw ADC reading ---

def readDCA1000(fileName):
    """Read raw ADC binary file from DCA1000."""
    numLanes = 4
    adcData = np.fromfile(open(fileName, 'rb'), np.int16).reshape([2 * numLanes, -1], order='F')
    adcData = adcData[:4] + 1j * adcData[4:]
    return adcData


def readDCA1000_zerofilling(fileName, seqlist, bytes_in_frame=None):
    """Read raw ADC binary with zero-filling for lost packets."""
    from polysight.collection import radar

    numLanes = 4
    rawData = open(fileName, 'rb').read()
    cnt = 0
    for curr_seqn in seqlist:
        numlost = curr_seqn - cnt - 1
        if numlost > 0:
            rawData = (rawData[:cnt * radar.BYTES_IN_PACKET]
                       + radar.FILLING_PACKET * numlost
                       + rawData[cnt * radar.BYTES_IN_PACKET:])
        cnt = curr_seqn
    if bytes_in_frame is not None:
        rawData += b'\x00' * len(rawData) % bytes_in_frame
    adcData = np.frombuffer(rawData, np.int16).reshape([2 * numLanes, -1], order='F')
    adcData = adcData[:4] + 1j * adcData[4:]
    return adcData


def shift_aveAdcData_to_argmax(fileDir, radarName, focus_angle,
                                idx_pos_start=None, idx_pos_end=None,
                                idx_adc_start=16, idx_adc_end=-64,
                                ref_nfft=1024, nfft_scale=128, sync_idx=256,
                                fileName_ori='tx0', fileName_shifted='tx0_shifted_argmax'):
    """Shift averaged ADC data to align with argmax peak."""
    fileName = os.path.join(fileDir, f'adcData/{radarName}/{fileName_ori}.npy')
    saveName = os.path.join(fileDir, f'adcData/{radarName}/{fileName_shifted}.npy')
    assert not os.path.exists(saveName), f"File already exists: {saveName}"

    adcData = np.load(fileName)
    if idx_pos_start is None:
        idx_pos_start = 0
    if idx_pos_end is None:
        idx_pos_end = adcData.shape[1]

    num_vec, angle_steering_vec = gen_angle_steering_vec(90, 0.5, 4)
    avesig = np.mean(adcData[:, idx_pos_start:idx_pos_end], axis=(1, 2))
    focused_sig = angle_steering_vec[focus_angle, :] @ np.fft.fft(avesig[:, idx_adc_start:idx_adc_end])

    nfft = nfft_scale * ref_nfft
    sig_ifft = np.fft.ifft(focused_sig)
    argmax = np.argmax(np.abs(np.fft.fft(sig_ifft, nfft)))
    df = (argmax - sync_idx * nfft_scale) / nfft
    shifted_sig = sig_ifft * np.exp(-1j * 2 * np.pi * (df * np.arange(len(sig_ifft))))
    da = np.angle(np.fft.fft(shifted_sig, ref_nfft)[sync_idx])
    adcData_shifted = adcData * np.exp(-1j * (2 * np.pi * df * np.arange(adcData.shape[-1]) + da))
    np.save(saveName, adcData_shifted.astype(np.complex64))
    return adcData_shifted
