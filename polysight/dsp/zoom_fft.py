"""Zoom FFT (Chirp-Z Transform) for high-resolution range profiling."""

import scipy.signal
import cupyx.scipy.signal


def gen_zoom_fft(alignment_cfg):
    """Generate a SciPy ZoomFFT transform (CPU)."""
    siglen = abs(alignment_cfg['idx_adc_start'] - alignment_cfg['idx_adc_end'])
    ns_zoom = (abs(alignment_cfg['idx_fft_start_zoom'] - alignment_cfg['idx_fft_end_zoom'])
               * alignment_cfg['nfft_scale'])
    return scipy.signal.ZoomFFT(
        siglen,
        [alignment_cfg['idx_fft_start_zoom'], alignment_cfg['idx_fft_end_zoom']],
        ns_zoom,
        fs=alignment_cfg['fs_zoom'],
    )


def gen_zoom_fft_cupy(alignment_cfg):
    """Generate a CuPy ZoomFFT transform (GPU)."""
    siglen = abs(alignment_cfg['idx_adc_start'] - alignment_cfg['idx_adc_end'])
    ns_zoom = (abs(alignment_cfg['idx_fft_start_zoom'] - alignment_cfg['idx_fft_end_zoom'])
               * alignment_cfg['nfft_scale'])
    return cupyx.scipy.signal.ZoomFFT(
        siglen,
        [alignment_cfg['idx_fft_start_zoom'], alignment_cfg['idx_fft_end_zoom']],
        ns_zoom,
        fs=alignment_cfg['fs_zoom'],
    )
