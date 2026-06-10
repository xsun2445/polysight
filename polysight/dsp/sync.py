"""Multi-channel radar data synchronization.

Provides reference-based synchronization for aligning ADC data cubes
across spatial positions using sync peak tracking.
"""

import numpy as np
from polysight.dsp import signal
from polysight.collection import decode


ADC_PARAMS = {
    'chirps': 1,
    'configs': 1,
    'frames': 1,
    'rx': 4,
    'samples': 256,
    'IQ': 2,
    'bytes': 2,
}

BYTES_IN_FRAME = (ADC_PARAMS['chirps'] * ADC_PARAMS['rx'] * ADC_PARAMS['configs']
                  * ADC_PARAMS['frames'] * ADC_PARAMS['IQ'] * ADC_PARAMS['samples']
                  * ADC_PARAMS['bytes'])


def reposition_from_file(fileName, bytes_in_frame=BYTES_IN_FRAME,
                          thresh_packet_transfer=0.15, force_to_number=None):
    """Reposition frames from a binary file using its packet log."""
    logName = fileName
    if '_log.csv' not in fileName:
        logName = fileName.replace('.bin', '_log.csv')
    [seqn_list, byte_list, ts_list] = decode.readLog(logName)
    frame_locs, byte_locs = decode.framePositioning(
        bytes_in_frame, seqn_list, byte_list, ts_list,
        thresh_packet_transfer, force_to_number)
    return frame_locs, byte_locs
