"""Decode raw binary radar data into organized ADC sample arrays.

Handles frame positioning from packet logs, chirp alignment, and
multi-channel synchronization.
"""

import csv
import numpy as np
import os
import glob

from polysight.collection import radar
from polysight.dsp import signal


def readLog(fileName_log):
    """Read packet log CSV: returns [seqnum, bytecnt, timestamp]."""
    timestamp = []
    bytecnt = []
    seqnum = []
    with open(fileName_log, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if timestamp:
                timestamp.append(float(row[-1]) - timestamp[0])
            else:
                timestamp.append(float(row[-1]))
            seqnum.append(int(row[0]))
            bytecnt.append(int(row[1]))
    timestamp[0] = 0
    return [seqnum, bytecnt, timestamp]


def framePositioning_force_to_number(bytes_in_frame, seqn_list, byte_list, force_to_number):
    """Force number of frames per location to be force_to_number."""
    curr_seqn = 1
    prev_seqn = 0
    cnt_frame_at_loc = 0
    cnt_byte = 0
    ptr = 0

    frame_locs = []
    byte_locs = []
    flag_discard_frame = False

    while ptr < len(seqn_list):
        while cnt_byte < bytes_in_frame:
            if curr_seqn - prev_seqn > 1:
                flag_discard_frame = True
                cnt_byte += radar.BYTES_IN_PACKET * (curr_seqn - prev_seqn - 1)
            prev_seqn = curr_seqn
            cnt_byte += byte_list[ptr]
            byte_locs.append(cnt_frame_at_loc // force_to_number)
            ptr += 1

        cnt_byte -= bytes_in_frame
        cnt_frame_at_loc += 1
        if flag_discard_frame:
            frame_locs.append(-1)
        else:
            frame_locs.append((cnt_frame_at_loc - 1) // force_to_number)
        flag_discard_frame = False

    return frame_locs, byte_locs


def framePositioning_missing_frame(bytes_in_frame, seqn_list, byte_list, ts_list,
                                    thresh_packet_transfer=0.05, ave_delay_time=0.1):
    """Frame positioning handling missing frames via timestamp analysis."""
    ptr_p = 0
    cnt_frame = 0
    cnt_byte = 0
    curr_seqn = 1
    prev_seqn = 0
    ptr_t = 0
    loc_idx = 0
    frame_locs = []
    byte_locs = []
    flag_discard_frame = False

    while ptr_p < len(seqn_list) and ptr_t < len(seqn_list):
        prev_delayed_time = ts_list[ptr_t + 1] - ts_list[ptr_t]
        while prev_delayed_time < thresh_packet_transfer and ptr_t < len(seqn_list):
            ptr_t += 1
            prev_delayed_time = ts_list[ptr_t + 1] - ts_list[ptr_t]
        ptr_t += 1

        if prev_delayed_time // ave_delay_time > 1 and ptr_t < len(seqn_list) - 1:
            loc_idx += int(prev_delayed_time // ave_delay_time - 1)

        while ptr_p < ptr_t:
            while cnt_byte < bytes_in_frame:
                curr_seqn = seqn_list[ptr_p]
                if curr_seqn - prev_seqn > 1:
                    flag_discard_frame = True
                    cnt_byte += radar.BYTES_IN_PACKET * (curr_seqn - prev_seqn - 1)
                prev_seqn = curr_seqn
                cnt_byte += byte_list[ptr_p]
                ptr_p += 1
                byte_locs.append(loc_idx)

            cnt_byte -= bytes_in_frame
            cnt_frame += 1
            if flag_discard_frame:
                frame_locs.append(-1)
            else:
                frame_locs.append(loc_idx)
                flag_discard_frame = False

        loc_idx += 1

    return frame_locs, byte_locs


def framePositioning(bytes_in_frame, seqn_list, byte_list, ts_list,
                     thresh_packet_transfer=0.05, ave_delay_time=None, force_to_number=None):
    """Determine frame-to-location mapping from packet logs."""
    if ave_delay_time is not None:
        return framePositioning_missing_frame(bytes_in_frame, seqn_list, byte_list, ts_list,
                                              thresh_packet_transfer, ave_delay_time)
    if force_to_number is not None:
        return framePositioning_force_to_number(bytes_in_frame, seqn_list, byte_list, force_to_number)

    ptr_p = 0
    cnt_frame = 0
    cnt_byte = 0
    curr_seqn = 1
    prev_seqn = 0
    ptr_t = 0
    loc_idx = 0
    frame_locs = []
    byte_locs = []
    flag_discard_frame = False

    while ptr_p < len(seqn_list) and ptr_t < len(seqn_list):
        while ts_list[ptr_t + 1] - ts_list[ptr_t] < thresh_packet_transfer and ptr_t < len(seqn_list):
            ptr_t += 1
        ptr_t += 1

        while ptr_p < ptr_t:
            while cnt_byte < bytes_in_frame:
                curr_seqn = seqn_list[ptr_p]
                if curr_seqn - prev_seqn > 1:
                    flag_discard_frame = True
                    cnt_byte += radar.BYTES_IN_PACKET * (curr_seqn - prev_seqn - 1)
                prev_seqn = curr_seqn
                cnt_byte += byte_list[ptr_p]
                ptr_p += 1
                byte_locs.append(loc_idx)

            cnt_byte -= bytes_in_frame
            cnt_frame += 1
            if flag_discard_frame:
                frame_locs.append(-1)
            else:
                frame_locs.append(loc_idx)
                flag_discard_frame = False

        loc_idx += 1

    return frame_locs, byte_locs


def calc_frame_byte(cfg_collection):
    """Calculate bytes per frame from collection config."""
    adc_params = {
        'num_chirp': cfg_collection['num_chirp'],
        'num_config': cfg_collection['num_config'],
        'num_frame': cfg_collection['num_frame'],
        'num_ch': 4,
        'num_adc': cfg_collection['num_adc'],
        'IQ': 2,
        'bytes': 2,
    }
    return _calc_frame_byte(adc_params)


def _calc_frame_byte(adc_params):
    return (adc_params['num_chirp'] * adc_params['num_ch'] * adc_params['num_config']
            * adc_params['IQ'] * adc_params['num_adc'] * adc_params['bytes'])


def decodeDataLine(fileName, cfg_collection, cfg_decode=None, fileName_log=None):
    """Decode a single scan line: raw binary -> aligned ADC numpy array."""
    from itertools import product

    if fileName_log is None:
        fileName_log = fileName.replace('.bin', '_log.csv')

    if cfg_decode is None:
        cfg_decode = {
            'method': 'alignchirp',
            'fft_upscale': 128,
            'save': True,
            'saveSubDir': '_adcData',
        }

    if 'num_ch' not in cfg_collection:
        cfg_collection['num_ch'] = 4
    if 'sync_ch' not in cfg_decode:
        cfg_decode['sync_ch'] = 0

    seqn_list, bytecnt_list, ts_list = readLog(fileName_log)

    if 'packet_zero_filling' not in cfg_collection or not cfg_collection['packet_zero_filling']:
        rawData = signal.readDCA1000_zerofilling(fileName, seqn_list)
    else:
        rawData = signal.readDCA1000(fileName)

    rawData = rawData.astype(np.complex64)
    rawData = rawData.reshape(cfg_collection['num_ch'], -1,
                              cfg_collection['num_config'],
                              cfg_collection['num_adc'])

    # Amplitude correction
    amp_shift_saveName = os.path.join(os.path.dirname(__file__), '..', '..', 'data',
                                      'RV_rx0_correction.npy')
    if os.path.exists(amp_shift_saveName):
        amp_shift = np.load(amp_shift_saveName)
        amp_shift /= np.max(amp_shift)
        _fftData = np.fft.fft(rawData, 4096, -1)
        _fftData /= amp_shift
        rawData = np.fft.ifft(_fftData, 4096, -1)[:, :, :, :cfg_collection['num_adc']]

    thresh_packet_transfer = 0.045
    ave_delay_time = 0.09
    frame_locs, byte_locs = framePositioning(
        calc_frame_byte(cfg_collection), seqn_list, bytecnt_list, ts_list,
        thresh_packet_transfer=thresh_packet_transfer, ave_delay_time=ave_delay_time)

    if cfg_collection['moving_direction'] == 'row':
        num_locs = cfg_collection['total_x']
    else:
        num_locs = cfg_collection['total_y']

    if num_locs * 16 == rawData.shape[1]:
        force_to_number = 16
        frame_locs, byte_locs = framePositioning(
            calc_frame_byte(cfg_collection), seqn_list, bytecnt_list, ts_list,
            force_to_number=force_to_number)

    frame_locs = np.array(frame_locs)

    alignedData = np.zeros([cfg_collection['num_config'],
                            cfg_collection['num_ch'],
                            num_locs,
                            cfg_collection['num_adc']],
                           dtype=np.complex64)

    for idx_config in range(cfg_collection['num_config']):
        for idx_loc in range(num_locs):
            temp = rawData[:, frame_locs == idx_loc, idx_config, :]
            if temp.size == 0:
                print(f'idx {idx_loc} found no data, {fileName}')
                continue

            if cfg_decode['method'] == 'alignchirp':
                ave_chirp = signal.getAveChirpAllChannel(
                    temp, upscale=cfg_decode['fft_upscale'], sync_ch=cfg_decode['sync_ch'])
            elif cfg_decode['method'] == 'mean':
                ave_chirp = np.mean(temp, axis=1)
            alignedData[idx_config, :, idx_loc, :] = ave_chirp.astype(np.complex64)

    if cfg_decode['save']:
        radar_name = split_path(fileName)[-2]
        base_name, ext_name = os.path.splitext(split_path(fileName)[-1])
        saveRootDir = os.path.join(*split_path(fileName)[:-2], cfg_decode['saveSubDir'])
        if not os.path.exists(saveRootDir):
            os.mkdir(saveRootDir)
        saveDir = os.path.join(saveRootDir, radar_name)
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        for idx_config, idx_ch in product(range(cfg_collection['num_config']),
                                          range(cfg_collection['num_ch'])):
            saveName = base_name + '_' + str(idx_config) + '_' + str(idx_ch) + '.npy'
            saveName = os.path.join(saveDir, saveName)
            np.save(saveName, alignedData[idx_config, idx_ch])

    return alignedData


def decodeDataLine_multithread_wrapper(*args):
    try:
        decodeDataLine(*args)
    except Exception as e:
        print(Exception(f"An error occurred: {e}. input: {args[0]}"))


def decodeDataLine_multiprocess(fileDir, radar_name, cfg_collection, cfg_decode,
                                 fileName_log=None, num_core=12):
    """Decode all scan lines in a directory using multiprocessing."""
    import multiprocessing

    fileName = os.path.join(fileDir, radar_name, '*.bin')
    fileList = glob.glob(fileName)
    print(radar_name, len(fileList))

    args_multiprocess = [(x, cfg_collection, cfg_decode) for x in fileList]

    if num_core > 1:
        with multiprocessing.Pool(processes=num_core) as pool:
            pool.starmap(decodeDataLine_multithread_wrapper, args_multiprocess)
    else:
        for args in args_multiprocess:
            try:
                decodeDataLine(*args)
            except Exception as e:
                print(f"An error occurred:{args[0]} {e}")


def combineLine(fileDir, cfg_collection=None, save=True, saveSubDir='adcData'):
    """Combine decoded scan lines into a 2D data cube."""
    from polysight.collection import collect
    from itertools import product

    axis = cfg_collection['moving_direction']
    total_x = cfg_collection['total_x']
    total_y = cfg_collection['total_y']
    num_config = cfg_collection['num_config']
    num_adc = cfg_collection['num_adc']
    num_ch = cfg_collection.get('num_ch', 4)
    ext_name = '.npy'

    fileList = glob.glob(os.path.join(fileDir, '*' + ext_name))
    path_ary = collect.MasterDevice.generate_path_ary(total_x, total_y, mode=axis)

    for idx_config, idx_ch in product(range(num_config), range(num_ch)):
        dataCube = np.zeros([total_y, total_x, num_adc], dtype=np.complex64)
        i_row = 0
        for loc in path_ary[::2]:
            file_name = '_'.join([str(x) for x in list(loc) + [idx_config, idx_ch]]) + ext_name
            file_name = os.path.join(fileDir, file_name)
            if file_name in fileList:
                dataLine = np.load(file_name)
                if i_row % 2 == 1:
                    dataLine = dataLine[::-1]
                if axis == 'row':
                    dataCube[i_row] = dataLine
                elif axis == 'col':
                    dataCube[:, i_row] = dataLine
            i_row += 1

        if save:
            radar_name = split_path(fileDir)[-1]
            saveRootDir = os.path.join(*split_path(fileDir)[:-2], saveSubDir)
            if not os.path.exists(saveRootDir):
                os.mkdir(saveRootDir)
            saveDir = os.path.join(saveRootDir, radar_name)
            if not os.path.exists(saveDir):
                os.mkdir(saveDir)
            saveName = os.path.join(saveDir, str(idx_config) + '_' + str(idx_ch) + ext_name)
            np.save(saveName, dataCube)


def loadSameConfig(fileDir, idx_config, ext_name='.npy'):
    """Load different channels of adcData with same configuration."""
    _fileList = glob.glob(os.path.join(fileDir, '*' + ext_name))
    fileList = [name for name in _fileList
                if os.path.splitext(split_path(name)[-1])[0].split('_')[0] == str(idx_config)]
    fileList = sorted(fileList, key=lambda x: os.path.splitext(split_path(x)[-1])[0].split('_')[1])
    num_ch = len(fileList)
    assert num_ch == 4, "num_ch != 4: num of channel should always be 4."
    dataList = [np.load(name) for name in fileList]
    adcData = np.zeros([num_ch] + list(dataList[0].shape), dtype=np.complex64)
    for idx, data in enumerate(dataList):
        adcData[idx] = data
    return adcData


def _syncMat(adcData, cfg_sync):
    """Synchronize all spatial locations to a reference using sync channel."""
    sync_ch = cfg_sync['sync_ch']
    ref_loc = cfg_sync['ref_loc']
    total_x = cfg_sync['total_x']
    total_y = cfg_sync['total_y']

    ref = signal.find_sync(adcData[*ref_loc, sync_ch], **cfg_sync['sync_peak'])
    adcData_aligned = np.zeros_like(adcData)

    for y in range(total_y):
        print(y)
        for x in range(total_x):
            [df, da] = signal.find_offset(adcData[y, x, sync_ch], ref, **cfg_sync['sync_matrix'])
            adcData_aligned[y, x] = signal.move_offset(adcData[y, x], df, da)

    adcData_aligned = np.moveaxis(adcData_aligned, 2, 0)
    return adcData_aligned


def syncMat(fileDir, idx_config, cfg_sync):
    """Load, synchronize, and optionally save aligned ADC data."""
    adcData = loadSameConfig(fileDir, idx_config)
    adcData = np.moveaxis(adcData, 0, 2)
    adcData_aligned = _syncMat(adcData, cfg_sync)

    if cfg_sync['save']:
        ext_name = '.npy'
        radar_name = split_path(fileDir)[-1].replace('_', '')
        if 'dir_prefix' in cfg_sync and cfg_sync['dir_prefix']:
            radar_name = cfg_sync['dir_prefix'] + radar_name

        saveDir = os.path.join(*split_path(fileDir)[:-1], radar_name)
        if not os.path.exists(saveDir):
            os.mkdir(saveDir)
        for ch in range(4):
            saveName = os.path.join(saveDir, str(idx_config) + '_' + str(ch) + ext_name)
            np.save(saveName, adcData_aligned[ch])

    return adcData_aligned


def split_path(path):
    """Split a path into all its components."""
    path = os.path.split(path)
    if path[1]:
        return list(split_path(path[0])) + [path[-1]]
    else:
        return [path[0]]
