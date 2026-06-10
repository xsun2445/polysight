"""2D SAR data collection orchestrator.

Coordinates multi-radar capture with motorized 2D scanning stage.
Manages DCA1000 FPGA boards, motor control, and data saving.
"""

import json
import os
import threading
from threading import Thread
import struct
import time
import datetime
import socket

import numpy as np

from polysight.collection import radar
from polysight.collection.motor import Motor


def printlog(*str_to_print, fileName=None, verbose=True):
    """Log to file and/or console."""
    if fileName:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.4%f"),
              *str_to_print, file=open(fileName, 'a+'))
    if verbose:
        print(*str_to_print)


class _DCA1000(radar.DCA1000):
    """DCA1000 subclass with continuous recording support."""

    def record(self, saveName, radar_name, fileName_log=None,
               timeout=4, zero_filling=True):
        """Record radar data to file with packet loss handling.

        Parameters
        ----------
        saveName : str
            Path to save binary data.
        radar_name : str
            Radar identifier for logging.
        fileName_log : str or None
            Log file path.
        timeout : float
            Socket timeout in seconds.
        zero_filling : bool
            Fill missing packets with zeros.
        """
        self.data_socket.settimeout(timeout)

        f_data = open(saveName, 'ab+')
        logName = saveName.replace('.bin', '_log.csv')
        f_log = open(logName, 'a+')

        prev_seq = 0

        while True:
            try:
                data, addr = self.data_socket.recvfrom(radar.MAX_PACKET_SIZE)
                packet_num = struct.unpack('<1l', data[:4])[0]
                prev_seq += 1

                if prev_seq < packet_num:
                    printlog('radar:', radar_name, 'missing packet',
                             packet_num, 'diff', packet_num - prev_seq,
                             fileName=fileName_log)
                    while prev_seq < packet_num:
                        if zero_filling:
                            data = radar.FILLING_PACKET + data
                        prev_seq += 1

                f_data.write(data[10:])
                f_log.write(','.join([str(packet_num),
                                      str(len(data) - 10),
                                      str(time.time())]) + '\n')

            except socket.timeout:
                printlog(f"Info: radar {radar_name} timedout",
                         fileName=fileName_log)
                break
            except Exception as e:
                printlog(f"Error: radar {radar_name} {e}",
                         fileName=fileName_log)
                break

        f_data.close()
        f_log.close()


class _Motor(Motor):
    """Motor subclass with simplified move interface."""

    def move(self, *argv):
        """Move motor by (dx, dy) or (dx, dy, dtrigger) in mm.

        Converts mm to steps and sends to Arduino stepper controller.
        """
        assert len(argv) == 2 or len(argv) == 3
        cmd = list(argv)[:2] + [0] + list(argv)[2:]
        msg = ','.join([str(int(x * self.stepper_ratio)) for x in cmd])
        self.serial_device.write(msg.encode())
        self.serial_device.readline()


class MasterDevice:
    """Orchestrates multi-radar 2D SAR data collection.

    Coordinates DCA1000 capture boards and motorized scanning stage
    for bistatic polarimetric SAR imaging.

    Parameters
    ----------
    cfg_collection_path : str
        Path to data collection configuration JSON.
    cfg_device_path : str
        Path to device communication configuration JSON.
    cfg_radar_path : str
        Path to radar ID configuration JSON.
    """

    def __init__(self, cfg_collection_path, cfg_device_path, cfg_radar_path):
        self.cfg_collection_path = cfg_collection_path
        self.cfg_device_path = cfg_device_path

        self.cfg_collection = json.load(open(cfg_collection_path))
        self.cfg_device = json.load(open(cfg_device_path))
        self.cfg_radar = json.load(open(cfg_radar_path))

        self.motor = _Motor(
            serial_port=self.cfg_device['comm_cfg']['motor']['serial_port'],
            baud_rate=self.cfg_device['comm_cfg']['motor']['baud_rate'],
            timeout=self.cfg_device['comm_cfg']['motor']['timeout'],
            stepper_ratio=self.cfg_device['comm_cfg']['motor']['stepper_ratio'])
        # Wait for Arduino to initialize
        time.sleep(2)

        self.flag_terminate = False
        self.isReady = False
        self.threads_status = [False] * len(self.cfg_device['activated_radar'])
        self.flag_re_trigger = False

    def _exit(self):
        self.flag_terminate = True
        self.isReady = False

    def printlog(self, *str_to_print, fileName=None, verbose=True):
        if fileName is None:
            fileName = self.fileName_collection_log
        printlog(*str_to_print, fileName=fileName, verbose=verbose)

    def capture(self, radar_cfg_dict):
        """Radar capturing thread for a single DCA1000 board."""
        if not os.path.exists(radar_cfg_dict['saveDir']):
            os.mkdir(radar_cfg_dict['saveDir'])

        dca = _DCA1000(dca_ip=radar_cfg_dict['ip'],
                        data_port=radar_cfg_dict['data_port'],
                        config_port=radar_cfg_dict['cfg_port'])
        dca.configure()

        saveName = radar_cfg_dict['saveName']
        self.printlog('radar', radar_cfg_dict['name'], saveName)

        self.printlog(radar_cfg_dict['name'], 'started',
                      dca._send_command(radar.CMD.RECORD_START_CMD_CODE).hex())
        dca.record(saveName, radar_cfg_dict['name'],
                   timeout=radar_cfg_dict['timeout'],
                   zero_filling=self.cfg_collection['packet_zero_filling'],
                   fileName_log=self.fileName_collection_log)
        self.printlog(radar_cfg_dict['name'], 'stopped',
                      dca._send_command(radar.CMD.RECORD_STOP_CMD_CODE).hex())

    def start_collection(self):
        """Run 2D raster-scan data collection."""
        if self.cfg_collection['initial_position_pointer'] != 0:
            if input('pos_ptr != 0?') != 'yes':
                return

        masterDir = os.path.join(
            self.cfg_collection['saving_root_dir'],
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.fileName_collection_log = os.path.join(masterDir, 'collection.log')
        if not os.path.exists(masterDir):
            os.mkdir(masterDir)

        self.printlog('log location: ', self.fileName_collection_log)
        self.printlog('master directory: ', masterDir)

        # Save configs to collection directory
        json.dump(self.cfg_collection,
                  open(os.path.join(masterDir, 'data_collection_cfg.json'), "w"),
                  indent=4)
        json.dump(self.cfg_device,
                  open(os.path.join(masterDir, 'devices_comm_cfg.json'), "w"),
                  indent=4)
        json.dump(self.cfg_radar,
                  open(os.path.join(masterDir, 'radar_id.json'), "w"),
                  indent=4)

        if "retrigger_list" in self.cfg_collection:
            self.flag_re_trigger = True

        path_ary = MasterDevice.generate_path_ary(
            self.cfg_collection['total_x'],
            self.cfg_collection['total_y'],
            self.cfg_collection['moving_direction'])
        pos_ptr = self.cfg_collection['initial_position_pointer']
        curr_pos = path_ary[pos_ptr]
        prev_pos = path_ary[pos_ptr]
        self.curr_name = curr_pos

        dx = self.cfg_collection['dx']
        dy = self.cfg_collection['dy']
        if self.cfg_collection['moving_direction'] == 'row':
            self.cfg_collection['dstep'] = self.cfg_collection['dx']
        else:
            self.cfg_collection['dstep'] = self.cfg_collection['dy']
        dstep = self.cfg_collection['dstep']

        if 'packet_zero_filling' not in self.cfg_collection:
            self.cfg_collection['packet_zero_filling'] = True
        self.printlog('zero filling: ', self.cfg_collection['packet_zero_filling'])

        self.radar_configs = []
        for thread_idx, radar_idx in enumerate(self.cfg_device['activated_radar']):
            self.radar_configs.append(self.cfg_radar[str(radar_idx)].copy())
            self.radar_configs[-1]['saveDir'] = os.path.join(
                masterDir, self.radar_configs[-1]['name'])
            self.radar_configs[-1]['saveName'] = os.path.join(
                self.radar_configs[-1]['saveDir'],
                '_'.join([str(x) for x in self.curr_name]) + '.bin')
            self.radar_configs[-1]['timeout'] = self.cfg_collection['radar_timeout']

        flag_trigger = False
        t_start = time.time()
        t_prev = time.time()
        all_threads = []

        while pos_ptr < len(path_ary):
            curr_pos = path_ary[pos_ptr]
            self.printlog('pos_ptr', pos_ptr, 'curr_loc', curr_pos,
                          'prev_loc', prev_pos)

            if (curr_pos == prev_pos).all():
                pos_ptr += 1
                continue

            flag_trigger = (pos_ptr % 2 == 1)

            if flag_trigger:
                # Recording: moving along major axis
                for radar_cfg in self.radar_configs:
                    radar_cfg['saveName'] = os.path.join(
                        radar_cfg['saveDir'],
                        '_'.join([str(x) for x in self.curr_name]) + '.bin')
                    all_threads.append(
                        Thread(target=self.capture, args=(radar_cfg,)))

                for t in all_threads:
                    t.start()

                self.motor.move((curr_pos[0] - prev_pos[0]) * dx,
                                (curr_pos[1] - prev_pos[1]) * dy, dstep)
                self.isReady = False

                t_now = time.time()
                t_duration = t_now - t_start
                t_diff = t_now - t_prev
                t_prev = t_now
                self.printlog(f'last axis time: {t_diff:.2f}')
                self.printlog('past time: {:.0f} hr {:.0f} min {:.1f} s'.format(
                    t_duration // 3600, t_duration % 3600 // 60, t_duration % 60))

            else:
                # Saving: moving along non-major axis
                self.motor.move((curr_pos[0] - prev_pos[0]) * dx,
                                (curr_pos[1] - prev_pos[1]) * dy)

                t_wait_for_receiving = time.time()
                time.sleep(2)

                if self.flag_re_trigger:
                    retrigger_thread = threading.Thread(
                        target=self.retrigger(), daemon=False)
                    all_threads.append(retrigger_thread)
                    retrigger_thread.start()

                for t in all_threads:
                    t.join()
                all_threads = []

                t_wait_for_stablizing = (self.cfg_collection['vibration_pd']
                                         - (time.time() - t_wait_for_receiving))
                if t_wait_for_stablizing > 0:
                    time.sleep(t_wait_for_stablizing)

                self.curr_name = curr_pos

            pos_ptr += 1
            prev_pos = curr_pos
            self.printlog()

        # After collection
        time.sleep(self.cfg_collection['vibration_pd'])
        self._exit()
        self.motor.move(-curr_pos[0] * dx, -curr_pos[1] * dy)
        self.printlog('collection done.')

    def retrigger(self):
        """Notify MATLAB instances to retrigger radar capture via TCP."""
        def notify_matlab(matlab_ip, port, message="Retrigger\n"):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
                    client.setsockopt(socket.IPPROTO_TCP,
                                      socket.TCP_NODELAY, 1)
                    client.connect((matlab_ip, port))
                    client.sendall(message.encode())
                    client.close()
            except Exception as e:
                print(f"Error: {e}")

        retrigger_list = self.cfg_collection["retrigger_list"]
        for dname in retrigger_list:
            _ip = self.cfg_device['comm_cfg'][dname]['IP']
            _port = self.cfg_device['comm_cfg'][dname]['server_access_port']
            notify_matlab(_ip, _port)
        self.printlog('retriggered')

    def mainthread(self):
        """Run collection with error handling."""
        try:
            self.start_collection()
        except (KeyboardInterrupt, SystemExit):
            self.printlog('KeyboardInterrupt')
            self._exit()
        except Exception as e:
            self.printlog('Error: ', e)
            self._exit()

    def move_ptr(self, curr_ptr, tar_ptr):
        """Move motor from one path position to another."""
        total_x = self.cfg_collection['total_x']
        total_y = self.cfg_collection['total_y']
        dx = self.cfg_collection['dx']
        dy = self.cfg_collection['dy']
        axis = self.cfg_collection['moving_direction']
        path_ary = MasterDevice.generate_path_ary(total_x, total_y, axis)
        curr_pos = path_ary[curr_ptr]
        tar_pos = path_ary[tar_ptr]

        print('target position: ', tar_pos)
        print('current position: ', curr_pos)
        if input('moving stage?') == 'yes':
            self.motor.move((tar_pos[0] - curr_pos[0]) * dx,
                            (tar_pos[1] - curr_pos[1]) * dy)

    @staticmethod
    def generate_path_ary(total_x, total_y, mode='row'):
        """Generate 2D raster-scan path array.

        Returns start/end positions for each row/column for continuous
        motion during data collection.

        Parameters
        ----------
        total_x, total_y : int
            Grid dimensions.
        mode : str
            'row' or 'col' for scan direction.

        Returns
        -------
        ndarray, shape [2*total_y, 2] or [2*total_x, 2]
        """
        if mode == 'row':
            x_list = [[0, total_x - 1] if y % 2 == 0
                       else [total_x - 1, 0] for y in range(total_y)]
            y_list = [[y, y] for y in range(total_y)]
        elif mode == 'col':
            x_list = [[x, x] for x in range(total_x)]
            y_list = [[0, total_y - 1] if x % 2 == 0
                       else [total_y - 1, 0] for x in range(total_x)]

        flatten = lambda ary: [x for subary in ary for x in subary]
        return np.array(list(zip(flatten(x_list), flatten(y_list))))

    @staticmethod
    def generate_path_ary_step(total_x, total_y, mode='row'):
        """Generate step-wise 2D path array (one position per step)."""
        if mode == 'row':
            x_list = [list(range(total_x)) if y % 2 == 0
                       else list(reversed(range(total_x)))
                       for y in range(total_y)]
            y_list = [[y] * total_x for y in range(total_y)]
        elif mode == 'col':
            x_list = [[x] * total_y for x in range(total_x)]
            y_list = [list(range(total_y)) if x % 2 == 0
                       else list(reversed(range(total_y)))
                       for x in range(total_x)]

        flatten = lambda ary: [x for subary in ary for x in subary]
        return np.array(list(zip(flatten(x_list), flatten(y_list))))
