"""DCA1000 EVM board interface via Ethernet.

Provides socket-based communication with the DCA1000 FPGA for radar data
capture from AWR2243 mmWave sensors.
"""

import codecs
import socket
import struct
from enum import Enum
import numpy as np
import os
import time


# Socket buffersize
RECVBUFFSIZE = 2**30
# STATIC
MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456
# DYNAMIC
FILLING_PACKET = b'\x00' * BYTES_IN_PACKET
DELIMITER = "@$@$"


class CMD(Enum):
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'
    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    CONFIG_EEPROM_CMD_CODE = '0400'
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'
    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'
    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'

    def __str__(self):
        return str(self.value)


CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'


class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from

    Examples:
        >>> dca = DCA1000()
        >>> adc_data = dca.read(timeout=.1)
        >>> frame = dca.organize(adc_data, 128, 4, 256)
    """

    def __init__(self, system_ip='192.168.33.30', dca_ip='192.168.33.180',
                 data_port=4098, config_port=4096):
        self.cfg_dest = (dca_ip, config_port)
        self.cfg_recv = (system_ip, config_port)
        self.data_recv = (system_ip, data_port)

        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        self.data_socket.bind(self.data_recv)
        self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RECVBUFFSIZE)
        self.config_socket.bind(self.cfg_recv)

        self.data = []
        self.packet_count = []
        self.byte_count = []
        self.frame_buff = []
        self.curr_buff = None
        self.last_frame = None
        self.lost_packets = None

    def configure(self):
        """Initializes and connects to the FPGA."""
        resp = self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE)
        print(resp.hex())
        print(self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE).hex())
        print(self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', '01010102031e').hex())
        print(self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c00537000000').hex())

    def __del__(self):
        print(self._send_command(CMD.RECORD_STOP_CMD_CODE).hex())
        self.close()

    def close(self):
        """Closes the sockets."""
        self.data_socket.close()
        self.config_socket.close()

    def record(self, saveName, timeout=10):
        """Record raw ADC data to a binary file."""
        self.data_socket.settimeout(timeout)
        f_data = open(saveName, 'wb')
        logName = os.path.join(os.path.dirname(saveName),
                               os.path.basename(saveName).split('.')[0] + '_log.csv')
        f_log = open(logName, 'w')

        cnt_byte = 0
        prev_seq = 0

        while True:
            try:
                data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
                packet_num = struct.unpack('<1l', data[:4])[0]
                cnt_byte += len(data) - 10

                if packet_num - prev_seq > 1:
                    print(f'Missing packet: {packet_num - prev_seq}')

                prev_seq += 1
                while prev_seq < packet_num:
                    print('missing packet!')
                    data = FILLING_PACKET + data
                    prev_seq += 1

                if packet_num % 500 == 0:
                    print(packet_num)

                f_data.write(data[10:])
                f_log.write(','.join([str(packet_num), str(len(data) - 10), str(time.time())]) + '\n')

            except Exception as e:
                print(f"Error: {e}")
                f_data.close()
                f_log.close()
                break

    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Send a single command to the FPGA."""
        self.config_socket.settimeout(timeout)
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        return resp

    def _read_data_packet(self):
        """Read a single ADC packet via UDP."""
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        return packet_num, byte_count, packet_data

    def _stop_stream(self):
        """Send the stop command to the FPGA."""
        return self._send_command(CMD.RECORD_STOP_CMD_CODE)

    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples):
        """Reorganize raw ADC data into a full frame.

        Returns:
            ndarray: shape (num_chirps, num_rx, num_samples)
        """
        ret = np.zeros(len(raw_frame) // 2, dtype=complex)
        ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
        ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
        return ret.reshape((num_chirps, num_rx, num_samples))

    @staticmethod
    def close_mmwave_control():
        import re
        try:
            x = os.popen("netstat -ano | findstr 0.0.0.0:4096").read()
            xx = re.findall(r'\d+', x)
            print(x)
            cmdstr = "taskkill -PID " + str(xx[5]) + " -F"
            x = os.popen(cmdstr).read()
            print(x)
            print("mmWavestudio spy control closed successfully!")
        except Exception:
            print("no mmWavestudio spy control")

    @staticmethod
    def config_eeprom(dca_ip='192.168.33.180', dca_port=4096,
                      system_ip='192.168.33.30',
                      DCA1000IPAddress='192.168.33.180',
                      DCA1000MACAdress='12.34.56.78.90.12',
                      DCA1000ConfigPort=4096, DCA1000DataPort=4098,
                      systemIPAdress='192.168.33.30'):
        cmd_code = CMD.CONFIG_EEPROM_CMD_CODE
        len_code = '1200'
        body_code = []
        body_code.append(''.join([f'{int(x):02x}' for x in systemIPAdress.split('.')[::-1]]))
        body_code.append(''.join([f'{int(x):02x}' for x in DCA1000IPAddress.split('.')[::-1]]))
        body_code.append(''.join([f'{int(x):02x}' for x in DCA1000MACAdress.split('.')[::-1]]))
        body_code.append(''.join([f'{DCA1000ConfigPort:04x}'[2:], f'{DCA1000ConfigPort:04x}'[:2]]))
        body_code.append(''.join([f'{DCA1000DataPort:04x}'[2:], f'{DCA1000DataPort:04x}'[:2]]))
        body_code = ''.join(body_code)

        resp = DCA1000.send_command_code(cmd_code, length=len_code,
                                         body=body_code, system_ip=system_ip,
                                         dca_ip=dca_ip, config_port=dca_port)
        print(resp.hex())

    @staticmethod
    def send_command_code(cmd, length='0000', body='',
                          system_ip='192.168.33.30', dca_ip='192.168.30.180',
                          config_port=4096, data_port=4098, timeout=1):
        config_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        config_socket.bind((system_ip, config_port))
        config_socket.settimeout(timeout)
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            config_socket.sendto(msg, (dca_ip, config_port))
            resp, addr = config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        config_socket.close()
        return resp
