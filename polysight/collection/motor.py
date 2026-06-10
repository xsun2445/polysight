"""Stepper motor serial interface for 2D/3D motion stages."""

import serial


class Motor:
    """Motor controller using H-rail + trigger Arduino firmware."""

    def __init__(self, serial_port='COM3', baud_rate=9600, timeout=60, stepper_ratio=200):
        self.stepper_ratio = stepper_ratio
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_device = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=timeout)

    def __exit__(self):
        self.serial_device.close()

    def reconnect(self):
        self.serial_device.close()
        self.serial_device = serial.Serial(port=self.serial_port, baudrate=self.baud_rate,
                                           timeout=self.timeout)

    def move(self, *argv):
        """Move the stage by (dx, dy, [dtrigger]) in mm."""
        assert len(argv) == 2 or len(argv) == 3
        msg = ','.join([str(int(x * self.stepper_ratio)) for x in argv])
        self.serial_device.write(msg.encode())
        self.serial_device.readline()


class Motor_2d_using_3d(Motor):
    """2D motor adapter using 3 motors (one motor not used, d1d always 0)."""

    def move(self, *argv):
        """Move by (dx, dy, [dtrigger]) in mm, d1d fixed to 0."""
        assert len(argv) == 2 or len(argv) == 3
        d1d = 0
        cmd = list(argv)[:2] + [d1d] + list(argv)[2:]
        msg = ','.join([str(int(x * self.stepper_ratio)) for x in cmd])
        self.serial_device.write(msg.encode())
        self.serial_device.readline()
