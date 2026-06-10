# Hardware

<p align="center">
  <img src="../assets/setup_figure_v2.png" width="600" alt="Measurement setup">
</p>

## Setup Overview

The PolySight hardware consists of three subsystems:

- **Radar**: 3 TI AWR2243 FMCW radar boards (76–81 GHz), each paired with a DCA1000 EVM for raw ADC capture. One board transmits (TX), two boards receive orthogonal polarizations (HRX, VRX). The TX and RX are separated by ~1.5 m in a bistatic configuration, with a copper shield isolating the TX antenna. Currently each radar requires a separate PC running TI mmWaveStudio for configuration and triggering.
- **Motion stage**: A 2-axis linear stage rasters the target across the radar aperture to synthesize a large SAR image. The vertical axis uses two stepper motors (left and right) for stability, and the horizontal axis uses one stepper motor. An Arduino controls all three motors via stepper drivers and sends a trigger pulse at each raster position.
- **Synchronization**: A Raspberry Pi 4B receives the trigger pulse from the Arduino (which corresponds to the SAR antenna locations on motion stage) and simultaneously triggers all three radar boards via hardware GPIO pins, ensuring time-aligned ADC captures across boards. Uses [WiringPi-Python-MultiPin](https://github.com/xsun2445/WiringPi-Python-MultiPin) for simultaneous multi-pin GPIO toggling.

## Directory Structure

```
hardware/
├── arduino/
│   └── motor_serialport_control.ino   # Arduino motor + trigger control
├── 3d_models/
│   ├── radar_frame_horizontal.stl     # Radar mounting frame (horizontal)
│   ├── radar_frame_22.5.stl           # Radar mounting frame (22.5° tilt)
│   ├── corner_reflector_30_1.5.stl    # Corner reflector (30 mm, 1.5 mm)
│   └── corner_reflector_50,1.5.stl    # Corner reflector (50 mm, 1.5 mm)
└── README.md
```

## Arduino Motor Control

The Arduino firmware (`motor_serialport_control.ino`) orchestrates the 2-D raster scan:

- **Vertical axis**: 2 stepper motors driven by the same step/direction signals for synchronized movement
- **Horizontal axis**: 1 stepper motor for lateral scanning
- **Trigger output**: GPIO pin 2 sends a pulse at each raster position to the Raspberry Pi
- **Serial interface**: Accepts commands from the host PC to configure scan parameters (speed, step count, grid size)

Pin assignments:

| Pin | Function |
|-----|----------|
| 2   | Trigger (to Raspberry Pi) |
| 3, 8 | Vertical left motor (PUL, DIR) |
| 4, 9 | Vertical right motor (PUL, DIR) |
| 5, 10 | Horizontal motor (PUL, DIR) |

## Raspberry Pi Synchronization

The Raspberry Pi 4B acts as the hardware synchronization bridge between the motion stage and the radar boards:

1. Listens for the trigger pulse from the Arduino (one pulse per raster position)
2. Simultaneously toggles multiple GPIO pins to trigger all three AWR2243BOOST boards to start ADC capture
3. Uses [WiringPi-Python-MultiPin](https://github.com/xsun2445/WiringPi-Python-MultiPin) to set multiple GPIO pins in a single register write, minimizing timing offsets between radar boards

## 3D Models

STL files for 3D-printed mounting hardware:

- **Radar frames**: Custom brackets for mounting AWR2243+DCA1000 boards at horizontal or 22.5° tilt angles
- **Corner reflectors**: Calibration targets for SAR image verification
