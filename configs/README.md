# Configs

JSON configuration files for Polysight data collection and processing. Each file
defines one aspect of the acquisition setup and is loaded at runtime.



| File | Purpose |
| --- | --- |
| `data_collection_cfg.json` | Scan geometry and per-acquisition radar parameters. |
| `devices_comm_cfg.json` | Chirp profile and communication/triggering setup for all devices. |
| `radar_id.json` | Static identity map of each physical radar board. |

## `data_collection_cfg.json`

Defines the raster scan grid and frame timing for a capture session.

- **Scan grid** — `dx`/`dy` step size, `total_x`/`total_y` extent, `moving_direction`
  (`"row"`/`"column"`), and `initial_position_pointer`.
- **Frame timing** — `num_trigger`, `period_frame`, `num_frame`, `num_chirp`,
  `num_config`, `vibration_pd`, `radar_timeout`.
- **Data shape** — `num_adc` (samples per chirp), `num_ch` (RX channels).
- **Geometry / mounting** — board distances and angles (`board_dist`,
  `txboard_angle_y`, `rxboard_*_angle_*`, `rxboard_tilt_offset`, `object_distance`).
- **Run metadata** — `retrigger_list`, `saving_root_dir`, `description`,
  `packet_zero_filling`.

## `devices_comm_cfg.json`

Defines the radar waveform and how the host talks to each device.

- **`chirp_configs`** — FMCW waveform: `f0`, `slope`, `ramp_time`, `adc_start_time`,
  `num_adc`, `Fs`, plus `activated_tx` / `activated_rx`.
- **`comm_cfg`** — device count and a block per node (`master`, `slave_*`, `motor`,
  `trigger`) giving `radar_name`, `port_type`, IP/serial port, and access port.

## `radar_id.json`

Per-board static identity, keyed by board index: `name` (e.g. `RH`, `RV`, `LS`),
`ip`, `mac`, and `cfg_port` / `data_port`. The `radar_name` referenced in
`devices_comm_cfg.json` maps back to the `name` here.
