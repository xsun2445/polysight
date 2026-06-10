"""Antenna location definitions for AWR2243 boards and frame geometry.

Defines the physical antenna positions for the polarimetric bistatic SAR
setup with tilted RX boards.
"""

import scipy.constants
import numpy as np
from scipy.spatial.transform import Rotation

C = scipy.constants.speed_of_light
PI = scipy.constants.pi


def awr2243AntLoc():
    """Locations of antennas on awr2243boost board: [rx1-rx4, tx1-tx3].

    Returns:
        ndarray: shape [7, 3] representing xyz locations of all 7 antennas.
    """
    lambda_f = 1.9e-3 * 2
    gap_rxtx = 5.2448e-3

    rx_locs = np.array([
        [0,              0, 0],
        [lambda_f / 2,   0, 0],
        [lambda_f,       0, 0],
        [lambda_f / 2 * 3, 0, 0]])

    tx_locs = np.array([
        [lambda_f / 2 * 3 + gap_rxtx,              0,            0],
        [lambda_f / 2 * 3 + gap_rxtx + lambda_f,   lambda_f / 2, 0],
        [lambda_f / 2 * 3 + gap_rxtx + lambda_f * 2, 0,          0]])

    return np.vstack((rx_locs, tx_locs))


def defineAntLoc_frame_v6(txboard_offset, txboard_angle_y, rxboard_offset,
                           rxboard_v_angle_y, rxboard_angle_y, rxboard_angle_z,
                           rxboard_tilt_offset):
    """Antenna locations for frame v6 (RX boards tilted 22.5 degrees).

    Coordinate system (facing the setup):
    - x pointing towards sky
    - y pointing right side
    - z pointing outside
    """
    upleft_hole_to_rx1_offset = np.vstack([15.837, -3.350, 0]) * 1e-3
    txboard_origin_to_hole_offset = np.vstack([8.623, 120.069, 36.333]) * 1e-3 + np.vstack([0, 0, 10]) * 1e-3

    ant_locs_ori = awr2243AntLoc().T

    txboard_angle_z = -45
    txboard_ant_locs = (Rotation.from_euler('z', txboard_angle_z, degrees=True).as_matrix()
                        @ (ant_locs_ori + upleft_hole_to_rx1_offset))
    txboard_ant_locs = (Rotation.from_euler('y', txboard_angle_y, degrees=True).as_matrix()
                        @ (txboard_ant_locs + txboard_origin_to_hole_offset))
    txboard_offset = np.vstack([txboard_offset, 0, 0])
    txboard_ant_locs += txboard_offset

    rxboard_origin_to_h_hole_offset = np.vstack([-56.581, 96.552, 36.333]) * 1e-3
    rxboard_origin_to_v_hole_offset = np.vstack([-31.971, 105.052, 36.333]) * 1e-3

    rxboard_h_ant_locs = ant_locs_ori + upleft_hole_to_rx1_offset + rxboard_origin_to_h_hole_offset

    rxboard_v_angle_z = 90
    rxboard_v_ant_locs = (Rotation.from_euler('z', rxboard_v_angle_z, degrees=True).as_matrix()
                          @ (ant_locs_ori + upleft_hole_to_rx1_offset))
    rxboard_v_ant_locs = (Rotation.from_euler('y', rxboard_v_angle_y, degrees=True).as_matrix()
                          @ rxboard_v_ant_locs)
    rxboard_v_ant_locs += rxboard_origin_to_v_hole_offset

    rxboard_h_ant_locs = Rotation.from_euler('z', rxboard_angle_z, degrees=True).as_matrix() @ rxboard_h_ant_locs
    rxboard_v_ant_locs = Rotation.from_euler('z', rxboard_angle_z, degrees=True).as_matrix() @ rxboard_v_ant_locs

    rxboard_tilt_offset = np.vstack(rxboard_tilt_offset) * 1e-3
    rxboard_h_ant_locs += rxboard_tilt_offset
    rxboard_v_ant_locs += rxboard_tilt_offset

    rxboard_h_ant_locs = Rotation.from_euler('y', rxboard_angle_y, degrees=True).as_matrix() @ rxboard_h_ant_locs
    rxboard_v_ant_locs = Rotation.from_euler('y', rxboard_angle_y, degrees=True).as_matrix() @ rxboard_v_ant_locs

    rxboard_offset = np.vstack([rxboard_offset, 0, 0])
    rxboard_h_ant_locs += rxboard_offset
    rxboard_v_ant_locs += rxboard_offset

    return np.hstack([txboard_ant_locs, rxboard_h_ant_locs, rxboard_v_ant_locs]).T


def defineAntLoc_frame_v5(txboard_offset, txboard_angle_y, rxboard_offset,
                           rxboard_v_angle_y, rxboard_angle_y):
    """Antenna locations for frame v5 (non-tilted RX boards)."""
    upleft_hole_to_rx1_offset = np.vstack([15.837, -3.350, 0]) * 1e-3
    txboard_origin_to_hole_offset = np.vstack([8.623, 120.069, 36.333]) * 1e-3 + np.vstack([0, 0, 10]) * 1e-3

    ant_locs_ori = awr2243AntLoc().T

    txboard_angle_z = -45
    txboard_ant_locs = (Rotation.from_euler('z', txboard_angle_z, degrees=True).as_matrix()
                        @ (ant_locs_ori + upleft_hole_to_rx1_offset))
    txboard_ant_locs = (Rotation.from_euler('y', txboard_angle_y, degrees=True).as_matrix()
                        @ (txboard_ant_locs + txboard_origin_to_hole_offset))
    txboard_offset = np.vstack([txboard_offset, 0, 0])
    txboard_ant_locs += txboard_offset

    rxboard_origin_to_h_hole_offset = np.vstack([-56.581, 96.552, 36.333]) * 1e-3
    rxboard_origin_to_v_hole_offset = np.vstack([-31.971, 105.052, 36.333]) * 1e-3

    rxboard_h_ant_locs = ant_locs_ori + upleft_hole_to_rx1_offset + rxboard_origin_to_h_hole_offset

    rxboard_v_angle_z = 90
    rxboard_v_ant_locs = (Rotation.from_euler('z', rxboard_v_angle_z, degrees=True).as_matrix()
                          @ (ant_locs_ori + upleft_hole_to_rx1_offset))
    rxboard_v_ant_locs = (Rotation.from_euler('y', rxboard_v_angle_y, degrees=True).as_matrix()
                          @ rxboard_v_ant_locs)
    rxboard_v_ant_locs += rxboard_origin_to_v_hole_offset

    rxboard_h_ant_locs = Rotation.from_euler('y', rxboard_angle_y, degrees=True).as_matrix() @ rxboard_h_ant_locs
    rxboard_v_ant_locs = Rotation.from_euler('y', rxboard_angle_y, degrees=True).as_matrix() @ rxboard_v_ant_locs

    rxboard_offset = np.vstack([rxboard_offset, 0, 0])
    rxboard_h_ant_locs += rxboard_offset
    rxboard_v_ant_locs += rxboard_offset

    return np.hstack([txboard_ant_locs, rxboard_h_ant_locs, rxboard_v_ant_locs]).T


def defineAntLoc(angle_azi_l, angle_azi_r, angle_tilt_l, angle_tilt_r, dist_l, dist_r):
    """Define antenna locations for all 4 radars (legacy frame).

    Returns:
        ndarray: shape [num_ant, 3]
    """
    hv_board_offset = np.vstack([31 * 1e-3, 8.5 * 1e-3, 0])
    hv_board_angle = 90
    board_hole_ant_offset = np.vstack([15.109698e-3, -1.775206e-3, 0])
    radar_z = 53 * 1e-3
    origin_hole_offset = np.vstack([-31e-3, 96.2e-3, radar_z])
    offset_origin_tilt_l = np.vstack([91.2e-3, 0, 0])
    offset_origin_tilt_r = np.vstack([0, 0, 0])

    ant_locs = awr2243AntLoc().T
    ant_locs += board_hole_ant_offset
    ant_locs_v = Rotation.from_euler('z', hv_board_angle, degrees=True).as_matrix() @ ant_locs
    ant_locs = np.hstack((ant_locs, ant_locs_v + hv_board_offset)) + origin_hole_offset

    rotmat_azi_l = Rotation.from_euler('y', angle_azi_l, degrees=True).as_matrix()
    rotmat_tilt_l = Rotation.from_euler('z', angle_tilt_l, degrees=True).as_matrix()
    ant_locs_l = rotmat_tilt_l @ (rotmat_azi_l @ ant_locs - offset_origin_tilt_l) + offset_origin_tilt_l
    ant_locs_l += np.vstack([dist_l, 0, 0])

    rotmat_tilt_r = Rotation.from_euler('z', angle_tilt_r, degrees=True).as_matrix()
    rotmat_azi_r = Rotation.from_euler('y', angle_azi_r, degrees=True).as_matrix()
    ant_locs_r = rotmat_tilt_r @ (rotmat_azi_r @ ant_locs - offset_origin_tilt_r) + offset_origin_tilt_r
    ant_locs_r += np.vstack([dist_r, 0, 0])

    ant_locs = np.hstack((ant_locs_l, ant_locs_r))
    ant_locs += np.vstack([0, -np.mean(ant_locs[1, :]), 0])

    return ant_locs.T


def getAntLoc(SAR_config, tx_idx, rx_idx):
    """Get tx and rx xyz locations (0-indexed).

    Returns:
        list: [tx_loc, rx_loc], each shape [1, 3]
    """
    if 'angle_azi_l' not in SAR_config:
        ant_locs = defineAntLoc(0, 0, -45, 0, -0.43, 0.4)
    else:
        ant_locs = defineAntLoc(SAR_config['angle_azi_l'],
                                SAR_config['angle_azi_r'],
                                SAR_config['angle_tilt_l'],
                                SAR_config['angle_tilt_l'],
                                SAR_config['dist_l'],
                                SAR_config['dist_r'])
    num_tx, num_rx = 3, 4
    tx_idx = (tx_idx // num_tx) * (num_tx + num_rx) + num_rx + tx_idx % num_tx
    rx_idx = (rx_idx // num_rx) * (num_tx + num_rx) + rx_idx % num_rx
    return [ant_locs[tx_idx, :], ant_locs[rx_idx, :]]


def isSameRadar(tx_idx, rx_idx):
    """Check if tx and rx belong to the same radar board."""
    return tx_idx // 3 == rx_idx // 4
