"""Run 2D SAR data collection.

Usage:
    python scripts/collect.py --cfg-dir hardware/configs/
"""

import argparse
import os
from polysight.collection.collect import MasterDevice


def main():
    parser = argparse.ArgumentParser(description='Run 2D SAR data collection')
    parser.add_argument('--cfg-dir', type=str, default='configs/',
                        help='Directory containing configuration JSONs')
    args = parser.parse_args()

    cfg_dir = args.cfg_dir
    master = MasterDevice(
        cfg_collection_path=os.path.join(cfg_dir, 'data_collection_cfg.json'),
        cfg_device_path=os.path.join(cfg_dir, 'devices_comm_cfg.json'),
        cfg_radar_path=os.path.join(cfg_dir, 'radar_id.json'),
    )
    master.mainthread()


if __name__ == '__main__':
    main()
