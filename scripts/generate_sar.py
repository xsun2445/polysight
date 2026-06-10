"""Generate SAR images from labeled collections.

Usage:
    python scripts/generate_sar.py --collection 20250423_181731_multi --labels-dir /path/to/labels/
    python scripts/generate_sar.py --collection 20250423_181731_multi --labels-dir /path/to/labels/ --save
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from polysight.dsp.sar import calc_sar_fromfile
from polysight.utils.io import save_sar
from polysight.eval.visualization import visualize_sar


def main():
    parser = argparse.ArgumentParser(description='Generate SAR images')
    parser.add_argument('--collection', type=str, required=True,
                        help='Collection name')
    parser.add_argument('--labels-dir', type=str, required=True,
                        help='Labels directory path')
    parser.add_argument('--radars', type=str, nargs='+', default=['RH', 'RV'],
                        help='Radar names to process')
    parser.add_argument('--save', action='store_true',
                        help='Save SAR images as .npy files')
    parser.add_argument('--sum-bins', action='store_true',
                        help='Coherently sum range bins')
    parser.add_argument('--chunk-size', type=int, default=0,
                        help='GPU chunk size (0=all at once)')
    args = parser.parse_args()

    label_dict, sar_list = calc_sar_fromfile(
        args.collection, args.radars,
        sum_bins=args.sum_bins, chunk_size=args.chunk_size,
        labels_dir=args.labels_dir)

    if args.save:
        for radar_name, sar in zip(args.radars, sar_list):
            path = save_sar(sar, radar_name, args.collection, args.labels_dir)
            print(f'Saved: {path}')

    labels = label_dict.get('labels', None)
    fig = visualize_sar(sar_list, args.radars, labels=labels,
                        title=args.collection)
    plt.show()


if __name__ == '__main__':
    main()
