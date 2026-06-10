"""Decode raw ADC binary data into numpy arrays.

Usage:
    python scripts/decode.py --data-dir /path/to/raw/collection/
"""

import argparse
from polysight.collection.decode import main_decode


def main():
    parser = argparse.ArgumentParser(description='Decode raw ADC data')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to raw collection directory')
    args = parser.parse_args()
    main_decode(args.data_dir)


if __name__ == '__main__':
    main()
