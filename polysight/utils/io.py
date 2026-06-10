"""Data loading and I/O helpers."""

import numpy as np
import json
import os


def load_label_json(collection_name, labels_dir):
    """Load a collection's label JSON file."""
    config_path = os.path.join(labels_dir, collection_name, f'{collection_name}.json')
    with open(config_path, 'r') as f:
        return json.load(f)


def load_sar_pair(collection_name, labels_dir):
    """Load pre-saved sar_h.npy and sar_v.npy for a collection.

    Returns:
        (sar_h, sar_v): Complex SAR image arrays.
    """
    folder = os.path.join(labels_dir, collection_name)
    sar_h = np.load(os.path.join(folder, 'sar_h.npy'))
    sar_v = np.load(os.path.join(folder, 'sar_v.npy'))
    return sar_h, sar_v


def save_sar(sar, radar_name, collection_name, labels_dir):
    """Save a SAR image to the labels directory."""
    radar_to_filename = {'LS': 'sar_l.npy', 'RH': 'sar_h.npy', 'RV': 'sar_v.npy'}
    folder = os.path.join(labels_dir, collection_name)
    os.makedirs(folder, exist_ok=True)
    save_path = os.path.join(folder, radar_to_filename[radar_name])
    np.save(save_path, sar)
    return save_path
