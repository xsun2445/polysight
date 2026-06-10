# PolySight Dataset

Polarimetric bistatic SAR dataset collected with a 77 GHz FMCW radar (TI AWR2243).
Two radar boards receive orthogonal polarizations (RH, RV) while a single TX board
illuminates the scene. A 2-D motorized stage rasters the target across the aperture.

## Directory Structure

```
data/
├── unsynced/          # Raw unsynchronized ADC captures (2 example collections)
│   └── <collection>/
│       ├── RH/                   # Raw DCA1000 captures, H-pol RX (.bin + _log.csv)
│       ├── RV/                   # Raw DCA1000 captures, V-pol RX (.bin + _log.csv)
│       ├── LS/                   # Raw DCA1000 captures, TX (.bin + _log.csv)
│       ├── _adcData/RH/          # Decoded per-frame .npy files, H-pol
│       ├── _adcData/RV/          # Decoded per-frame .npy files, V-pol
│       ├── adcData/RH/           # Synchronized per-frame .npy files, H-pol
│       ├── adcData/RV/           # Synchronized per-frame .npy files, V-pol
│       ├── collection.log
│       ├── data_collection_cfg.json
│       ├── devices_comm_cfg.json
│       └── radar_id.json
│
├── raw/               # Synchronized radar cubes (70 collections, ~72 GB, ~1 GB each)
│   └── <collection>/
│       ├── RH_synced.npy         # Synchronized ADC cube, H-pol receiver
│       ├── RV_synced.npy         # Synchronized ADC cube, V-pol receiver
│       ├── data_collection_cfg.json
│       ├── devices_comm_cfg.json
│       └── radar_id.json
│
├── labels/            # SAR images + bounding-box labels (85 collections)
│   └── <collection>/
│       ├── sar_h.npy             # Complex SAR image, H-pol (150x150, complex64)
│       ├── sar_v.npy             # Complex SAR image, V-pol (150x150, complex64)
│       └── <collection>.json     # Label file (bounding boxes + SAR config)
│
├── materials/                    # Cropped material samples (see below)
│   ├── ceramic_60.pkl
│   ├── water_55.pkl
│   └── ...
│
└── README.md
```

## Download

The polysight dataset is hosted on Hugging face: https://huggingface.co/datasets/xinghs/polysight

Files are downloaded into the current directory, preserving the original folder structure.

| Folder       | Size    | Contents                                  |
|--------------|---------|-------------------------------------------|
| `raw/`       | ~72 GB  | 70 synchronized radar cubes (~1 GB each)  |
| `labels/`    | ~66 MB  | 84 SAR image collections with labels      |
| `materials/` | ~1 MB   | 72 cropped material sample files (.pkl)   |
| `unsynced/`  | ~35 GB  | 2 example raw unsynchronized ADC captures |



Use `download.py` to selectively download subfolders, install the Hugging Face Hub client:

```bash
pip install huggingface_hub
```

```bash
# List all available folders
python download.py --list

# Download a top-level folder
python download.py materials

# Download specific collections
python download.py labels/20250603_185728_oil_water labels/20250605_194155_water

# Download with glob patterns
python download.py "labels/20250625*"

# Download multiple folders at once
python download.py materials labels raw

# Download everything
python download.py --all
```


---




## Pipeline

```
unsynced/  ──[decode + sync]──>  raw/  ──[SAR generation]──>  labels/  ──[crop]──>  materials/
```

### 1. `unsynced/` - Raw ADC Captures

Raw ADC frames captured by three DCA1000 FPGA boards over Ethernet. Each radar
board produces independent packet streams that are **not yet time-aligned**.
These are provided as examples for reproducing the full decode and synchronization
pipeline (`polysight.collection.decode`). Most users can skip this stage and start
from `raw/`.

### 2. `raw/` - Synchronized Radar Cubes

Time-synchronized 3-D ADC data cubes, one per polarization channel:

- **`RH_synced.npy`** - Horizontal polarization receiver
- **`RV_synced.npy`** - Vertical polarization receiver

Each file is a NumPy array of shape `(num_positions, num_antennas, num_adc_samples)`
containing the decoded, packet-loss-corrected, and cross-board synchronized IF signal.
Generated from `unsynced/` data using `scripts/decode.py`.

### 3. `labels/` - SAR Images + Labels

SAR images generated from `raw/` cubes via GPU-accelerated FFT back-projection
(`scripts/generate_sar.py`, `notebooks/02_sar_generation.ipynb`).

Each collection folder contains:

- **`sar_h.npy`** / **`sar_v.npy`** - Complex-valued SAR images (150x150, complex64)
- **`<collection>.json`** - Metadata and bounding-box labels, created with
  `notebooks/03_labeling.ipynb`

The JSON label file contains:

```json
{
  "description": "1tx, oil, water, copper",
  "labels": {
    "oil":    [x0, y0, x1, y1],
    "water":  [x0, y0, x1, y1],
    "copper": [x0, y0, x1, y1]
  },
  "sar_config": {
    "numX": 350, "numY": 200,
    "dx": 0.002, "dy": 0.002,
    "output_size_x": 150, "output_size_y": 150,
    "chirp_configs": { "f0": 76.01e9, "slope": 38.8e12, ... },
    "antenna_configs": { ... },
    "ant_locs": [[x,y,z], ...],
    "RH_config": { "alignment_configs": { ... }, ... },
    "RV_config": { ... }
  }
}
```

### 4. `materials/` - Cropped Material Samples

Pickled dictionaries containing cropped SAR pixel regions for individual materials,
extracted from labeled SAR images. Each `.pkl` file represents one material measured
at a specific incident angle. Used by the evaluation notebooks for permittivity
estimation and material classification.

**File naming**: `<material>_<angle>.pkl` (e.g., `ceramic_60.pkl`, `water_55.pkl`)

**Structure of each `.pkl` file**:

```python
{
    "name":           str,        # Material name (e.g., "ceramic")
    "thickness":      float,      # Sample thickness in meters (e.g., 0.00127)
    "collection_name": str,       # Source collection (e.g., "20250623_182841_multi")
    "incident_angle": float,      # Measurement angle in degrees (e.g., 60.0)
    "ref_name":       str,        # Reference material name (e.g., "copper")
    "ref_thickness":  float,      # Reference thickness in meters
    "values":         list[ndarray],  # 3 complex64 arrays: [H/V ratio, H, V]
                                      # each shaped (rows, cols) from bounding box
    "ref_values":     list[ndarray],  # Same structure for the reference material
}
```

The `values` and `ref_values` lists each contain 3 arrays corresponding to the
H/V polarimetric ratio, H-pol, and V-pol SAR pixel regions cropped from the
bounding box. The polarimetric ratio (element 0) is used with Fresnel equations
to estimate relative permittivity via `polysight.eval.permittivity.ratio_to_epsilon`.

## Collection Categories

| Category      | Count | Description                                      |
|---------------|------:|--------------------------------------------------|
| liquids       |    11 | Oil, water, soy sauce, syrup, sugar-water series  |
| solids        |    24 | Plastics, wood, rubber, metals, composites        |
| roughness     |     3 | Wood grain orientations, sandpaper grits          |
| ablation      |     9 | Parameter sensitivity                             |
| concentration |     7 | Sugar-water concentration gradient (0-6 spoons)   |
| ceramics      |     8 | Alumina ceramic vs plastic at varying positions   |
| misc          |     8 | Oil, powder, rubber, PVC, mixed materials         |
