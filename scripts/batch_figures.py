"""Generate SAR figures for all labeled collections.

Usage:
    python scripts/batch_figures.py --labels-dir /path/to/labels/ --output-dir ./figures/
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from polysight.utils.io import load_label_json, load_sar_pair
from polysight.eval.visualization import make_sar_figure


# Dataset categories for batch processing
CATEGORIES = {
    'liquids': [
        '20250423_181731_multi', '20250423_184005_multi', '20250423_192153_multi',
        '20250423_201256_multi', '20250423_203729_multi', '20250423_213303_multi',
        '20250428_193747_multi', '20250428_200004_multi', '20250428_211714_multi',
        '20250428_214252_multi', '20250502_152636_multi', '20250502_155042_multi',
        '20250502_161455_multi', '20250502_163909_multi',
    ],
    'solids': [
        '20250424_170820_multi', '20250424_192022_multi', '20250424_195149_multi',
        '20250424_201556_multi', '20250424_203953_multi', '20250424_210333_multi',
        '20250424_213121_multi', '20250424_215653_multi', '20250424_222109_multi',
        '20250424_224524_multi', '20250424_230951_multi', '20250424_233406_multi',
        '20250425_000037_multi', '20250425_002412_multi', '20250425_004752_multi',
        '20250425_011231_multi', '20250425_013628_multi', '20250425_020138_multi',
        '20250425_022527_multi', '20250429_191434_multi', '20250429_195420_multi',
        '20250430_142809_multi', '20250430_145411_multi',
    ],
    'roughness': [
        '20250425_191254_multi', '20250425_193822_multi', '20250425_200445_multi',
        '20250425_202804_multi',
    ],
    'ablation': [
        '20250426_135718_multi', '20250426_142342_multi', '20250426_145543_multi',
        '20250426_155506_multi', '20250426_162239_multi', '20250426_174752_multi',
        '20250426_181403_multi', '20250426_184241_multi',
    ],
    'concentration': [
        '20250429_201634_multi', '20250429_204259_multi', '20250429_210903_multi',
        '20250429_213428_multi', '20250429_215919_multi',
    ],
    'ceramics': [
        '20250430_163416_multi', '20250430_170029_multi', '20250430_172710_multi',
        '20250430_175336_multi', '20250430_182204_multi', '20250430_184805_multi',
        '20250430_191454_multi', '20250430_200047_multi',
    ],
    'misc': [
        '20250502_135707_multi', '20250502_141637_multi', '20250502_143615_multi',
        '20250502_150014_multi', '20250502_170455_multi', '20250502_172931_multi',
        '20250502_175332_multi', '20250502_181903_multi',
    ],
}


def main():
    parser = argparse.ArgumentParser(description='Batch SAR figure generation')
    parser.add_argument('--labels-dir', type=str, required=True,
                        help='Labels directory path')
    parser.add_argument('--output-dir', type=str, default='./figures/',
                        help='Output directory for PNG figures')
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--categories', type=str, nargs='*', default=None,
                        help='Categories to process (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cats = args.categories if args.categories else list(CATEGORIES.keys())
    total = sum(len(CATEGORIES[c]) for c in cats)
    idx = 0

    for cat in cats:
        cat_dir = os.path.join(args.output_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)

        for cname in CATEGORIES[cat]:
            idx += 1
            print(f'[{idx}/{total}] {cat}/{cname}')
            try:
                label_dict = load_label_json(cname, args.labels_dir)
                sar_h, sar_v = load_sar_pair(cname, args.labels_dir)
                labels = label_dict.get('labels', {})
                subtitle = label_dict.get('description', '')
                title = f'{cat}: {cname}'

                fig = make_sar_figure(sar_h, sar_v, labels, title, subtitle,
                                      dpi=args.dpi)
                fig.savefig(os.path.join(cat_dir, f'{cname}.png'),
                            bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f'  ERROR: {e}')


if __name__ == '__main__':
    main()
