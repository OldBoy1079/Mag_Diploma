import os, random
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

random.seed(0)
current_dir = os.getcwd()
OUT_DIR = '../five_fold_utils'
os.makedirs(OUT_DIR, exist_ok=True)


def collect_brand(pkl_paths, all_car_dict):
    ind_cars, ood_cars = set(), set()
    for p in tqdm(pkl_paths):
        meta = torch.load(p, weights_only=False)[1]
        car = meta['car']
        (ind_cars if meta['label'] == '00' else ood_cars).add(car)
        all_car_dict.setdefault(car, []).append(p)

    def shuffled(s):
        lst = sorted(s)
        random.shuffle(lst)
        return lst

    ind_sorted, ood_sorted = shuffled(ind_cars), shuffled(ood_cars)
    print(f'  ind: {len(ind_sorted)}, ood: {len(ood_sorted)}')
    return ind_sorted, ood_sorted


brands = {
    'brand1': [f'{current_dir}/battery_brand1/train/*.pkl',
               f'{current_dir}/battery_brand1/test/*.pkl'],
    'brand2': [f'{current_dir}/battery_brand2/train/*.pkl',
               f'{current_dir}/battery_brand2/test/*.pkl'],
    'brand3': [f'{current_dir}/battery_brand3/data/*.pkl'],
}

all_car_dict = {}
for i, (name, patterns) in enumerate(brands.items(), start=1):
    print(f'Processing {name}...')
    files = [p for pat in patterns for p in glob(pat)]
    ind, ood = collect_brand(files, all_car_dict)
    np.save(f'{OUT_DIR}/ind_odd_dict{i}.npz',
            {'ind_sorted': ind, 'ood_sorted': ood})


np.save(f'{OUT_DIR}/all_car_dict.npz', all_car_dict)
print(f'Saved all_car_dict with {len(all_car_dict)} cars')