import csv
import json
import os

import datasets
import pandas as pd
import numpy as np

ds = datasets.load_dataset('wit_dataset_script.py', data_dir='train_data')
test_ds = ds['train']
print(ds)

def transform(example):

    example['pixel_values'] = np.load(example['pixels_file'])
    return example
