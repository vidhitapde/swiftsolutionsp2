import pandas as pd
import numpy as np
import math as math
import random
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import threading
import os

# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['x', 'y']     # label columns
    data = pd.read_csv(filename, sep=r'\s+', names=data_names)

    if not all (data.dtypes == 'float64'):
        raise Exception("Locations are not in float64 format.")

    return data

def main():
    print('\nComputeDronePath')


    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)

if __name__ == '__main__':
  main()