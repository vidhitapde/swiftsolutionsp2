import pandas as pd
import numpy as np
import math as math
import random
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['x', 'y']     # label columns
    data = pd.read_csv(filename, sep=r'\s+', names=data_names)

    if not all (data.dtypes == 'float64'):
        raise Exception("Locations are not in float64 format.")

    return data

# calculate euclidean distance to be used in creating distance matrix
def euclidean_distance(point1, point2):
    return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)

def create_distance_matrix(data):
    n = len(data)
    dist_matrix = np.zeros((n, n))   #N x N matrix of zeros
   
    for i in range(n):
        for j in range(i, n):
            dist_matrix[i, j] = dist_matrix[j, i] = euclidean_distance(data.iloc[i], data.iloc[j])

    return dist_matrix

def knn_clustering(data, k):
    cluster_centers = set()

    i = 0
    while len(cluster_centers) < k:
        index = random.randint(0, len(data))
        cluster_centers.add(index) 

    print(cluster_centers)

def main():
    print('\nComputeDronePath')


    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)

    # knn_clustering(data, 4)

if __name__ == '__main__':
  main()