import pandas as pd
import numpy as np
import math as math
import random
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
    if type(point1) == type(point2):    # both points are from dataframe
        return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)
    else:
        return math.sqrt((point1[0] - point2.iloc[0])**2 + (point1[1] - point2.iloc[1])**2)     # point2 is from dataframe, point1 is a coordinate in form [x,y]


def create_dist_matrix(data,cluster_centers):
    n = len(data)
    cluster_size = len(cluster_centers)
    dist_matrix = np.zeros((n, n))   #N x N matrix of zeros
    cluster_list = list(cluster_centers)

    for i in range(n):
        for j in range(cluster_size):
            dist_matrix[i, j] = dist_matrix[j, i] = euclidean_distance(data.iloc[j], data.iloc[i])

    return dist_matrix


def initialize_k_means_clusters(data, k):
    cluster_centers = set()
    i = 0
    while len(cluster_centers) < k:
        index = random.randint(0, len(data))
        cluster_centers.add(index)
    x = []
    y = []
    coord = []
    center_coords = []

    for point in cluster_centers:
        x = (data.iloc[point-1,0])
        y = (data.iloc[point-1,1])
        coord = [x, y]
        center_coords.append(coord)

    return cluster_centers, center_coords


def k_means_assign_and_updates(data, dist, center_coords, cluster_centers = None):
    n = dist.shape[0]
    locations = list(range(1,n+1))
    if cluster_centers is not None:
        updated_locations = [loc for loc in locations if loc not in cluster_centers]
    else:
        updated_locations = locations
    result_paths = [[num] for num in center_coords]

    for locs in updated_locations:
        smallest_distance = min(center_coords, key=lambda center:euclidean_distance(center,data.iloc[locs-1]))
        for center in result_paths:
            if smallest_distance == center[0]:
                center.append(locs)
    x = []
    y = []
    coord = []
    new_centers = []

    for cluster in result_paths:
        first_point = cluster[0]
        x.append(first_point[0])
        y.append(first_point[1])
        for point in cluster[1:]:
            x.append(data.iloc[point-1,0])
            y.append(data.iloc[point-1,1])
        avg_x = sum(x) / len(x)
        avg_y = sum(y) / len(y)
        coord = [avg_x, avg_y]
        new_centers.append(coord)
        x.clear()
        y.clear()
    return new_centers, result_paths
  
  

def main():
    print('\nComputeDronePath')

    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)
    random_cluster_centers,centers_coords = initialize_k_means_clusters(data,2)
    distance_matrix = create_dist_matrix(data, random_cluster_centers)
    new_centers,result_paths = k_means_assign_and_updates(data, distance_matrix, centers_coords, random_cluster_centers)
    newer_centers,newer_result_paths = k_means_assign_and_updates(data, distance_matrix, new_centers,None)

    condition = False
    while not condition:
        newer_centers,newer_result_paths =  k_means_assign_and_updates(data, distance_matrix, new_centers,None)
        if(newer_centers == new_centers) and (newer_result_paths == result_paths):
            condition = True
            print(f"--------------------------------------------")
            print(f"final center values: {newer_centers} \n")
            print(f"final result path: {newer_result_paths} \n")
            print(f"--------------------------------------------")
            break
        else:
            new_centers = newer_centers
            result_paths = newer_result_paths
            condition = False


if __name__ == '__main__':
  main()