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
    if type(point1) == type(point2):
        return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)
    else:
        return math.sqrt((point1[0] - point2.iloc[0])**2 + (point1[1] - point2.iloc[1])**2)


def create_dist_matrix(data,cluster_centers):
    n = len(data)
    cluster_size = len(cluster_centers)
    dist_matrix = np.zeros((n, n))   #N x N matrix of zeros
    center_coords = []

    for i in cluster_centers:
        center_coords.append([data.iloc[i-1][0],data.iloc[i-1][1]])
    print(center_coords)
    for i in range(n):
        for j in range(cluster_size):
            dist_matrix[i, j] = dist_matrix[j, i] = euclidean_distance(center_coords[j], data.iloc[i])

    return dist_matrix




def knn_clustering(data, k):
    cluster_centers = set()
    i = 0
    while len(cluster_centers) < k:
        index = random.randint(0, len(data))
        cluster_centers.add(index)

    return cluster_centers

#using the random cluster centroids, and finding the points that are closest to the centroid

def knn_labelling(dist,cluster_centers,data):
    n = dist.shape[0]
    locs = list(range(1,n + 1))
    # locs_new = [loc for loc in locs if loc not in cluster_centers]
    result_lists = [[num] for num in cluster_centers] 
    locations_w_index = {centroid: index for index, centroid in enumerate(cluster_centers)}
    print(locations_w_index)
    for locations in locs:
        smallest_distance = min(cluster_centers, key=lambda centroid:dist[locations-1][locations_w_index[centroid]])
        for centroid in result_lists:
            if smallest_distance == centroid[0]:
                centroid.append(locations)
    print(f"Results: {result_lists}")

    print("Calculating new cluster centroids")
    # need to find the average point for each cluster 
    x = []
    y = []
    coord = []
    centers = []
    print("-------------------------------------------------------")
    for cluster in result_lists:
        for point in cluster:
            x.append(data.iloc[point-1,0])
            y.append(data.iloc[point-1,1])
        print(f"Length of X: {len(x)}")
        print(f"Length of Y: {len(y)}")
        avg_x = sum(x) / len(x)
        avg_y = sum(y) / len(y)
        coord = [avg_x,avg_y]
        centers.append(coord)
        print(f"New Center Values: {centers}")
        x.clear()
        y.clear()
    
    return centers

        

def main():
    print('\nComputeDronePath')

    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)

    cluster_centers = knn_clustering(data, 4)
    distance_matrix = create_dist_matrix(data, cluster_centers)
    centers = knn_labelling(distance_matrix, cluster_centers,data)
    



if __name__ == '__main__':
  main()