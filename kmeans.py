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
    if type(point1) == type(point2):    # both points are from dataframe
        return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)
    else:
        return math.sqrt((point1[0] - point2.iloc[0])**2 + (point1[1] - point2.iloc[1])**2)     # point2 is from dataframe, point1 is a coordinate in form [x,y]
    
# find 4 random points to act as initial cluster centers
def find_cluster_centers(data, k):
    cluster_centers = set()
    i = 0
    while len(cluster_centers) < k:
        index = random.randint(0, len(data))
        cluster_centers.add(index)

    return cluster_centers

# 2d list where each row is one center cluster and the column is the distance of all points to that center
# i.e., dist_matrix[1] is a list containing the distances of all points to point 1
# cluster_centers is list of indexes
# center_test is list of coordinate points
def create_dist_table(data, cluster_centers=None, center_test=None):
    center_coords = []
    if(center_test == None):
        for i in cluster_centers:
            center_coords.append([data.iloc[i-1][0], data.iloc[i-1][1]])
    elif(center_test != None):
        center_coords = center_test

    dist_matrix = np.zeros((len(center_coords), len(data)))
    for i in range(len(center_coords)):
        for j in range(len(data)):
            dist_matrix[i,j] = euclidean_distance(center_coords[i], data.iloc[j])

    return dist_matrix

# def hash_function(val):
#     sum_of_chars = 0
#     for char in val:
#         sum_of_chars += ord(char)

#     return sum_of_chars

#using the random cluster centroids, and finding the points that are closest to the centroid
# center_coords is list of coordinates in form [x,y]
def kmeans_labelling(dist, center_coords):
    distance = float('inf')
    closest_center = 0
    result_hash = [[] for _ in range(len(center_coords))]

    for i in range(len(dist[0])):
        for j in range(len(center_coords)):
            if dist[j][i] < distance:
                distance = dist[j][i]
                closest_center = j
        result_hash[closest_center].append(i)
        distance = float('inf')

    return result_hash

    # n = dist.shape[0]
    # locs = list(range(1,n + 1))
    # # locs_new = [loc for loc in locs if loc not in cluster_centers]
    # result_lists = [[num] for num in cluster_centers] 
    # locations_w_index = {centroid: index for index, centroid in enumerate(cluster_centers)}
    # print(locations_w_index)
    # for locations in locs:
    #     smallest_distance = min(cluster_centers, key=lambda centroid:dist[locations-1][locations_w_index[centroid]])
    #     for centroid in result_lists:
    #         if smallest_distance == centroid[0]:
    #             centroid.append(locations)
    # print(f"Results: {result_lists}")


def kmeans_algorithm(data, k):
    cluster_centers = find_cluster_centers(data, k)
    distance_matrix = create_dist_table(data, cluster_centers)
    center_coords = []
    for i in cluster_centers:
        center_coords.append([data.iloc[i][0], data.iloc[i][1]])

    initial_labels = kmeans_labelling(distance_matrix, center_coords)

    print("Calculating new cluster centroids")
    # need to find the average point for each cluster 
    x = []
    y = []
    coord = []
    avg_centers = []
    
    for cluster in initial_labels:
        for point in cluster:
            x.append(data.iloc[point-1,0])
            y.append(data.iloc[point-1,1])
        avg_x = sum(x) / len(x)
        avg_y = sum(y) / len(y)
        coord = [avg_x,avg_y]
        avg_centers.append(coord)
        
        x.clear()
        y.clear()
    
    dist_matrix_with_avg = create_dist_table(data, None, avg_centers)
    avg_labels = kmeans_labelling(dist_matrix_with_avg, avg_centers)
    test = 0

    while sorted(avg_labels) != sorted(initial_labels):
        test += 1
        print(test)
        initial_labels = avg_labels
        coord = []
        avg_centers = []
    
        for cluster in initial_labels:
            for point in cluster:
                x.append(data.iloc[point-1,0])
                y.append(data.iloc[point-1,1])
            avg_x = sum(x) / len(x)
            avg_y = sum(y) / len(y)
            coord = [avg_x,avg_y]
            avg_centers.append(coord)

        center_coords = avg_centers.copy()
        x.clear()
        y.clear()
    
        dist_matrix_with_avg = create_dist_table(data, None, avg_centers)
        avg_labels = kmeans_labelling(dist_matrix_with_avg, avg_centers)

    return avg_labels, center_coords
        

def main():
    print('\nComputeDronePath')

    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)

    cluster_centers = find_cluster_centers(data, 3)
    distance_matrix = create_dist_table(data, cluster_centers)

    kmeans_labelling(distance_matrix, cluster_centers)

    paths, centers = kmeans_algorithm(data, 3)
    print(centers)


if __name__ == '__main__':
  main()