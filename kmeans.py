import pandas as pd
import numpy as np
import math as math
import random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime, timedelta

# read in coordinates from file to create dataset
def extract_coords(filename):
    data_names = ['x', 'y']     # label columns
    data = pd.read_csv(filename, sep=r'\s+', names=data_names)

    if not all (data.dtypes == 'float64'):
        raise Exception("Locations are not in float64 format.")
    
    if len(data) > 4096:
        raise Exception("Error: Locations exceeds 4096.")

    return data

# calculate euclidean distance to be used in creating distance matrix
def euclidean_distance(point1, point2):
    if type(point1) == type(point2):    # both points are from dataframe
        return math.sqrt((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)
    else:
        return math.sqrt((point1[0] - point2.iloc[0])**2 + (point1[1] - point2.iloc[1])**2)     # point2 is from dataframe, point1 is a coordinate in form [x,y]
    
#calculating sum of squared errors between each point and its respective center value
def sum_of_squared_errors(point1,point2):
    if type(point1) == type(point2):    # both points are from dataframe
        return ((point1.iloc[0] - point2.iloc[0])**2 + (point1.iloc[1] - point2.iloc[1])**2)
    else:
        return ((point1[0] - point2.iloc[0])**2 + (point1[1] - point2.iloc[1])**2)     # point2 is from dataframe, point1 is a coordinate in form [x,y]   


def create_dist_matrix(data, path):
    n = len(path)
    dist_matrix = np.zeros((n, n))   #N x N matrix of zeros

    #find dists to center
    for i in range(1, n):
        dist_matrix[0][i] = dist_matrix[i][0] = euclidean_distance(path[0], data.iloc[path[i]-1])

    for i in range(1,n):
        for j in range(i, n):
            dist_matrix[i, j] = dist_matrix[j, i] = euclidean_distance(data.iloc[path[i]-1], data.iloc[path[j]-1])

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


def k_means_assign_and_updates(data, center_coords, cluster_centers = None):
    n = len(data) 
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


def nearest_neighbor(dist, clusters):
    n = dist.shape[0]
    remaining_locations = list(range(2, n + 1)) #starts from location 2, because location 1 is starting spot

    best_cost = float('inf')
    best_route = None
    current_optimal_path = [1] #always start with the first x,y coordinate in txt file
    total_cost = 0.0
    nearby_loc = None

    current_location = 1
    while remaining_locations:
        best_cost = float('inf') #current distance will always be less than this, also need to reset for each loc
        for loc in remaining_locations:
            #euclidian distance for (x1,y1) and (x2,y2) represented by point value in row and col in the dist matrix
            current_distance = dist[current_location-1][loc-1]
            if(current_distance < best_cost) and current_distance != 0.0:
                best_cost = current_distance
                nearby_loc = loc # need to put the path together
        total_cost = total_cost + best_cost #putting together the total distance for the route
        current_optimal_path.append(clusters[nearby_loc-1])
        current_location = nearby_loc
        remaining_locations.remove(nearby_loc)

    remaining_locations.append(1) #appends start location to get to the end of the route from that point
    if(len(remaining_locations) == 1):
        current_distance = dist[0][nearby_loc-1]
        total_cost += current_distance
        current_optimal_path.append(remaining_locations[0])
        remaining_locations.remove(remaining_locations[0])
  
    return total_cost, current_optimal_path


def anytime_nearest_neighbor_timed(dist, trial_time, clusters):
   n = dist.shape[0]
   best_so_far, best_route_so_far = nearest_neighbor(dist, clusters)

   it = 0

   nearby_loc = None
   
   start_time = time.time()
   
   while (time.time() < start_time + trial_time):
        remaining_locations = list(range(2, n + 1)) #starts from location 2, because location 1 is starting spot
        current_optimal_path = [1] #always start with the first x,y coordinate in txt file
        total_cost = 0.0
        current_location = 1
        while remaining_locations:
            best_cost = float('inf') #current distance will always be less than this, also need to reset for each loc
            second_best_cost = float('inf')
            second_nearby_loc = None
            for loc in remaining_locations:
                #euclidian distance for (x1,y1) and (x2,y2) represented by point value in row and col in the dist matrix
                current_distance = dist[current_location-1][loc-1]
                if(current_distance < best_cost) and current_distance != 0.0:
                    second_best_cost = best_cost
                    best_cost = current_distance
                    second_nearby_loc = nearby_loc
                    nearby_loc = loc # need to put the path together
                elif (current_distance < second_best_cost) and current_distance != 0.0:
                    second_best_cost = current_distance
                    second_nearby_loc = loc
            if(random.random() < 0.1  and second_nearby_loc in remaining_locations and second_nearby_loc is not None):
                best_cost = second_best_cost
                nearby_loc = second_nearby_loc
            
            total_cost = total_cost + best_cost #putting together the total distance for the route
            current_optimal_path.append(clusters[nearby_loc-1])
            current_location = nearby_loc
            remaining_locations.remove(nearby_loc)


        remaining_locations.append(1) #appends start location to get to the end of the route from that point
        if(len(remaining_locations) == 1):
            current_distance = dist[remaining_locations[0]-1][nearby_loc-1]
            total_cost += current_distance
            current_optimal_path.append(remaining_locations[0])
            remaining_locations.remove(remaining_locations[0])
        if total_cost < best_so_far:
            best_so_far = total_cost
            best_route_so_far = current_optimal_path[:]
        it += 1

   return best_so_far, best_route_so_far

def find_all_clusters(data):
    all_centers = []
    all_clusters = []

    for i in range(1,5):
        random_cluster_centers, centers_coords = initialize_k_means_clusters(data,i)
        new_centers, result_paths = k_means_assign_and_updates(data, centers_coords, random_cluster_centers)
        newer_centers, newer_result_paths = k_means_assign_and_updates(data, new_centers,None)

        condition = False
        while not condition:
            newer_centers,newer_result_paths =  k_means_assign_and_updates(data, new_centers,None)
            if(newer_centers == new_centers) and (newer_result_paths == result_paths):
                condition = True
                break
            else:
                new_centers = newer_centers
                result_paths = newer_result_paths
                condition = False
        
        all_centers.append(newer_centers)
        all_clusters.append(newer_result_paths)

    return all_centers, all_clusters


def calculate_paths(data, all_clusters, end_time):
    all_dist_matrices = []

    create_dist_matrix(data, all_clusters[0][0])  # test call to see print statements

    for clusters in all_clusters:
        k_matrices = []
        if(len(clusters) > 1):
            for sgl_cluster in clusters:
                distance_matrix = create_dist_matrix(data, sgl_cluster)
                k_matrices.append(distance_matrix)

            all_dist_matrices.append(k_matrices)
        else:
            distance_matrix = create_dist_matrix(data, clusters[0])
            all_dist_matrices.append(distance_matrix)

    all_costs = []
    all_routes = []
    total_cost = []
    

    time_remaining = end_time - time.time()
    time_interval = time_remaining/4
    for i in range(len(all_dist_matrices)):
        if len(all_dist_matrices[i]) > 4:     # single drone case
           # print("spliced_cluster:", spliced_cluster)
            spliced_cluster=all_clusters[i][0]
            cost, route = anytime_nearest_neighbor_timed(all_dist_matrices[i], time_interval, spliced_cluster)
            all_costs.append(cost)
            all_routes.append(route)
            total_cost.append(cost)
        else:
            k_costs = []
            k_routes = []
            for j in range(len(all_dist_matrices[i])):
                spliced_cluster=all_clusters[i][j]
                cost, route = anytime_nearest_neighbor_timed(all_dist_matrices[i][j], time_interval/len(all_dist_matrices[i]), spliced_cluster)
                k_costs.append(cost)
                k_routes.append(route)

            all_costs.append(k_costs)
            all_routes.append(k_routes)
            total_cost.append(sum(k_costs))

    return all_costs, all_routes, total_cost


def calculating_sse(paths,data):
    squared_errors = []
    for locations in paths[1:]:
        squared_errors.append(sum_of_squared_errors(paths[0],data.iloc[locations-1]))
    return squared_errors

def plot_graph(drones, best_route,data,file_name, all_centers):
    colors = ['blue', 'orange', 'green', 'purple', 'pink', 'brown', ]
    if drones == 1:
        x = []
        y = []
        for loc in best_route:
            x.append(data.iloc[loc-1, 0])
            y.append(data.iloc[loc-1, 1])
        plt.plot(x, y, color='blue')
        plt.scatter(x[0], y[0], marker = 'o', facecolors = 'purple', s=100)

    else:

        for i in range(drones):
           route= best_route[i]
           center = all_centers[i]
           
           x = [center[0]]
           y = [center[1]]
           #SKIP THe last point
           for loc in route[1:len(route)-1]:
               x.append(data.iloc[loc-1, 0])
               y.append(data.iloc[loc-1, 1])
           plt.plot(x, y, color=colors[i])
           plt.scatter(x[0], y[0], marker = 'o', facecolors = 'red', s=100)

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.title(f'Best Route using {drones} Drone(s)', fontsize=15)
    plt.axis('equal')
    plt.savefig(f"{file_name}_OVERALL_SOLUTION_.jpeg")
    plt.close()


def main():
    print('\nComputeDronePath')

    input_file = input('Enter the name of file: ')
    file_name_w_ext = os.path.basename(input_file)
    file_name,ext = os.path.splitext(file_name_w_ext)
    
    data = extract_coords(input_file)
    curr_time = datetime.now()
    end_time = curr_time + timedelta(minutes=5)
    str_end_time = end_time.strftime("%I:%M %p")
    print(f"There are {len(data)} nodes: Solutions will be available by {str_end_time}")

    end_time = time.time() + 20    # curr time + 5 minutes

    all_centers, all_clusters = find_all_clusters(data)
    # print("all_clusters:", all_clusters)
    all_costs, all_routes, total_cost = calculate_paths(data, all_clusters, end_time)
    print("FINAL all_routes:", all_routes)

    # convert center points to int - round to nearest int then truncate to int
    all_centers_int = []
    for centers in all_centers:
        all_centers_int.append(np.round(centers).astype(int))

    # # convert center points to floats rounded to four decimal places - to be used for circle tests
    # all_centers_round_4 = []
    # for center in all_centers:
    #     rounded_points = []
    #     for point in center:
    #         x = round(point[0], 4)
    #         y = round(point[1], 4)
    #         rounded_points.append([x.item(),y.item()])  # item() converts np float to python native float
    #     all_centers_round_4.append(rounded_points)
    
    index = ['  i.', ' ii.', 'iii.', ' iv.']
    for i in range(len(total_cost)):
        sse_sum_values = 0
        print(f'{i+1}) If you use {i+1} drone(s), the total route will be {total_cost[i]:.1f} meters')
        if len(all_routes[i]) > 4:
            print(f'    {index[0]}    Landing Pad 1 should be at {all_centers_int[0][0]}, serving {len(all_routes[0]) - 2} locations, route is {all_costs[0]:.1f} meters')
            squared_errors = calculating_sse(all_clusters[0][0],data)
            total_sum = sum(squared_errors)
            # print(f"Total Sum of Squared Errors for 1 Drone: {total_sum:.1f}") #uncomment when testing for sse
        else:
            for j in range(len(all_routes[i])):
                print(f'    {index[j]}    Landing Pad {j+1} should be at {all_centers_int[i][j]}, serving {len(all_routes[i][j]) - 2} locations, route is {all_costs[i][j]:.1f} meters')
                multiple_squared_errors = calculating_sse(all_clusters[i][j],data)
                total_sum_values = sum(multiple_squared_errors)
                sse_sum_values += total_sum_values 
            # print(f"Total Sum of Squared Errors for {j+1} Drones: {sse_sum_values:.1f}") #uncomment when testing for sse
    print("Please select your choice 1 to 4: ", end = "")
    choice = int(input())
    plot_route = all_routes[choice-1]
    distance = int(total_cost[choice-1])
    # print("all routes:", all_routes)
    # print_route = best_route[0][:-1]
    # print("Best route:", print_route)
    #remove first and last elements from each drone path for output files
    best_route= []
    if choice == 1:
        trimmed_route = plot_route[1:-1]
        best_route.append(trimmed_route)
    else:
        for i in range(choice):
            trimmed_route = plot_route[i][1:-1]
            best_route.append(trimmed_route)
    inner = ""
    for i in range(choice):
        if(i== 1):
            inner+=  f"{file_name}_{i+1}_SOLUTION_{(all_costs[choice-1][i]):.0f}.txt"
        else:
            inner+=  f"{file_name}_{choice}_SOLUTION_{(all_costs[choice-1][i]):.0f}.txt, "
    print(f"Writing {inner} to disk \n")
    for i in range(choice):
        with open(f"{file_name}_{i+1}_SOLUTION_{(all_costs[choice-1][i]):.0f}.txt","w") as f:
            for i,point in enumerate(best_route[i]):
                f.write(f"{point}"+"\n")
                
    plot_graph(choice, plot_route, data, file_name_w_ext, all_centers_int[choice - 1])

if __name__ == '__main__':
  main()