#this file is to test #drones/output and #drones/SSE over 10 consective runs and record the average result.

import time
import numpy as np

from kmeans import (    
    extract_coords,
    find_all_clusters,
    calculate_paths,
    create_dist_matrix,
    nearest_neighbor,
    calculating_sse      
)

def compute_sse_totals(all_clusters, data):
    sse_totals = []

    for clusters_for_k in all_clusters:
        if len(clusters_for_k) == 1:
            errors = calculating_sse(clusters_for_k[0], data)
            sse_totals.append(sum(errors))
        else:
            total_sse = 0.0
            for cluster in clusters_for_k:
                errors = calculating_sse(cluster, data)
                total_sse += sum(errors)
            sse_totals.append(total_sse)

    return sse_totals


def run_once(input_file, trial_minutes=5, quick=False):
    data = extract_coords(input_file)

    all_centers, all_clusters = find_all_clusters(data)

    sse_totals = compute_sse_totals(all_clusters, data)

    if quick:
        total_costs = []

        all_dist_matrices = []
        for clusters in all_clusters:
            k_matrices = []
            if len(clusters) > 1:
                for sgl_cluster in clusters:
                    distance_matrix = create_dist_matrix(data, sgl_cluster)
                    k_matrices.append(distance_matrix)
                all_dist_matrices.append(k_matrices)
            else:
                distance_matrix = create_dist_matrix(data, clusters[0])
                all_dist_matrices.append([distance_matrix])

        for matrix in all_dist_matrices:
            if len(matrix) > 4:  
                cost, route = nearest_neighbor(matrix[0])
                total_costs.append(cost)
            else:
                k_costs = []
                for cluster_matrix in matrix:
                    cost, route = nearest_neighbor(cluster_matrix)
                    k_costs.append(cost)
                total_costs.append(sum(k_costs))

        return total_costs, sse_totals

    else:
        end_time = time.time() + trial_minutes * 60
        all_costs, all_routes, total_cost = calculate_paths(data, all_clusters, end_time)
        return total_cost, sse_totals



def run_many(input_file, n_runs=10, trial_minutes=5, quick=False):
    results_costs = []
    results_sse = []

    mode = "QUICK" if quick else "FULL"
    print(f"\n==============================")
    print(f"Running {n_runs} trials for file: {input_file}")
    print(f"Mode: {mode} | Time per run (FULL mode): {trial_minutes} minute(s)")
    print(f"==============================\n")

    for i in range(n_runs):
        print(f"Trial {i+1}/{n_runs} for {input_file}...")
        totals, sse_totals = run_once(input_file, trial_minutes=trial_minutes, quick=quick)
        print(f"  -> Total route lengths this run  (1–4 drones): {totals}")
        print(f"  -> Total SSE values this run     (1–4 drones): {sse_totals}")
        results_costs.append(totals)
        results_sse.append(sse_totals)

    results_costs = np.array(results_costs, dtype=float)
    results_sse = np.array(results_sse, dtype=float)

    avg_costs = results_costs.mean(axis=0)
    avg_sse = results_sse.mean(axis=0)

    print(f"\n===== Averages for {input_file} over {n_runs} runs =====")
    print("Average total route lengths:")
    for drones, avg in enumerate(avg_costs, start=1):
        print(f"  {drones} drone(s): {avg:.2f} meters")

    print("\nAverage total SSE values:")
    for drones, avg in enumerate(avg_sse, start=1):
        print(f"  {drones} drone(s): {avg:.2f}")

    return (avg_costs, results_costs), (avg_sse, results_sse)



if __name__ == "__main__":
    files = [
        "P2_test_cases/Almond9832.txt",
        "P2_test_cases/pecan1212.txt",
        "P2_test_cases/Walnut2621.txt"
    ]

    QUICK_TEST = False   

    if QUICK_TEST:
        for f in files[:1]:
            run_many(f, n_runs=2, trial_minutes=0.05, quick=True)
    else:
        for f in files:
            run_many(f, n_runs=10, trial_minutes=5, quick=False)