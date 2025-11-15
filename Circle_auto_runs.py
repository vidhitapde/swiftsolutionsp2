import numpy as np
from kmeans import (
    extract_coords,
    find_all_clusters,
    calculating_sse
)

def compute_sse_and_centers(all_clusters, all_centers, data, k):
    idx = k - 1
    clusters_k = all_clusters[idx]  
    centers_k = all_centers[idx]     

    per_drone_sses = []
    total_sse = 0.0

    for cluster in clusters_k:
        errors = calculating_sse(cluster, data)
        sse_cluster = sum(errors)
        per_drone_sses.append(sse_cluster)
        total_sse += sse_cluster

    return total_sse, centers_k, per_drone_sses


def run_sse_for_file(input_file, k):
    print(f"\n==============================")
    print(f"File: {input_file}")
    print(f"Using {k} drone(s)")
    print(f"==============================\n")

    data = extract_coords(input_file)
    all_centers, all_clusters = find_all_clusters(data)

    sse_total, centers_k, per_drone_sses = compute_sse_and_centers(
        all_clusters, all_centers, data, k
    )

    if k == 2:
        pairs = sorted(zip(centers_k, per_drone_sses), key=lambda p: p[0][0])
        centers_k = [p[0] for p in pairs]
        per_drone_sses = [p[1] for p in pairs]

    print(f"Total SSE (k={k}): {sse_total:.4f}")
    print("Centers:")
    for i, c in enumerate(centers_k, start=1):
        print(f"  Drone {i} center: [{c[0]:.4f}, {c[1]:.4f}]")
    print("Per-drone SSE:")
    for i, s in enumerate(per_drone_sses, start=1):
        print(f"  Drone {i} SSE: {s:.4f}")

    return {
        "k": k,
        "sse_total": sse_total,
        "centers": centers_k,          
        "per_drone_sses": per_drone_sses  
    }


if __name__ == "__main__":

    files = [
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance0",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance1",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance2",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance3",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance4",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance5",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance6",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance7",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance8",
        "Circle-points-for-testing/1-circle/128-points/1-radius/instance9",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance0",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance1",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance2",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance3",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance4",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance5",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance6",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance7",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance8",
        "Circle-points-for-testing/2-circle/128-points/1-radius/instance9"
        
    ]

    first10_sses = []
    first10_centers = []

    last10_sses_d1 = []
    last10_sses_d2 = []
    last10_centers_d1 = []
    last10_centers_d2 = []

    for i, f in enumerate(files):
        if i < 10:
            k = 1
        else:
            k = 2

        result = run_sse_for_file(f, k)

        if k == 1:
            first10_sses.append(result["sse_total"])
            first10_centers.append(result["centers"][0])  
        else:
            c1, c2 = result["centers"]
            sse1, sse2 = result["per_drone_sses"]

            last10_centers_d1.append(c1)
            last10_centers_d2.append(c2)

            last10_sses_d1.append(sse1)
            last10_sses_d2.append(sse2)

    first10_sses = np.array(first10_sses, dtype=float)
    first10_centers = np.array(first10_centers, dtype=float)

    last10_sses_d1 = np.array(last10_sses_d1, dtype=float)
    last10_sses_d2 = np.array(last10_sses_d2, dtype=float)
    last10_centers_d1 = np.array(last10_centers_d1, dtype=float)
    last10_centers_d2 = np.array(last10_centers_d2, dtype=float)

    avg_sse_first10 = first10_sses.mean()
    avg_center_first10 = first10_centers.mean(axis=0)

    avg_sse_d1 = last10_sses_d1.mean()
    avg_sse_d2 = last10_sses_d2.mean()
    avg_center_d1 = last10_centers_d1.mean(axis=0)
    avg_center_d2 = last10_centers_d2.mean(axis=0)

    print("\n\n=========== FINAL GROUP RESULTS ===========")

    print("\n--- FIRST 10 FILES (k = 1) ---")
    print(f"Average SSE (1 drone): {avg_sse_first10:.4f}")
    print(f"Average center (1 drone): "
          f"[{avg_center_first10[0]:.4f}, {avg_center_first10[1]:.4f}]")

    print("\n--- LAST 10 FILES (k = 2) ---")
    print(f"Averages for Drone 1 center: "
          f"[{avg_center_d1[0]:.4f}, {avg_center_d1[1]:.4f}]")
    print(f"Averages for Drone 2 center: "
          f"[{avg_center_d2[0]:.4f}, {avg_center_d2[1]:.4f}]")
    print(f"Average SSE for Drone 1: {avg_sse_d1:.4f}")
    print(f"Average SSE for Drone 2: {avg_sse_d2:.4f}")

    print("\n===========================================\n")


