import os
from clustering_1 import *
import pandas as pd
import numpy as np
import random
import time
import gc


def bfr_cluster(dim, k, n, block_size, in_path, out_path, mahalanobis_threshold=3.0):
    """Performs BFR clustering on large datasets."""
    print(f"Starting BFR clustering algorithm")
    if k is not None:
        print(f"Using specified k={k} clusters")
    else:
        print(f"Will determine optimal k automatically from first data chunk")

    print(f"Processing data in chunks of {block_size} points - wait...")

    start_time = time.time()

    DS = {}
    RS = []
    CS = []

    total_lines = 0
    with open(in_path, 'r') as f:
        for _ in f:
            total_lines += 1

    data_chunks = pd.read_csv(in_path, chunksize=block_size, header=None)
    first_chunk = next(data_chunks)

    processed_points = len(first_chunk)
    progress = (processed_points / total_lines) * 100
    initial_points = first_chunk.iloc[:, :dim].values

    if k is None:
        print("Using custom k_means from clustering_1.py to determine optimal k")

        points_list = [tuple(point) for point in initial_points]

        clusters_output = []
        k_means(dim, None, len(points_list), points_list, clusters_output)

        k = len(clusters_output)
        centers = [calculate_centroid(cluster) for cluster in clusters_output]

        assignments = np.zeros(len(initial_points), dtype=int)

        for i, point in enumerate(initial_points):
            min_dist = float('inf')
            nearest_cluster = 0

            for j, center in enumerate(centers):
                center_array = np.array(center)
                dist = np.linalg.norm(point - center_array)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = j

            assignments[i] = nearest_cluster
    else:
        centers = [initial_points[0]]
        for _ in range(1, k):
            distances = np.array([min([np.linalg.norm(p - c) ** 2 for c in centers]) for p in initial_points])
            probs = distances / distances.sum()
            cumprobs = np.cumsum(probs)
            r = random.random()
            ind = np.searchsorted(cumprobs, r)
            centers.append(initial_points[ind])

        for iter_idx in range(5):
            assignments = np.argmin([np.linalg.norm(initial_points - c, axis=1) for c in centers], axis=0)

            for j in range(k):
                if np.sum(assignments == j) > 0:
                    centers[j] = np.mean(initial_points[assignments == j], axis=0)

    for j in range(k):
        cluster_points = initial_points[assignments == j]
        if len(cluster_points) > 0:
            DS[j] = {
                'n': len(cluster_points),
                'sum': np.sum(cluster_points, axis=0),
                'sum_sq': np.sum(cluster_points ** 2, axis=0),
                'centroid': centers[j] if isinstance(centers[j], np.ndarray) else np.array(centers[j]),
                'variance': np.var(cluster_points, axis=0) if len(cluster_points) > 1 else np.ones(dim)
            }

    chunk_idx = 1
    for chunk in data_chunks:
        chunk_start_time = time.time()
        chunk_points = chunk.iloc[:, :dim].values

        processed_points += len(chunk_points)
        progress = (processed_points / total_lines) * 100

        import sys

        def update_overall_progress(progress, processed_points, total_lines):
            sys.stdout.write(f"\rBFR Overall progress: {progress:.2f}% ({processed_points}/{total_lines} points)    ")
            sys.stdout.flush()

        update_overall_progress(progress, processed_points, total_lines)

        ds_assigned = 0
        rs_assigned = 0

        for point_idx, point in enumerate(chunk_points):
            assigned = False

            min_mahalanobis = float('inf')
            best_cluster = -1

            for cluster_id, stats in DS.items():
                diff = point - stats['centroid']
                if np.any(stats['variance'] <= 0):
                    mahalanobis_dist = np.linalg.norm(diff)
                else:
                    mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / stats['variance']))

                if mahalanobis_dist < min_mahalanobis:
                    min_mahalanobis = mahalanobis_dist
                    best_cluster = cluster_id

            if min_mahalanobis < mahalanobis_threshold:
                DS[best_cluster]['n'] += 1
                DS[best_cluster]['sum'] += point
                DS[best_cluster]['sum_sq'] += point ** 2
                DS[best_cluster]['centroid'] = DS[best_cluster]['sum'] / DS[best_cluster]['n']
                DS[best_cluster]['variance'] = (DS[best_cluster]['sum_sq'] / DS[best_cluster]['n'] -
                                                DS[best_cluster]['centroid'] ** 2)
                DS[best_cluster]['variance'] = np.maximum(DS[best_cluster]['variance'], 1e-6)
                assigned = True
                ds_assigned += 1

            if not assigned:
                RS.append(point)
                rs_assigned += 1

        if len(RS) >= 5 * k:
            rs_array = np.array(RS)

            rs_centers = [rs_array[0]]
            for _ in range(1, k):
                rs_distances = np.array([min([np.linalg.norm(p - c) ** 2 for c in rs_centers]) for p in rs_array])
                rs_probs = rs_distances / (rs_distances.sum() + 1e-10)
                rs_cumprobs = np.cumsum(rs_probs)
                r = random.random()
                rs_ind = np.searchsorted(rs_cumprobs, r)
                if rs_ind >= len(rs_array):
                    rs_ind = len(rs_array) - 1
                rs_centers.append(rs_array[rs_ind])

            for iter_idx in range(5):
                rs_assignments = np.argmin([np.linalg.norm(rs_array - c, axis=1) for c in rs_centers], axis=0)

                for j in range(k):
                    if np.sum(rs_assignments == j) > 0:
                        rs_centers[j] = np.mean(rs_array[rs_assignments == j], axis=0)

            new_RS = []
            cs_created = 0

            for j in range(k):
                cluster_points = rs_array[rs_assignments == j]
                if len(cluster_points) >= 5:
                    cs_stats = {
                        'n': len(cluster_points),
                        'sum': np.sum(cluster_points, axis=0),
                        'sum_sq': np.sum(cluster_points ** 2, axis=0),
                        'centroid': rs_centers[j],
                        'variance': np.var(cluster_points, axis=0) if len(cluster_points) > 1 else np.ones(dim)
                    }
                    CS.append(cs_stats)
                    cs_created += 1
                else:
                    new_RS.extend(cluster_points)
            RS = new_RS

            original_cs_count = len(CS)
            i = 0
            merges = 0

            while i < len(CS):
                merged = False
                j = i + 1
                while j < len(CS):
                    cs_dist = np.linalg.norm(CS[i]['centroid'] - CS[j]['centroid'])

                    if cs_dist < mahalanobis_threshold:
                        CS[i]['n'] += CS[j]['n']
                        CS[i]['sum'] += CS[j]['sum']
                        CS[i]['sum_sq'] += CS[j]['sum_sq']
                        CS[i]['centroid'] = CS[i]['sum'] / CS[i]['n']
                        CS[i]['variance'] = (CS[i]['sum_sq'] / CS[i]['n'] - CS[i]['centroid'] ** 2)
                        CS[i]['variance'] = np.maximum(CS[i]['variance'], 1e-6)

                        del CS[j]
                        merged = True
                        merges += 1
                    else:
                        j += 1

                if not merged:
                    i += 1
            i = 0
            cs_to_ds_merges = 0

            while i < len(CS):
                min_cs_mahalanobis = float('inf')
                best_cs_cluster = -1

                for cluster_id, stats in DS.items():
                    diff = CS[i]['centroid'] - stats['centroid']
                    if np.any(stats['variance'] <= 0):
                        cs_mahalanobis_dist = np.linalg.norm(diff)
                    else:
                        cs_mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / stats['variance']))

                    if cs_mahalanobis_dist < min_cs_mahalanobis:
                        min_cs_mahalanobis = cs_mahalanobis_dist
                        best_cs_cluster = cluster_id

                if min_cs_mahalanobis < mahalanobis_threshold:
                    DS[best_cs_cluster]['n'] += CS[i]['n']
                    DS[best_cs_cluster]['sum'] += CS[i]['sum']
                    DS[best_cs_cluster]['sum_sq'] += CS[i]['sum_sq']
                    DS[best_cs_cluster]['centroid'] = DS[best_cs_cluster]['sum'] / DS[best_cs_cluster]['n']
                    DS[best_cs_cluster]['variance'] = (DS[best_cs_cluster]['sum_sq'] / DS[best_cs_cluster]['n'] -
                                                       DS[best_cs_cluster]['centroid'] ** 2)
                    DS[best_cs_cluster]['variance'] = np.maximum(DS[best_cs_cluster]['variance'], 1e-6)

                    del CS[i]
                    cs_to_ds_merges += 1
                else:
                    i += 1
        chunk_idx += 1

    print("\nFinal processing stage: Assigning all remaining points and clusters...")

    final_assignments = {}

    for i, point in enumerate(RS):
        min_mahalanobis = float('inf')
        best_cluster = -1

        for cluster_id, stats in DS.items():
            diff = point - stats['centroid']
            if np.any(stats['variance'] <= 0):
                mahalanobis_dist = np.linalg.norm(diff)
            else:
                mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / stats['variance']))

            if mahalanobis_dist < min_mahalanobis:
                min_mahalanobis = mahalanobis_dist
                best_cluster = cluster_id

        final_assignments[i] = best_cluster

    next_cluster_id = max(DS.keys()) + 1 if DS else 0
    cs_merged = 0
    cs_new = 0

    for cs_stats in CS:
        min_cs_mahalanobis = float('inf')
        best_cs_cluster = -1

        for cluster_id, stats in DS.items():
            diff = cs_stats['centroid'] - stats['centroid']
            if np.any(stats['variance'] <= 0):
                cs_mahalanobis_dist = np.linalg.norm(diff)
            else:
                cs_mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / stats['variance']))

            if cs_mahalanobis_dist < min_cs_mahalanobis:
                min_cs_mahalanobis = cs_mahalanobis_dist
                best_cs_cluster = cluster_id

        if min_cs_mahalanobis < mahalanobis_threshold:
            DS[best_cs_cluster]['n'] += cs_stats['n']
            DS[best_cs_cluster]['sum'] += cs_stats['sum']
            DS[best_cs_cluster]['sum_sq'] += cs_stats['sum_sq']
            DS[best_cs_cluster]['centroid'] = DS[best_cs_cluster]['sum'] / DS[best_cs_cluster]['n']
            DS[best_cs_cluster]['variance'] = (DS[best_cs_cluster]['sum_sq'] / DS[best_cs_cluster]['n'] -
                                               DS[best_cs_cluster]['centroid'] ** 2)
            DS[best_cs_cluster]['variance'] = np.maximum(DS[best_cs_cluster]['variance'], 1e-6)
            cs_merged += 1
        else:
            DS[next_cluster_id] = cs_stats
            next_cluster_id += 1
            cs_new += 1

    print(f"Writing final cluster assignments to {out_path}...")
    data_chunks = pd.read_csv(in_path, chunksize=block_size, header=None)

    with open(out_path, "w", newline="") as f:
        point_idx = 0
        write_chunk_idx = 0

        for chunk in data_chunks:
            chunk_points = chunk.iloc[:, :dim].values
            chunk_size = len(chunk_points)
            write_chunk_idx += 1

            for point in chunk_points:
                min_dist = float('inf')
                best_ds_cluster = 0

                for cluster_id, stats in DS.items():
                    dist = np.linalg.norm(point - stats['centroid'])
                    if dist < min_dist:
                        min_dist = dist
                        best_ds_cluster = cluster_id

                f.write(",".join(map(str, point)) + f",{best_ds_cluster}\n")
                point_idx += 1

                if point_idx % 10000 == 0:
                    write_progress = (point_idx / total_lines) * 100
                    import sys

                    def update_writing_progress(write_progress, point_idx, total_lines):
                        sys.stdout.write(
                            f"\r  Writing progress: {write_progress:.2f}% ({point_idx}/{total_lines} points)    ")
                        sys.stdout.flush()

                    update_writing_progress(write_progress, point_idx, total_lines)

    total_time = time.time() - start_time
    print(f"BFR clustering completed in {total_time:.2f} seconds!")
    print(f"Results written to {out_path}")

    return {cluster_id: stats['centroid'] for cluster_id, stats in DS.items()}
def cure_cluster(dim, k, n, block_size, in_path, out_path, return_tuples=False):
    """Perform CURE clustering using hierarchical clustering as the base algorithm."""
    print("=======================================")
    print("CURE Clustering Algorithm Starting (using custom h_clustering)")
    print(f"Parameters: dim={dim}, k={k}, n={n}, block_size={block_size}")
    print("=======================================")

    start_time = time.time()

    total_lines = 0
    with open(in_path, 'r') as f:
        for _ in f:
            total_lines += 1
    print(f"Input file contains {total_lines} lines")

    total_loaded = 0
    results = []

    reader = pd.read_csv(in_path, chunksize=block_size, usecols=list(range(dim)), dtype=np.float32)

    representatives_by_cluster = {}

    for chunk_idx, chunk in enumerate(reader):
        chunk_start_time = time.time()
        points = chunk.to_numpy()

        if total_loaded + len(points) > n:
            points = points[:n - total_loaded]
            total_loaded = n
        else:
            total_loaded += len(points)

        chunk_progress = (total_loaded / n) * 100
        import sys

        def update_progress(chunk_progress, total_loaded, n):
            sys.stdout.write(f"\rProgress: {chunk_progress:.2f}% ({total_loaded}/{n} points)    ")
            sys.stdout.flush()

        update_progress(chunk_progress, total_loaded, n)

        if chunk_idx == 0:
            sample_size = min(len(points), max(2000, block_size // 20))
            sampled_indices = np.random.choice(len(points), sample_size, replace=False)
            sampled_points = [tuple(points[i]) for i in sampled_indices]

            if k is None:
                print("No k specified, will use auto-detection in h_clustering...")

            print("Training initial clustering model using custom h_clustering...")
            initial_clusters = []
            h_clustering(dim, k, -1, sampled_points, initial_clusters)

            if k is None:
                k = len(initial_clusters)
                print(f"h_clustering auto-detected k={k} clusters")

            sample_labels = np.zeros(len(sampled_points), dtype=int)
            for cluster_idx, cluster in enumerate(initial_clusters):
                for point in cluster:
                    try:
                        point_idx = sampled_points.index(point)
                        sample_labels[point_idx] = cluster_idx
                    except ValueError:
                        continue

            cluster_sizes = [np.sum(sample_labels == i) for i in range(k)]

            num_representatives = min(10, sample_size // k)
            centroids = []

            for i in range(k):
                cluster_points = np.array(
                    [sampled_points[j] for j in range(len(sampled_points)) if sample_labels[j] == i])
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    centroids.append(centroid)

                    cluster_representatives = []
                    if len(cluster_points) <= num_representatives:
                        cluster_representatives = cluster_points.copy()
                    else:
                        distances_from_centroid = np.linalg.norm(cluster_points - centroid, axis=1)
                        farthest_idx = np.argmax(distances_from_centroid)
                        cluster_representatives.append(cluster_points[farthest_idx])

                        mask = np.ones(len(cluster_points), dtype=bool)
                        mask[farthest_idx] = False

                        while len(cluster_representatives) < num_representatives and np.any(mask):
                            current_reps = np.array(cluster_representatives)
                            min_distances = np.full(len(cluster_points), float('inf'))

                            for idx in range(len(cluster_points)):
                                if mask[idx]:
                                    point = cluster_points[idx]
                                    dists = np.linalg.norm(current_reps - point, axis=1)
                                    min_distances[idx] = np.min(dists)

                            min_distances[~mask] = -np.inf

                            next_rep_idx = np.argmax(min_distances)
                            cluster_representatives.append(cluster_points[next_rep_idx])
                            mask[next_rep_idx] = False

                    shrink_factor = 0.2
                    shrunk_representatives = []
                    for rep in cluster_representatives:
                        vector_to_centroid = centroid - rep
                        shrunk_rep = rep + shrink_factor * vector_to_centroid
                        shrunk_representatives.append(shrunk_rep)

                    representatives_by_cluster[i] = np.array(shrunk_representatives)
                else:
                    centroid = np.zeros(dim)
                    centroids.append(centroid)
                    representatives_by_cluster[i] = np.array([centroid])
        labels = np.zeros(len(points), dtype=int)
        for i, point in enumerate(points):
            min_dist = float('inf')
            nearest_cluster = 0

            for cluster_idx, reps in representatives_by_cluster.items():
                dists = np.linalg.norm(reps - point, axis=1)
                closest_rep_dist = np.min(dists)

                if closest_rep_dist < min_dist:
                    min_dist = closest_rep_dist
                    nearest_cluster = cluster_idx

            labels[i] = nearest_cluster

        cluster_counts = np.bincount(labels, minlength=k)

        with open(out_path, "a", newline="") as f:
            for i, point in enumerate(points):
                f.write(",".join(map(str, point)) + f",{labels[i]}\n")

        if return_tuples:
            results.extend([tuple(list(points[i]) + [labels[i]]) for i in range(len(points))])

        if total_loaded >= n:
            print(f"Reached target of {n} points, stopping...")
            break

        del points
        gc.collect()

    total_time = time.time() - start_time
    print("=======================================")
    print(f"CURE clustering completed in {total_time:.2f} seconds!")
    print(f"Total points processed: {total_loaded+1}")
    print(f"Results saved to: {out_path}")
    print("=======================================")

    return results if return_tuples else None
def generate_bfr_dataset(output_file, dim, num_clusters, points_per_cluster, std_dev=1.5, chunk_size=1000):
    """Generate a dataset optimized for BFR algorithm with spherical clusters."""
    total_points = num_clusters * points_per_cluster

    print(f"Generating BFR dataset with:")
    print(f"- Dimensions: {dim}")
    print(f"- Clusters: {num_clusters}")
    print(f"- Points per cluster: {points_per_cluster}")
    print(f"- Total points: {total_points}")
    print(f"- Features: Perfectly spherical clusters with clear separation")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    centers = []
    min_distance = 50.0

    for _ in range(num_clusters):
        while True:
            center = np.random.uniform(-100, 100, dim)

            if not centers or all(np.linalg.norm(center - existing) > min_distance for existing in centers):
                centers.append(center)
                break

    print(f"Generated {len(centers)} well-separated cluster centers")

    with open(output_file, 'w') as f:
        chunks_count = (total_points + chunk_size - 1) // chunk_size

        print(f"BFR - Generating {chunks_count} chunks with amount of points - wait... ")

        for chunk_idx in range(chunks_count):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)
            current_chunk_size = end_idx - start_idx

            points_per_cluster_in_chunk = current_chunk_size // num_clusters
            remainder = current_chunk_size % num_clusters

            for cluster_idx in range(num_clusters):
                if cluster_idx < remainder:
                    cluster_points = points_per_cluster_in_chunk + 1
                else:
                    cluster_points = points_per_cluster_in_chunk

                if cluster_points <= 0:
                    continue

                center = centers[cluster_idx]

                current_std = std_dev * (0.5 + random.random())

                for i in range(cluster_points):
                    point = np.random.normal(loc=center, scale=current_std)

                    f.write(','.join(map(str, point)) + f',{cluster_idx}\n')

    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 ** 2)

    print(f"BFR dataset generated successfully.")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Saved to: {output_file}")

def generate_cure_dataset(output_file, dim, num_clusters, points_per_cluster, chunk_size = 1000):
    """Generate a dataset optimized for CURE algorithm with non-spherical clusters."""
    total_points = num_clusters * points_per_cluster

    print(f"Generating CURE dataset with:")
    print(f"- Dimensions: {dim}")
    print(f"- Clusters: {num_clusters}")
    print(f"- Points per cluster: {points_per_cluster}")
    print(f"- Total points: {total_points}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        chunks_count = (total_points + chunk_size - 1) // chunk_size

        centers = []
        for _ in range(num_clusters):
            center = np.random.uniform(-80, 80, dim)
            centers.append(center)

        print(f"CURE - Generating {chunks_count} chunks with amount of points - wait...")
        for chunk_idx in range(chunks_count):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, total_points)
            current_chunk_size = end_idx - start_idx
            points_per_cluster_in_chunk = current_chunk_size // num_clusters
            remainder = current_chunk_size % num_clusters

            for cluster_idx in range(num_clusters):
                if cluster_idx < remainder:
                    cluster_points = points_per_cluster_in_chunk + 1
                else:
                    cluster_points = points_per_cluster_in_chunk

                if cluster_points <= 0:
                    continue

                center = centers[cluster_idx]

                if cluster_idx % 6 == 0:
                    t = np.linspace(0, 10 * np.pi, cluster_points)
                    radii = np.linspace(0.5, 15, cluster_points)
                    x = radii * np.cos(t)
                    y = radii * np.sin(t)

                elif cluster_idx % 6 == 1:
                    t = np.linspace(-3, 3, cluster_points)
                    x = t
                    y = 10 * np.sin(t) * np.cos(t)

                elif cluster_idx % 6 == 2:
                    angles = np.linspace(0, 2 * np.pi, cluster_points)
                    radius = 12
                    noise = np.random.normal(0, 0.3, cluster_points)
                    x = (radius + noise) * np.cos(angles)
                    y = (radius + noise) * np.sin(angles)

                elif cluster_idx % 6 == 3:
                    angles = np.linspace(0, np.pi, cluster_points)
                    outer_radius = 12
                    inner_radius = 10
                    is_outer = np.random.choice([True, False], size=cluster_points, p=[0.5, 0.5])
                    radius = np.where(is_outer, outer_radius, inner_radius)
                    noise = np.random.normal(0, 0.2, cluster_points)
                    x = (radius + noise) * np.cos(angles)
                    y = (radius + noise) * np.sin(angles)

                elif cluster_idx % 6 == 4:
                    side = 20
                    points_per_side = cluster_points // 4

                    side_points = []

                    top_x = np.linspace(-side / 2, side / 2, points_per_side)
                    top_y = np.ones(points_per_side) * side / 2
                    top_noise = np.random.normal(0, 0.2, (points_per_side, 2))
                    side_points.append(np.column_stack((top_x, top_y)) + top_noise)

                    right_x = np.ones(points_per_side) * side / 2
                    right_y = np.linspace(side / 2, -side / 2, points_per_side)
                    right_noise = np.random.normal(0, 0.2, (points_per_side, 2))
                    side_points.append(np.column_stack((right_x, right_y)) + right_noise)

                    bottom_x = np.linspace(side / 2, -side / 2, points_per_side)
                    bottom_y = np.ones(points_per_side) * -side / 2
                    bottom_noise = np.random.normal(0, 0.2, (points_per_side, 2))
                    side_points.append(np.column_stack((bottom_x, bottom_y)) + bottom_noise)

                    left_x = np.ones(points_per_side) * -side / 2
                    left_y = np.linspace(-side / 2, side / 2, points_per_side)
                    left_noise = np.random.normal(0, 0.2, (points_per_side, 2))
                    side_points.append(np.column_stack((left_x, left_y)) + left_noise)

                    square_points = np.vstack(side_points)

                    remaining = cluster_points - square_points.shape[0]
                    if remaining > 0:
                        idx = np.random.choice(square_points.shape[0], remaining)
                        extra_points = square_points[idx] + np.random.normal(0, 0.2, (remaining, 2))
                        square_points = np.vstack((square_points, extra_points))

                    square_points = square_points[:cluster_points]

                    x = square_points[:, 0]
                    y = square_points[:, 1]

                else:
                    half_points = cluster_points // 2

                    x1 = np.linspace(-10, 10, half_points)
                    y1 = np.random.normal(0, 0.3, half_points)

                    x2 = np.random.normal(0, 0.3, cluster_points - half_points)
                    y2 = np.linspace(-10, 10, cluster_points - half_points)

                    x = np.concatenate([x1, x2])
                    y = np.concatenate([y1, y2])

                for i in range(cluster_points):
                    point = np.zeros(dim)

                    point[0] = x[i] + center[0]
                    point[1] = y[i] + center[1]

                    if dim > 2:
                        point[2:] = np.random.normal(center[2:], 0.5)

                    point += np.random.normal(0, 0.1, dim)
                    f.write(','.join(map(str, point)) + f',{cluster_idx}\n')

    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 ** 2)

    print(f"CURE dataset generated successfully.")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Saved to: {output_file}")