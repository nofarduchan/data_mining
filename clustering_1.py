import csv
import math
import time
import random
import numpy as np


def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
def generate_data(dim, k, n, out_path, extras={}):
    """Generate a dataset with given dimensions and clusters using divisive approach."""
    min_val = extras.get("min_val", -100)
    max_val = extras.get("max_val", 100)

    points = [tuple(random.uniform(min_val, max_val) for _ in range(dim)) for _ in range(n)]

    clusters = {0: points}
    next_cluster_id = 1

    while len(clusters) < k:
        largest_cluster_id = max(clusters.keys(), key=lambda i: len(clusters[i]))
        cluster_to_split = clusters[largest_cluster_id]

        if len(cluster_to_split) <= 1:
            break

        max_distance = -1
        centers = None
        sample_size = min(len(cluster_to_split), 100)
        sample_points = random.sample(cluster_to_split, sample_size)

        for i, p1 in enumerate(sample_points):
            for p2 in sample_points[i + 1:]:
                dist = euclidean_distance(p1, p2)
                if dist > max_distance:
                    max_distance = dist
                    centers = (p1, p2)

        del clusters[largest_cluster_id]
        clusters[largest_cluster_id] = []
        clusters[next_cluster_id] = []

        for point in cluster_to_split:
            dist1 = euclidean_distance(point, centers[0])
            dist2 = euclidean_distance(point, centers[1])

            if dist1 <= dist2:
                clusters[largest_cluster_id].append(point)
            else:
                clusters[next_cluster_id].append(point)

        next_cluster_id += 1

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, cluster_points in clusters.items():
            for point in cluster_points:
                writer.writerow(list(point) + [cluster_id])

    print(
        f"Generated {n} points in {dim} dimensions, divided into {len(clusters)} clusters using divisive approach, and saved to {out_path}")
    return clusters
def load_points(in_path, dim, n=-1, points=[]):
    """Load points from a CSV file."""
    points.clear()

    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        count = 0

        for row in reader:
            if n != -1 and count >= n:
                break

            if len(row) >= dim:
                point = tuple(float(row[i]) for i in range(dim))
                points.append(point)
                count += 1

    print(f"Loaded {len(points)} points in {dim} dimensions from {in_path}")
    return points
def save_points(clusts, out_path, out_path_tagged):
    """Save clustered points to CSV files."""
    all_points = []
    total_points = 0
    for i, cluster in enumerate(clusts):
        for point in cluster:
            all_points.append((point, i))
            total_points += 1

    random.shuffle(all_points)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point, _ in all_points:
            writer.writerow(point)

    with open(out_path_tagged, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, cluster in enumerate(clusts):
            for point in cluster:
                writer.writerow(list(point) + [i])

    print(f"Saved {total_points} points to {out_path}")
    print(f"Saved {total_points} points with cluster labels to {out_path_tagged}")
def calculate_cluster_cohesion(cluster, distance_func):
    """Calculate the cohesion of a cluster."""
    if len(cluster) <= 1:
        return 0

    total_distance = 0
    pairs_count = 0

    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            total_distance += distance_func(cluster[i], cluster[j])
            pairs_count += 1

    avg_distance = total_distance / pairs_count if pairs_count > 0 else 0
    return -avg_distance
def calculate_merged_cluster_cohesion(cluster1, cluster2, distance_func):
    """Calculate the cohesion of a merged cluster without actually merging them."""
    merged_cluster = cluster1 + cluster2
    return calculate_cluster_cohesion(merged_cluster, distance_func)
def h_clustering(dim, k, n, points, clusts=[]):
    """Perform bottom-up hierarchical clustering."""
    clusts.clear()

    start_time = time.time()

    if len(points) == 0:
        print("No points provided for clustering")
        return clusts

    distance_func = euclidean_distance

    print(f"\n=== Starting bottom-up hierarchical clustering ===")
    print(f"• Points: {len(points)}")
    print(f"• Dimensions: {dim}")
    print(f"• Target clusters: {'Auto-detect using cohesion' if k is None else k}")

    clusters = [[point] for point in points]
    print(f"• Created {len(clusters)} initial clusters")

    if k is not None and k > 0:
        return optimized_hierarchical_clustering(clusters, k, distance_func, clusts, start_time)

    cohesion_threshold = None
    if k is None:
        max_k = min(8, len(points) // 2)
        print(f"\n=== Auto-determining optimal number of clusters using cohesion threshold ===")
        print(f"• Determining cohesion threshold...")

        sample_size = min(100, len(clusters))
        sample_clusters = random.sample(clusters, sample_size)

        initial_cohesion_values = [calculate_cluster_cohesion(cluster, distance_func) for cluster in sample_clusters]

        test_merged_clusters = []
        for i in range(min(50, len(clusters))):
            idx1 = random.randint(0, len(clusters) - 1)
            remaining = [j for j in range(len(clusters)) if j != idx1]
            if remaining:
                idx2 = random.choice(remaining)
                test_merged = clusters[idx1] + clusters[idx2]
                test_merged_clusters.append(test_merged)

        merged_cohesion_values = [calculate_cluster_cohesion(cluster, distance_func) for cluster in
                                  test_merged_clusters]

        all_cohesion_values = initial_cohesion_values + merged_cohesion_values

        cohesion_threshold = np.percentile(all_cohesion_values, 25) if all_cohesion_values else -10

    last_report_time = time.time()
    report_interval = 3
    initial_clusters = len(clusters)

    merge_count = 0
    print(f"• Performing hierarchical clustering...(15-25 min)\n "
          f"  it takes some time because of time running is large! ")

    while len(clusters) > 1:
        if k is not None and len(clusters) <= k:
            print(f"\n ✓Reached target number of clusters: {k}")
            break

        current_time = time.time()
        if current_time - last_report_time > report_interval:
            elapsed = current_time - start_time
            progress = ((initial_clusters - len(clusters)) / (initial_clusters - 1)) * 100
            import sys

            def update_clustering_progress(progress, clusters):
                sys.stdout.write(f"\r  → Hierarchical progress with k={k}: {progress:.2f}% ")
                sys.stdout.flush()

            update_clustering_progress(progress, clusters)
            last_report_time = current_time

        best_cohesion = float('-inf')
        merge_indices = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                merged_cohesion = calculate_merged_cluster_cohesion(
                    clusters[i], clusters[j], distance_func
                )

                if merged_cohesion > best_cohesion:
                    best_cohesion = merged_cohesion
                    merge_indices = (i, j)

        if k is None and cohesion_threshold is not None and best_cohesion < cohesion_threshold:
            k = max(2, min(len(clusters), 8))
            print(f"\n✅ h_clustering detected optimal k (limited to 2-8) using cohesion_threshold: {k}")
            break

        if merge_indices[0] != -1 and merge_indices[1] != -1:
            i, j = merge_indices
            clusters[i].extend(clusters[j])
            clusters.pop(j)
            merge_count += 1
        else:
            print(f"  ⚠ Warning: No valid clusters to merge. Stopping.")
            break

    if k is None:
        k = max(2, min(len(clusters), 8))
        print(f"\n⚠ Could not determine optimal k using cohesion, using default: k={k}")

    while len(clusters) > k:
        worst_cohesion = float('inf')
        merge_indices = (-1, -1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                merged_cohesion = calculate_merged_cluster_cohesion(
                    clusters[i], clusters[j], distance_func
                )

                if merged_cohesion < worst_cohesion:
                    worst_cohesion = merged_cohesion
                    merge_indices = (i, j)

        if merge_indices[0] != -1 and merge_indices[1] != -1:
            i, j = merge_indices
            clusters[i].extend(clusters[j])
            clusters.pop(j)
        else:
            break

    clusts.extend(clusters)

    total_time = time.time() - start_time
    print(f"\n=== Hierarchical clustering completed ===")
    print(f"• Final clusters: {len(clusts)}")

    return clusts
def optimized_hierarchical_clustering(clusters, k, distance_func, clusts, start_time):
    """Optimized version of hierarchical clustering when k is known."""
    import heapq
    import sys

    if len(clusters) <= k:
        clusts.extend(clusters)
        print(f"\n=== Hierarchical clustering completed ===")
        print(f"• Final clusters: {len(clusts)}")
        return clusts

    centroids = []
    for cluster in clusters:
        if len(cluster) == 1:
            centroids.append(cluster[0])
        else:
            centroid = calculate_centroid(cluster)
            centroids.append(centroid)

    distances = []

    batch_size = 1000

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = distance_func(centroids[i], centroids[j])
            heapq.heappush(distances, (dist, i, j))

    if len(clusters) > batch_size:
        print()

    valid_clusters = list(range(len(clusters)))
    merged = [False] * len(clusters)
    result_clusters = []

    report_interval = 3
    last_report_time = time.time()
    total_merges_needed = len(clusters) - k
    merges_done = 0

    print(f"\n• Performing optimized hierarchical clustering (target: {k} clusters)...")

    while len(valid_clusters) > k and distances:
        current_time = time.time()
        if current_time - last_report_time > report_interval:
            progress = (merges_done / total_merges_needed) * 100
            sys.stdout.write(f"\r  → Hierarchical progress with k={k}: {progress:.2f}% ")
            sys.stdout.flush()
            last_report_time = current_time

        dist, i, j = heapq.heappop(distances)

        if merged[i] or merged[j]:
            continue

        merged[j] = True
        clusters[i].extend(clusters[j])

        centroids[i] = calculate_centroid(clusters[i])

        valid_clusters.remove(j)

        for l in valid_clusters:
            if l != i and not merged[l]:
                dist = distance_func(centroids[i], centroids[l])
                heapq.heappush(distances, (dist, i, l) if i < l else (dist, l, i))

        merges_done += 1

    for idx in valid_clusters:
        result_clusters.append(clusters[idx])
    clusts.extend(result_clusters)

    print(f"\n=== Hierarchical clustering completed ===")
    print(f"• Final clusters: {len(clusts)}")

    return clusts
def calculate_centroid(points):
    """Calculate the centroid of a cluster of points."""
    if not points:
        return None

    dim = len(points[0])
    centroid = tuple(sum(point[i] for point in points) / len(points) for i in range(dim))

    return centroid
def initialize_centroids_dispersed(points, k):
    """Choose initial centroids using a dispersed method."""
    points_array = np.array(points)
    n_points = len(points_array)

    if k <= 0 or k > n_points:
        return []

    selected_indices = set()

    first_idx = np.random.randint(0, n_points)
    selected_indices.add(first_idx)
    centroids = [points_array[first_idx].copy()]

    for _ in range(1, k):
        max_min_distance = -1
        farthest_idx = -1

        for i in range(n_points):
            if i in selected_indices:
                continue

            min_distance = float('inf')
            for centroid in centroids:
                dist = np.sum((points_array[i] - centroid) ** 2) ** 0.5
                min_distance = min(min_distance, dist)

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                farthest_idx = i

        if farthest_idx != -1:
            selected_indices.add(farthest_idx)
            centroids.append(points_array[farthest_idx].copy())
        else:
            remaining_indices = [i for i in range(n_points) if i not in selected_indices]
            if remaining_indices:
                random_idx = np.random.choice(remaining_indices)
                selected_indices.add(random_idx)
                centroids.append(points_array[random_idx].copy())

    return centroids
def k_means(dim, k, n, points, clusts=[]):
    """Perform k-means clustering on the given points."""
    clusts.clear()

    if len(points) == 0:
        print("No points provided for clustering")
        return clusts

    print(f"=== Starting k-means clustering ===")
    print(f"• Points: {len(points)}")
    print(f"• Dimensions: {dim}")
    print(f"• Target clusters: {'Auto-detect' if k is None else k}")

    if k is None:
        max_k = min(8, len(points) // 2)
        sses = []
        silhouette_scores = []

        print("Auto-determining optimal number of clusters using elbow method...")
        for test_k in range(1, max_k + 1):
            centroids = initialize_centroids_dispersed(points,
                                                       test_k) if 'initialize_centroids_dispersed' in globals() else [
                points[i] for i in random.sample(range(len(points)), test_k)]

            old_sse = float('inf')
            sse = 0
            max_iter = 100
            current_clusters = []

            for _ in range(max_iter):
                current_clusters = [[] for _ in range(test_k)]

                for point in points:
                    min_dist = float('inf')
                    nearest_cluster = 0

                    for i, centroid in enumerate(centroids):
                        dist = euclidean_distance(point, centroid)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_cluster = i

                    current_clusters[nearest_cluster].append(point)

                new_centroids = []
                for cluster in current_clusters:
                    if cluster:
                        new_centroids.append(calculate_centroid(cluster))
                    else:
                        idx = len(new_centroids)
                        if idx < len(centroids):
                            new_centroids.append(centroids[idx])
                        else:
                            new_centroids.append(random.choice(points))

                sse = 0
                for i, cluster in enumerate(current_clusters):
                    for point in cluster:
                        sse += euclidean_distance(point, new_centroids[i]) ** 2

                if abs(old_sse - sse) < 0.001 * old_sse:
                    break

                centroids = new_centroids
                old_sse = sse

            sses.append(sse)
            print(f"  k={test_k}, SSE={sse:.4f}")

            if test_k > 1:
                try:
                    points_array = np.array(points)
                    labels = np.zeros(len(points), dtype=int)

                    for i, cluster in enumerate(current_clusters):
                        for point in cluster:
                            point_idx = points.index(point)
                            labels[point_idx] = i

                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(points_array, labels)
                    silhouette_scores.append((test_k, score))
                except Exception as e:
                    silhouette_scores.append((test_k, -1))

        if len(sses) >= 3:
            diffs = [sses[i] - sses[i + 1] for i in range(len(sses) - 1)]

            elbow_found = False
            for i in range(len(diffs) - 1):
                if diffs[i + 1] < 0.3 * diffs[i]:
                    k = i + 2
                    print(f"✅ k-means detected optimal k (between 2-8) using the elbow method: {k}")
                    elbow_found = True
                    break

            if not elbow_found:
                if silhouette_scores:
                    valid_scores = [(k_val, score) for k_val, score in silhouette_scores if score > -1]
                    if valid_scores:
                        k, best_score = max(valid_scores, key=lambda x: x[1])
                        print(f"✅k-means detected optimal k (between 2-8) using the elbow method: {k}")
                    else:
                        k = 3
                else:
                    k = 3
                    print(f"No clear elbow and no silhouette scores, using default k={k}")
        else:
            k = 2
            print(f"Not enough data for elbow analysis, using k={k}")

    print(f"\n=== Performing k-means clustering with k={k} ===")

    centroids = initialize_centroids_dispersed(points, k) if 'initialize_centroids_dispersed' in globals() else [
        points[i] for i in random.sample(range(len(points)), k)]

    max_iter = 100
    old_sse = float('inf')
    current_clusters = []

    for iteration in range(max_iter):
        current_clusters = [[] for _ in range(k)]
        for point in points:
            min_dist = float('inf')
            nearest_cluster = 0

            for i, centroid in enumerate(centroids):
                dist = euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = i

            current_clusters[nearest_cluster].append(point)

        for i, cluster in enumerate(current_clusters):
            if not cluster:
                random_point = random.choice(points)
                current_clusters[i].append(random_point)
                print(f"  Note: Empty cluster {i} detected, assigned random point")

        new_centroids = [calculate_centroid(cluster) for cluster in current_clusters]

        sse = 0
        for i, cluster in enumerate(current_clusters):
            for point in cluster:
                sse += euclidean_distance(point, new_centroids[i]) ** 2

        if abs(old_sse - sse) < 0.001 * old_sse:
            break

        centroids = new_centroids
        old_sse = sse

    clusts.extend(current_clusters)

    print(f"\n=== k-means clustering completed ===")

    return clusts