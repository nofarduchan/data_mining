import csv
import random
import math
import numpy as np
from collections import defaultdict
import os

import pandas as pd


def euclidean_distance(point1, point2):
    """
    Computes the Euclidean distance between two points.

    Args:
        point1 (tuple): Coordinates of the first point.
        point2 (tuple): Coordinates of the second point.

    Returns:
        float: The Euclidean distance between the points.
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))

def generate_data(dim, k, n, out_path, extras={}):
    """
    Generates `n` points in `dim` dimensions, assigns each to the nearest cluster,
    and saves the data to a CSV file.

    Args:
        dim (int): Number of dimensions for each point.
        k (int): Number of clusters to generate.
        n (int): Total number of points to generate.
        out_path (str): Path to save the CSV file.
        extras (dict, optional): Additional parameters (e.g., `min_val`, `max_val`).

    Returns:
        dict: A dictionary where keys are cluster indices and values are lists of assigned points.
    """
    # Default parameters
    min_val = extras.get("min_val", -10)
    max_val = extras.get("max_val", 10)

    # Generate random points
    points = [tuple(random.uniform(min_val, max_val) for _ in range(dim)) for _ in range(n)]

    # Select k random initial cluster centers from the generated points
    initial_centers = random.sample(points, k)

    # Create dictionary to store clusters
    clusters = {i: [] for i in range(k)}

    # Assign each point to the nearest cluster
    for point in points:
        distances = [euclidean_distance(point, center) for center in initial_centers]
        closest_cluster = distances.index(min(distances))
        clusters[closest_cluster].append(point)

    # Save the data to a CSV file with cluster labels
    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for cluster_id, cluster_points in clusters.items():
            for point in cluster_points:
                writer.writerow(list(point) + [cluster_id])  # Save point with its cluster label

    print(f"Generated {n} points in {dim} dimensions, assigned to {k} clusters, and saved to {out_path}")
    return clusters

def load_points(in_path, dim, n=-1, points=[]):
    """
    Load points from a CSV file.

    Args:
        in_path (str): Path to the CSV file
        dim (int): Number of dimensions to read for each point
        n (int, optional): Maximum number of points to read. Defaults to -1 (all points).
        points (list, optional): List to store the loaded points. Defaults to [].

    Returns:
        list: List of loaded points
    """
    points.clear()  # Clear the existing points if any

    with open(in_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        count = 0

        for row in reader:
            if n != -1 and count >= n:
                break

            # Convert the first dim values to float and create a point tuple
            if len(row) >= dim:
                point = tuple(float(row[i]) for i in range(dim))
                points.append(point)
                count += 1

    print(f"Loaded {len(points)} points in {dim} dimensions from {in_path}")
    return points

def h_clustering(k, points, dist=None, clusts=[], cohesion_threshold=None):
    """
    Perform hierarchical clustering (bottom-up) on points.

    Args:
        k (int): Number of desired clusters (or None to auto-determine using cohesion)
        points (list): List of points to cluster
        dist (function, optional): Distance function. Defaults to Euclidean distance.
        clusts (list, optional): Output list to store the resulting clusters. Defaults to [].
        cohesion_threshold (float, optional): Threshold for cluster cohesion. Used when k is None.

    Returns:
        list: List of clusters, where each cluster is a list of points
    """
    if not dist:
        dist = euclidean_distance

    # Initialize clusters: each point is its own cluster
    clusts.clear()
    clusters = [[point] for point in points]

    print(f"Starting hierarchical clustering with {len(clusters)} individual clusters")

    # Run until we have k clusters or auto-determine stopping
    while len(clusters) > 1 and (k is None or len(clusters) > k):
        # Find the two clusters whose union is most cohesive
        best_cohesion = float('-inf')
        best_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate cohesion of merged cluster
                merged_cluster = clusters[i] + clusters[j]

                # Calculate cohesion (negative of average distance between points)
                cohesion = calculate_cluster_cohesion(merged_cluster, dist)

                if cohesion > best_cohesion:
                    best_cohesion = cohesion
                    best_pair = (i, j)

        # If no suitable pair found (shouldn't happen with our metric)
        if best_pair is None:
            break

        i, j = best_pair

        # For auto-determination using cohesion threshold
        if k is None and cohesion_threshold is not None:
            # If next merge would create a cluster with cohesion below threshold, stop
            if best_cohesion <= cohesion_threshold:
                print(
                    f"Stopping at {len(clusters)} clusters: next merge would create a cluster with cohesion {best_cohesion} below threshold {cohesion_threshold}")
                break

        # Merge the clusters with the best cohesion
        merged_cluster = clusters[i] + clusters[j]

        # Remove the two clusters and add the merged one
        if i > j:
            clusters.pop(i)
            clusters.pop(j)
        else:
            clusters.pop(j)
            clusters.pop(i)

        clusters.append(merged_cluster)

    clusts.extend(clusters)

    cluster_sizes = [len(cluster) for cluster in clusts]
    print(f"Hierarchical clustering completed with {len(clusts)} clusters")
    print(f"Cluster sizes: {cluster_sizes}")

    return clusts


# Helper function to calculate cluster cohesion
def calculate_cluster_cohesion(cluster, dist_func):
    """
    Calculate cohesion of a cluster as the negative of average distance between points.
    Higher values indicate better cohesion.

    Args:
        cluster (list): List of points in the cluster
        dist_func (function): Distance function

    Returns:
        float: Cohesion value (higher is better)
    """
    if len(cluster) <= 1:
        return 0  # Single point cluster has perfect cohesion

    # Calculate average distance between all pairs of points
    total_distance = 0
    pairs_count = 0

    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            total_distance += dist_func(cluster[i], cluster[j])
            pairs_count += 1

    avg_distance = total_distance / pairs_count

    # Return negative of average distance as cohesion (higher value = better cohesion)
    return -avg_distance

def assign_points_to_centroids(points, centroids):
    """
    Assign points to the nearest centroid.

    Args:
        points (list): List of points
        centroids (list): List of centroids

    Returns:
        dict: Dictionary mapping centroid index to list of points
    """
    clusters = defaultdict(list)

    for point in points:
        min_distance = float('inf')
        nearest_centroid_idx = 0

        for i, centroid in enumerate(centroids):
            distance = euclidean_distance(point, centroid)
            if distance < min_distance:
                min_distance = distance
                nearest_centroid_idx = i

        clusters[nearest_centroid_idx].append(point)

    return clusters

def calculate_centroid(points):
    """
    Calculate the centroid of a set of points.

    Args:
        points (list): List of points

    Returns:
        tuple: Centroid coordinates
    """
    if not points:
        return None

    dim = len(points[0])
    centroid = tuple(sum(point[i] for point in points) / len(points) for i in range(dim))

    return centroid

def k_means(dim, k, n, points, clusts=[]):
    """
    Perform classic k-means clustering on points with optional elbow method for k selection.

    Args:
        dim (int): Number of dimensions for each point
        k (int or None): Number of desired clusters (or None to auto-determine using elbow method)
        n (int): Number of points (not used but kept for interface consistency)
        points (list): List of points to cluster
        clusts (list, optional): Output list to store the resulting clusters. Defaults to [].

    Returns:
        list: List of clusters, where each cluster is a list of points
    """

    # Clear output clusters list
    clusts.clear()

    # If k is None, determine the optimal k using the elbow method
    if k is None:
        max_k = min(10, len(points) // 2)  # Try up to 10 clusters or half the points
        sses = []

        print("Auto-determining optimal number of clusters using elbow method...")
        for test_k in range(1, max_k + 1):
            # Initialize centroids randomly
            random_indices = random.sample(range(len(points)), test_k)
            centroids = [points[i] for i in random_indices]

            # Run k-means iterations
            old_sse = float('inf')
            sse = 0
            max_iter = 100

            for _ in range(max_iter):
                # Assign points to nearest centroids
                current_clusters = [[] for _ in range(test_k)]

                for point in points:
                    # Find the nearest centroid
                    min_dist = float('inf')
                    nearest_cluster = 0

                    for i, centroid in enumerate(centroids):
                        dist = euclidean_distance(point, centroid)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_cluster = i

                    current_clusters[nearest_cluster].append(point)

                # Calculate new centroids
                new_centroids = []
                for cluster in current_clusters:
                    if cluster:
                        new_centroids.append(calculate_centroid(cluster))
                    else:
                        # For empty clusters, keep the old centroid
                        idx = len(new_centroids)
                        if idx < len(centroids):
                            new_centroids.append(centroids[idx])
                        else:
                            # If somehow we lost track of centroids, pick a random point
                            new_centroids.append(random.choice(points))

                # Calculate SSE
                sse = 0
                for i, cluster in enumerate(current_clusters):
                    for point in cluster:
                        sse += euclidean_distance(point, new_centroids[i]) ** 2

                # Check for convergence
                if abs(old_sse - sse) < 0.001:
                    break

                centroids = new_centroids
                old_sse = sse

            sses.append(sse)
            print(f"  k={test_k}, SSE={sse:.4f}")

            # Simple elbow detection (must have at least 3 points to detect an elbow)
            if len(sses) >= 3:
                diff1 = sses[-3] - sses[-2]  # Improvement from k-2 to k-1
                diff2 = sses[-2] - sses[-1]  # Improvement from k-1 to k

                # If the improvement is marginal (less than 20% of previous improvement), stop
                if diff2 < 0.2 * diff1:
                    k = test_k - 1
                    print(f"Auto-detected optimal number of clusters: {k}")
                    break

        # If no elbow was found, choose a reasonable k
        if k is None:
            k = max(2, min(max_k // 2, 5))
            print(f"No clear elbow detected, using k={k}")

    # Now perform k-means with the determined k
    print(f"Starting classic k-means with k={k}")

    # Randomly initialize k centroids
    random_indices = random.sample(range(len(points)), k)
    centroids = [points[i] for i in random_indices]

    # Iterate until convergence or max iterations
    max_iterations = 100
    for iteration in range(max_iterations):
        # Assign points to nearest centroids
        new_clusters = [[] for _ in range(k)]

        for point in points:
            # Find the nearest centroid
            min_dist = float('inf')
            nearest_cluster = 0

            for i, centroid in enumerate(centroids):
                dist = euclidean_distance(point, centroid)
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = i

            new_clusters[nearest_cluster].append(point)

        # Check if any clusters are empty
        for i, cluster in enumerate(new_clusters):
            if not cluster:
                # Assign a random point to the empty cluster
                random_point = random.choice(points)
                new_clusters[i].append(random_point)

        # Calculate new centroids
        new_centroids = [calculate_centroid(cluster) for cluster in new_clusters]

        # Check for convergence (if centroids haven't changed)
        if all(old == new for old, new in zip(centroids, new_centroids)):
            print(f"K-means converged after {iteration + 1} iterations")
            break

        centroids = new_centroids

    else:
        print(f"K-means reached maximum iterations ({max_iterations})")

    # Copy results to output parameter
    clusts.extend(new_clusters)

    # Print cluster sizes
    cluster_sizes = [len(cluster) for cluster in clusts]
    print(f"K-means clustering completed with {len(clusts)} clusters")
    print(f"Cluster sizes: {cluster_sizes}")

    return clusts

def save_points(clusts, out_path, out_path_tagged):
    """
    Save clusters to CSV files.

    Args:
        clusts (list): List of clusters, where each cluster is a list of points
        out_path (str): Path to save all points in random order without cluster labels
        out_path_tagged (str): Path to save all points with cluster labels
    """
    # Create list of all points
    all_points = []
    total_points = 0
    for i, cluster in enumerate(clusts):
        for point in cluster:
            all_points.append((point, i))
            total_points += 1

    # Save all points in random order without cluster labels
    random.shuffle(all_points)
    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for point, _ in all_points:
            writer.writerow(point)

    # Save all points with cluster labels
    with open(out_path_tagged, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, cluster in enumerate(clusts):
            for point in cluster:
                writer.writerow(list(point) + [i])

    print(f"Saved {total_points} points to {out_path}")
    print(f"Saved {total_points} points with cluster labels to {out_path_tagged}")


def generate_large_data(dim, k, n, out_path, chunk_size=10000, extras={}):
    """
    Generate a large amount of data in chunks using generate_data function to avoid memory issues.

    Args:
        dim (int): Number of dimensions for each point
        k (int): Number of clusters to generate
        n (int): Total number of points to generate
        out_path (str): Path to save the CSV file
        chunk_size (int): Number of points to generate in each chunk
        extras (dict, optional): Additional parameters for customization

    Returns:
        int: Number of points written to the file
    """
    import os

    # Initialize file
    with open(out_path, 'w', newline='') as csvfile:
        pass  # Create empty file

    points_written = 0
    chunks_count = (n + chunk_size - 1) // chunk_size  # Ceiling division

    print(f"Generating {n} points in {dim} dimensions with {k} clusters")
    print(f"Using {chunks_count} chunks of up to {chunk_size} points each")

    for chunk_idx in range(chunks_count):
        # Calculate how many points to generate in this chunk
        remaining = n - points_written
        current_chunk_size = min(chunk_size, remaining)

        if current_chunk_size <= 0:
            break

        # Generate a temporary file for this chunk
        temp_file = f"{out_path}_temp_chunk_{chunk_idx}.csv"

        # Generate data for this chunk
        print(f"Generating chunk {chunk_idx + 1}/{chunks_count} with {current_chunk_size} points...")

        # Use generate_data to create this chunk
        clusters = generate_data(dim, k, current_chunk_size, temp_file, extras)

        # Append the chunk data to the final output file
        with open(temp_file, 'r') as chunk_file, open(out_path, 'a', newline='') as out_file:
            for line in chunk_file:
                out_file.write(line)
                points_written += 1

        # Remove temporary file
        try:
            os.remove(temp_file)
        except:
            print(f"Warning: Could not remove temporary file {temp_file}")

        print(f"Chunk {chunk_idx + 1} completed. Total points written: {points_written}/{n}")

    print(f"Completed writing {points_written} points to {out_path}")
    return points_written


def bfr_cluster(dim, k, n, block_size, in_path, out_path, mahalanobis_threshold=3.0):
    """
    Performs BFR (Bradley-Fayyad-Reina) clustering on large datasets.

    Parameters:
    - dim: Dimensionality of the data
    - k: Number of clusters
    - n: Number of data points
    - block_size: Size of data chunks to process
    - in_path: Input data file path
    - out_path: Output clusters file path
    - mahalanobis_threshold: Threshold for considering a point part of a cluster
    """
    # מבני נתונים להחזקת המידע שיידרש לאלגוריתם
    DS = {}  # Discard Set: נקודות שכבר סווגו לאשכולות
    RS = []  # Retained Set: נקודות שלא הצלחנו לסווג בינתיים
    CS = []  # Compression Set: קבוצות נקודות קרובות שאולי יוכלו להתמזג

    # קריאת החלק הראשון של הנתונים
    data_chunks = pd.read_csv(in_path, chunksize=block_size, header=None)
    first_chunk = next(data_chunks)

    # שימוש ב-k-means על החלק הראשון ליצירת k אשכולות ראשוניים
    initial_points = first_chunk.iloc[:, :dim].values

    if len(initial_points) < k:
        raise ValueError(f"Initial chunk size ({len(initial_points)}) is smaller than k ({k})")

    # אתחול k-means++ לבחירת מרכזים התחלתיים טובים יותר
    centers = [initial_points[0]]
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(p - c) ** 2 for c in centers]) for p in initial_points])
        probs = distances / distances.sum()
        cumprobs = np.cumsum(probs)
        r = random.random()
        ind = np.searchsorted(cumprobs, r)
        centers.append(initial_points[ind])

    # ניהול איטרציות k-means
    for _ in range(5):  # מספר איטרציות מוגבל של k-means
        # שיוך נקודות למרכז הקרוב ביותר
        assignments = np.argmin([np.linalg.norm(initial_points - c, axis=1) for c in centers], axis=0)

        # עדכון המרכזים
        for j in range(k):
            if np.sum(assignments == j) > 0:
                centers[j] = np.mean(initial_points[assignments == j], axis=0)

    # אתחול ה-DS עם הסטטיסטיקות של האשכולות הראשוניים
    for j in range(k):
        cluster_points = initial_points[assignments == j]
        if len(cluster_points) > 0:
            DS[j] = {
                'n': len(cluster_points),
                'sum': np.sum(cluster_points, axis=0),
                'sum_sq': np.sum(cluster_points ** 2, axis=0),
                'centroid': centers[j],
                'variance': np.var(cluster_points, axis=0) if len(cluster_points) > 1 else np.ones(dim)
            }

    # עיבוד שאר החלקים
    for chunk in data_chunks:
        chunk_points = chunk.iloc[:, :dim].values

        # טיפול בכל נקודה בחלק הנוכחי
        for point in chunk_points:
            assigned = False

            # ניסיון לשייך לאשכולות קיימים ב-DS
            min_mahalanobis = float('inf')
            best_cluster = -1

            for cluster_id, stats in DS.items():
                # חישוב מרחק מהלנוביס (או קירוב לו)
                diff = point - stats['centroid']
                if np.any(stats['variance'] <= 0):
                    # טיפול במקרה של אשכול עם נקודה אחת (אין שונות)
                    mahalanobis_dist = np.linalg.norm(diff)
                else:
                    # מרחק מהלנוביס: מנורמל לפי השונות של האשכול
                    mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / stats['variance']))

                if mahalanobis_dist < min_mahalanobis:
                    min_mahalanobis = mahalanobis_dist
                    best_cluster = cluster_id

            # אם המרחק מתחת לסף, שייך את הנקודה לאשכול
            if min_mahalanobis < mahalanobis_threshold:
                # עדכון הסטטיסטיקות ב-DS
                DS[best_cluster]['n'] += 1
                DS[best_cluster]['sum'] += point
                DS[best_cluster]['sum_sq'] += point ** 2
                DS[best_cluster]['centroid'] = DS[best_cluster]['sum'] / DS[best_cluster]['n']
                DS[best_cluster]['variance'] = (DS[best_cluster]['sum_sq'] / DS[best_cluster]['n'] -
                                                DS[best_cluster]['centroid'] ** 2)
                # טיפול במקרה של שונות שלילית (בעיות מספריות)
                DS[best_cluster]['variance'] = np.maximum(DS[best_cluster]['variance'], 1e-6)
                assigned = True

            # אם לא הצלחנו לשייך לאשכול קיים, מוסיפים ל-RS
            if not assigned:
                RS.append(point)

        # ניסיון לבצע k-means על ה-RS אם הוא גדול מספיק
        if len(RS) >= 5 * k:
            rs_array = np.array(RS)

            # חלוקת RS ל-k קבוצות
            # אתחול k-means++
            rs_centers = [rs_array[0]]
            for _ in range(1, k):
                rs_distances = np.array([min([np.linalg.norm(p - c) ** 2 for c in rs_centers]) for p in rs_array])
                rs_probs = rs_distances / (rs_distances.sum() + 1e-10)  # מניעת חלוקה באפס
                rs_cumprobs = np.cumsum(rs_probs)
                r = random.random()
                rs_ind = np.searchsorted(rs_cumprobs, r)
                if rs_ind >= len(rs_array):
                    rs_ind = len(rs_array) - 1
                rs_centers.append(rs_array[rs_ind])

            # איטרציות k-means מוגבלות
            for _ in range(5):
                rs_assignments = np.argmin([np.linalg.norm(rs_array - c, axis=1) for c in rs_centers], axis=0)

                for j in range(k):
                    if np.sum(rs_assignments == j) > 0:
                        rs_centers[j] = np.mean(rs_array[rs_assignments == j], axis=0)

            # יצירת קבוצות CS חדשות או שילוב עם קבוצות קיימות
            new_RS = []

            for j in range(k):
                cluster_points = rs_array[rs_assignments == j]
                if len(cluster_points) >= 5:  # מספיק נקודות ליצירת קבוצת CS
                    cs_stats = {
                        'n': len(cluster_points),
                        'sum': np.sum(cluster_points, axis=0),
                        'sum_sq': np.sum(cluster_points ** 2, axis=0),
                        'centroid': rs_centers[j],
                        'variance': np.var(cluster_points, axis=0) if len(cluster_points) > 1 else np.ones(dim)
                    }
                    CS.append(cs_stats)
                else:
                    # שומרים נקודות בודדות ב-RS
                    new_RS.extend(cluster_points)

            RS = new_RS

            # ניסיון למזג קבוצות CS קרובות
            i = 0
            while i < len(CS):
                merged = False
                j = i + 1
                while j < len(CS):
                    # חישוב מרחק בין מרכזי CS
                    cs_dist = np.linalg.norm(CS[i]['centroid'] - CS[j]['centroid'])

                    # אם המרחק קטן מספיק, מיזוג הקבוצות
                    if cs_dist < mahalanobis_threshold:
                        # מיזוג הסטטיסטיקות
                        CS[i]['n'] += CS[j]['n']
                        CS[i]['sum'] += CS[j]['sum']
                        CS[i]['sum_sq'] += CS[j]['sum_sq']
                        CS[i]['centroid'] = CS[i]['sum'] / CS[i]['n']
                        CS[i]['variance'] = (CS[i]['sum_sq'] / CS[i]['n'] - CS[i]['centroid'] ** 2)
                        CS[i]['variance'] = np.maximum(CS[i]['variance'], 1e-6)  # מניעת שונות שלילית

                        # הסרת הקבוצה שמוזגה
                        del CS[j]
                        merged = True
                    else:
                        j += 1

                if not merged:
                    i += 1

            # ניסיון להעביר קבוצות CS לאשכולות DS
            i = 0
            while i < len(CS):
                min_cs_mahalanobis = float('inf')
                best_cs_cluster = -1

                for cluster_id, stats in DS.items():
                    # חישוב מרחק מהלנוביס בין מרכזי CS ו-DS
                    diff = CS[i]['centroid'] - stats['centroid']
                    if np.any(stats['variance'] <= 0):
                        cs_mahalanobis_dist = np.linalg.norm(diff)
                    else:
                        cs_mahalanobis_dist = np.sqrt(np.sum((diff ** 2) / stats['variance']))

                    if cs_mahalanobis_dist < min_cs_mahalanobis:
                        min_cs_mahalanobis = cs_mahalanobis_dist
                        best_cs_cluster = cluster_id

                # אם המרחק מתחת לסף, מיזוג CS עם DS
                if min_cs_mahalanobis < mahalanobis_threshold:
                    # מיזוג הסטטיסטיקות
                    DS[best_cs_cluster]['n'] += CS[i]['n']
                    DS[best_cs_cluster]['sum'] += CS[i]['sum']
                    DS[best_cs_cluster]['sum_sq'] += CS[i]['sum_sq']
                    DS[best_cs_cluster]['centroid'] = DS[best_cs_cluster]['sum'] / DS[best_cs_cluster]['n']
                    DS[best_cs_cluster]['variance'] = (DS[best_cs_cluster]['sum_sq'] / DS[best_cs_cluster]['n'] -
                                                       DS[best_cs_cluster]['centroid'] ** 2)
                    DS[best_cs_cluster]['variance'] = np.maximum(DS[best_cs_cluster]['variance'], 1e-6)

                    # הסרת הקבוצה שמוזגה
                    del CS[i]
                else:
                    i += 1

    # בסיום כל הנתונים, ניסיון אחרון לשייך את כל הנקודות ב-RS ואת כל הקבוצות ב-CS
    # שיוך RS
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

        # שיוך לאשכול הקרוב ביותר ללא סף מרחק
        final_assignments[i] = best_cluster

    # שיוך CS (יצירת אשכולות חדשים אם צריך)
    next_cluster_id = max(DS.keys()) + 1 if DS else 0

    for cs_stats in CS:
        # בדיקה אם CS קרובה מספיק לאשכול קיים
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

        # אם קרובה מספיק, מיזוג עם DS, אחרת יצירת אשכול חדש
        if min_cs_mahalanobis < mahalanobis_threshold:
            DS[best_cs_cluster]['n'] += cs_stats['n']
            DS[best_cs_cluster]['sum'] += cs_stats['sum']
            DS[best_cs_cluster]['sum_sq'] += cs_stats['sum_sq']
            DS[best_cs_cluster]['centroid'] = DS[best_cs_cluster]['sum'] / DS[best_cs_cluster]['n']
            DS[best_cs_cluster]['variance'] = (DS[best_cs_cluster]['sum_sq'] / DS[best_cs_cluster]['n'] -
                                               DS[best_cs_cluster]['centroid'] ** 2)
            DS[best_cs_cluster]['variance'] = np.maximum(DS[best_cs_cluster]['variance'], 1e-6)
        else:
            DS[next_cluster_id] = cs_stats
            next_cluster_id += 1

    # שמירת התוצאות לקובץ
    # קריאת כל הנתונים שוב ושיוך לאשכולות הסופיים
    data_chunks = pd.read_csv(in_path, chunksize=block_size, header=None)

    with open(out_path, "w", newline="") as f:
        point_idx = 0

        for chunk in data_chunks:
            chunk_points = chunk.iloc[:, :dim].values

            for point in chunk_points:
                min_dist = float('inf')
                best_ds_cluster = 0

                for cluster_id, stats in DS.items():
                    dist = np.linalg.norm(point - stats['centroid'])
                    if dist < min_dist:
                        min_dist = dist
                        best_ds_cluster = cluster_id

                # כתיבת הנקודה עם תג האשכול
                f.write(",".join(map(str, point)) + f",{best_ds_cluster}\n")
                point_idx += 1

    # החזרת מילון עם מרכזי האשכולות
    return {cluster_id: stats['centroid'] for cluster_id, stats in DS.items()}


def cure_cluster(dim, k, n, block_size, in_path, out_path, num_representatives=10, shrink_factor=0.3):
    """
    Performs CURE clustering on large datasets.

    Parameters:
    - dim: Dimensionality of the data
    - k: Number of clusters
    - n: Number of data points
    - block_size: Size of data chunks to process
    - in_path: Input data file path
    - out_path: Output clusters file path
    - num_representatives: Number of representative points per cluster
    - shrink_factor: Shrinking factor for representative points (0 to 1)
    """
    # קריאת מדגם נתונים ראשוני
    sample_size = min(5000, n)  # מדגם בגודל סביר
    sampled_points = []

    # קריאת מדגם מהקובץ
    data_chunks = pd.read_csv(in_path, chunksize=block_size, header=None)
    points_collected = 0

    for chunk in data_chunks:
        if points_collected >= sample_size:
            break

        chunk_points = chunk.iloc[:, :dim].values
        points_to_collect = min(len(chunk_points), sample_size - points_collected)

        if points_to_collect > 0:
            indices = random.sample(range(len(chunk_points)), points_to_collect)
            sampled_points.extend(chunk_points[i] for i in indices)
            points_collected += points_to_collect

    # המרה למערך numpy
    sampled_points = np.array(sampled_points)

    # אתחול אשכולות כאשר כל נקודה היא אשכול נפרד
    clusters = []
    for i in range(len(sampled_points)):
        clusters.append({
            'points': [sampled_points[i]],
            'mean': sampled_points[i],
            'representatives': [sampled_points[i]]
        })

    # איחוד אשכולות עד להגעה ל-k אשכולות
    while len(clusters) > k:
        # מציאת זוג האשכולות הקרובים ביותר
        min_distance = float('inf')
        merge_pair = (0, 1)

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # חישוב המרחק המינימלי בין נקודות מייצגות
                min_rep_distance = float('inf')

                for rep_i in clusters[i]['representatives']:
                    for rep_j in clusters[j]['representatives']:
                        dist = np.linalg.norm(rep_i - rep_j)
                        if dist < min_rep_distance:
                            min_rep_distance = dist

                if min_rep_distance < min_distance:
                    min_distance = min_rep_distance
                    merge_pair = (i, j)

        # איחוד האשכולות הקרובים ביותר
        i, j = merge_pair
        merged_points = clusters[i]['points'] + clusters[j]['points']

        # חישוב המרכז החדש
        merged_mean = np.mean(merged_points, axis=0)

        # בחירת נקודות מייצגות
        # מציאת הנקודות המרוחקות ביותר זו מזו
        merged_array = np.array(merged_points)
        representatives = []

        if len(merged_points) <= num_representatives:
            representatives = merged_points
        else:
            # בחירת נקודות מייצגות מגוונות
            # אסטרטגיה: בחירת הנקודות המרוחקות ביותר מהמרכז
            distances_from_mean = [np.linalg.norm(p - merged_mean) for p in merged_array]
            sorted_indices = np.argsort(distances_from_mean)[::-1]  # מיון יורד
            representatives = [merged_array[i] for i in sorted_indices[:num_representatives]]

        # התכווצות הנקודות המייצגות לכיוון המרכז
        shrunk_representatives = []
        for rep in representatives:
            # התכווצות לפי הנוסחה: rep = rep + shrink_factor * (mean - rep)
            shrunk_rep = rep + shrink_factor * (merged_mean - rep)
            shrunk_representatives.append(shrunk_rep)

        # יצירת האשכול המאוחד
        merged_cluster = {
            'points': merged_points,
            'mean': merged_mean,
            'representatives': shrunk_representatives
        }

        # שמירת האשכול החדש והסרת האשכולות שאוחדו
        clusters[i] = merged_cluster
        del clusters[j]

    # מיון שאר הנקודות מהקובץ
    # איפוס קריאת הקובץ
    data_chunks = pd.read_csv(in_path, chunksize=block_size, header=None)

    # שיוך נקודות לאשכולות
    final_clusters = [[] for _ in range(k)]

    for chunk in data_chunks:
        chunk_points = chunk.iloc[:, :dim].values

        for point in chunk_points:
            # מציאת האשכול הקרוב ביותר על ידי חישוב המרחק המינימלי לנקודה מייצגת
            min_distance = float('inf')
            best_cluster = 0

            for cluster_idx, cluster in enumerate(clusters):
                for rep in cluster['representatives']:
                    dist = np.linalg.norm(point - rep)
                    if dist < min_distance:
                        min_distance = dist
                        best_cluster = cluster_idx

            final_clusters[best_cluster].append(point)

    # שמירת התוצאות לקובץ
    with open(out_path, "w", newline="") as f:
        for cluster_idx, cluster_points in enumerate(final_clusters):
            for point in cluster_points:
                f.write(",".join(map(str, point)) + f",{cluster_idx}\n")

    return final_clusters

def main():
    """
    Demonstrate the clustering functionality.
    """
    print("=" * 50)
    print("CLUSTERING ALGORITHMS DEMONSTRATION")
    print("=" * 50)

    # Create a directory for outputs if it doesn't exist
    output_dir = "clustering_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parameters
    dim = 4  # 2D points for easy visualization
    k = 4  # 3 clusters
    n = 100  # 100 points

    # 1. Generate data
    print("\n1. Generating data...")
    data_path = os.path.join(output_dir, "generated_data.csv")
    points = generate_data(dim, k, n, data_path, extras={'std_dev': 0.8})

    # 2. Load points
    print("\n2. Loading points...")
    loaded_points = []
    load_points(data_path, dim, n, points=loaded_points)

    # 3. Hierarchical clustering
    print("\n3. Performing hierarchical clustering...")
    h_clusters = []
    h_clustering(k, loaded_points, clusts=h_clusters)

    # Save hierarchical clustering results
    h_out_path = os.path.join(output_dir, "h_clustered.csv")
    h_out_path_tagged = os.path.join(output_dir, "h_clustered_tagged.csv")
    save_points(h_clusters, h_out_path, h_out_path_tagged)

    # 4. K-means clustering
    print("\n4. Performing k-means clustering...")
    k_clusters = []
    k_means(dim, k, n, loaded_points, clusts=k_clusters)

    # Save k-means clustering results
    k_out_path = os.path.join(output_dir, "k_clustered.csv")
    k_out_path_tagged = os.path.join(output_dir, "k_clustered_tagged.csv")
    save_points(k_clusters, k_out_path, k_out_path_tagged)

    # # 5. Auto-determining clusters
    # print("\n5. Performing k-means with auto-determined k...")
    # auto_k_clusters = []
    # k_means(dim, None, n, loaded_points, clusts=auto_k_clusters)
    #
    # # Save auto-k results
    # auto_k_out_path = os.path.join(output_dir, "auto_k_clustered.csv")
    # auto_k_out_path_tagged = os.path.join(output_dir, "auto_k_clustered_tagged.csv")
    # save_points(auto_k_clusters, auto_k_out_path, auto_k_out_path_tagged)

    print("\nDemonstration completed. All outputs saved to the", output_dir, "directory.")


# Run the demo
if __name__ == "__main__":
    main()