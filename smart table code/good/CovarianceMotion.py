from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs
import numpy as np
import os
import cv2
import customtkinter as ctk
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
from scipy.spatial import distance
from skimage.morphology import thin
from skimage.graph import route_through_array
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import defaultdict

root = ctk.CTk()
csv_path = ctk.filedialog.askopenfilename(title="Select tip CSV file")

def load_tips_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'Tip_X' in df.columns and 'Tip_Y' in df.columns:
        return list(zip(df['Tip_X'], df['Tip_Y'])), df
    else:
        raise ValueError("CSV must contain 'Tip_X' and 'Tip_Y' columns.")

def suggest_eps(tips, k=4, show_plot=True):
    from sklearn.neighbors import NearestNeighbors
    from kneed import KneeLocator

    tips_array = np.array(tips)
    nbrs = NearestNeighbors(n_neighbors=k).fit(tips_array)
    distances, _ = nbrs.kneighbors(tips_array)
    k_distances = np.sort(distances[:, k-1])

    # Detect the knee point
    knee_locator = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
    optimal_eps = k_distances[knee_locator.knee] if knee_locator.knee is not None else 50  # fallback

    if show_plot:
        plt.figure()
        plt.plot(k_distances, label='k-distance')
        if knee_locator.knee:
            plt.axvline(knee_locator.knee, color='r', linestyle='--', label=f'knee: eps={optimal_eps:.2f}')
        plt.title(f"k-Distance Plot (k={k})")
        plt.xlabel("Sorted points")
        plt.ylabel(f"{k}-NN Distance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return optimal_eps

def group_tips_by_distance(tips, eps=50, min_samples=2):
    '''
    params: 
    tips: A .csv extensioned file containing (x,y) coordinate pairs of tips including a header row
    eps: The euclidean distance threshold for two samples to be considered in the same cluster
    min_samples: The minimum required samples for a point in a cluster to be considered a core point

    '''
    tips_array = np.array(tips)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(tips_array)
    labels = clustering.labels_
    grouped = {}
    print(f"Total tips: {len(tips)}")
    print(f"Noise points: {sum(labels == -1)}")
    print(f"Detected clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
    for label in set(labels):
        if label == -1:
            continue  # Noise points
        grouped[f"cluster_{label}"] = tips_array[labels == label].tolist()
    return grouped, tips_array, labels

def compute_and_export_covariance_matrix(tip_groups, latex_output_path="covariance_report.tex"):
    '''
    tip_groups: dict of clusters -> list of (x, y) points
    '''
    lines = [
        "\\documentclass{article}",
        "\\usepackage{amsmath}",
        "\\begin{document}",
        "\\section*{Leaf Tip Covariance Analysis}"
    ]

    covariances = []

    for cluster, tips in tip_groups.items():
        cluster = cluster.replace('_', ' ').title()
        if len(tips) < 2:
            lines.append(f"\\subsection*{{{cluster}}}")
            lines.append("Not enough tips to compute covariance matrix.\\")
            continue

        coords_array = np.array(tips)
        cov_matrix = np.cov(coords_array.T)
        covariances.append(cov_matrix)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        lines.append(f"\\subsection*{{{cluster}}}")
        lines.append("Covariance matrix:")
        lines.append("\\[\\begin{bmatrix}" +
                     f"{cov_matrix[0,0]:.2f} & {cov_matrix[0,1]:.2f} \\\\" +
                     f"{cov_matrix[1,0]:.2f} & {cov_matrix[1,1]:.2f}" +
                     "\\end{bmatrix}\\]")
        lines.append("Eigenvalues (variances): $ " + ", $".join(f"{val:.2f}" for val in eigenvalues) + "\\\\")
        lines.append("Eigenvectors (directions):")
        lines.append("\\[\\begin{bmatrix}" +
                     f"{eigenvectors[0,0]:.2f} & {eigenvectors[0,1]:.2f} \\\\" +
                     f"{eigenvectors[1,0]:.2f} & {eigenvectors[1,1]:.2f}" +
                     "\\end{bmatrix}\\]")

    # --- Average covariance matrix across all clusters ---
    if covariances:
        avg_cov = np.mean(covariances, axis=0)
        lines.append("\\section*{Average Covariance Matrix Across All Clusters}")
        lines.append("\\[\\begin{bmatrix}" +
                     f"{avg_cov[0,0]:.2f} & {avg_cov[0,1]:.2f} \\\\" +
                     f"{avg_cov[1,0]:.2f} & {avg_cov[1,1]:.2f}" +
                     "\\end{bmatrix}\\]")

    lines.append("\\end{document}")
    with open(latex_output_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"LaTeX report written to {latex_output_path}")

def plot_clusters_with_covariance(tip_groups, tips_array=None, labels=None):
    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('Set1', len(tip_groups))

    for i, (cluster, tips) in enumerate(tip_groups.items()):
        tips_np = np.array(tips)
        if len(tips) < 2:
            continue

        plt.scatter(tips_np[:, 0], tips_np[:, 1], s=50, label=cluster, color=colors(i))

        # Covariance ellipse
        cov = np.cov(tips_np.T)
        eigenvals, eigenvecs = np.linalg.eig(cov)
        angle = np.degrees(np.arctan2(*eigenvecs[:,0][::-1]))
        width, height = 2 * np.sqrt(eigenvals)

        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=np.mean(tips_np, axis=0), width=width, height=height,
                          angle=angle, edgecolor=colors(i), fc='None', lw=2, ls='--')
        plt.gca().add_patch(ellipse)

    # Plot noise points in gray
    if tips_array is not None and labels is not None:
        noise_points = tips_array[labels == -1]
        if len(noise_points) > 0:
            plt.scatter(noise_points[:, 0], noise_points[:, 1], s=30, color='gray', alpha=0.5, label="Noise")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Leaf Tip Clusters and Covariance Ellipses")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()

    # --- Add: Heatmap and magnitude plots for each cluster ---
    disp_x = []
    disp_y = []
    disp_mag = []
    for cluster, tips in tip_groups.items():
        tips_np = np.array(tips)
        if len(tips_np) < 2:
            continue
        # Sort by X then Y for a simple trajectory (if time info is not available)
        tips_sorted = tips_np[np.lexsort((tips_np[:,1], tips_np[:,0]))]
        for i in range(1, len(tips_sorted)):
            disp_x.append(tips_sorted[i-1,0])
            disp_y.append(tips_sorted[i-1,1])
            disp_mag.append(np.hypot(tips_sorted[i,0]-tips_sorted[i-1,0], tips_sorted[i,1]-tips_sorted[i-1,1]))
    if disp_x and disp_y and disp_mag:
        plt.figure(figsize=(8,6))
        plt.hist2d(disp_x, disp_y, weights=disp_mag, bins=30, cmap='hot')
        plt.colorbar(label='Motion magnitude')
        plt.gca().invert_yaxis()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Heatmap of Motion Magnitude')
        plt.show()

        # Magnitude plot (distribution of displacement magnitudes)
        plt.figure(figsize=(8,4))
        plt.hist(disp_mag, bins=30, color='dodgerblue', alpha=0.7)
        plt.xlabel('Displacement magnitude (pixels)')
        plt.ylabel('Count')
        plt.title('Distribution of Displacement Magnitudes')
        plt.grid(True)
        plt.show()

def extract_date(image_name):
    # Assumes date is the first 10 characters: 'MM-DD-YYYY'
    return image_name[:10]

def group_tips_by_day(full_df):
    day_groups = defaultdict(list)
    for _, row in full_df.iterrows():
        day = extract_date(row['Image_Name'])
        day_groups[day].append((row['Tip_X'], row['Tip_Y']))
    return day_groups

# --- Choose analysis mode here ---
# Set to True for per-day analysis, False for entire experiment
PER_DAY = True
AVERAGE_COVARIANCE = True  # <--- Add this option

tips, full_df = load_tips_from_csv(csv_path)

all_covariances = []
temp_covariances = []

if PER_DAY:
    day_groups = group_tips_by_day(full_df)
    for day, tips in day_groups.items():
        print(f"\nProcessing day: {day}")
        eps = suggest_eps(tips, show_plot=False)
        grouped_tips, tips_array, labels = group_tips_by_distance(tips, eps=eps, min_samples=2)
        for cluster, cluster_tips in grouped_tips.items():
            if len(cluster_tips) > 1:
                cov = np.cov(np.array(cluster_tips).T)
                print(f"Covariance for {cluster} on {day}:\n{cov}\n")
                all_covariances.append(cov)
                temp_covariances.append(cov)
        print(f'Average covariance for {day}:\n {np.mean(temp_covariances, axis=0) if temp_covariances else "N/A"}\n')
        compute_and_export_covariance_matrix(
            grouped_tips, 
            latex_output_path=f"covariance_report_{day}.tex"
        )
        #plot_clusters_with_covariance(grouped_tips, tips_array, labels)
        temp_covariances.clear()
else:
    print("\nProcessing entire experiment")
    tips = full_df[['Tip_X', 'Tip_Y']].values
    eps = suggest_eps(tips, show_plot=False)
    grouped_tips, tips_array, labels = group_tips_by_distance(tips, eps=eps, min_samples=2)
    for cluster, cluster_tips in grouped_tips.items():
        if len(cluster_tips) > 1:
            cov = np.cov(np.array(cluster_tips).T)
            print(f"Covariance for {cluster}:\n{cov}\n")
            all_covariances.append(cov)
    compute_and_export_covariance_matrix(
        grouped_tips, 
        latex_output_path="covariance_report_experiment.tex"
    )
    plot_clusters_with_covariance(grouped_tips, tips_array, labels)

# --- Average all covariance matrices ---
if AVERAGE_COVARIANCE and all_covariances:
    avg_cov = np.mean(all_covariances, axis=0)
    print("\nAverage covariance matrix across all clusters:")
    print(avg_cov)