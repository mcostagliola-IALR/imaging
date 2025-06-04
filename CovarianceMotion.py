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
    params:
    tip_groups: The first return entry in the output tuple from group_tips_by_distance()
    latex_output_path: A string that the resulting LaTeX file should be saved as
    '''
    lines = [
        "\\documentclass{article}",
        "\\usepackage{amsmath}",
        "\\begin{document}",
        "\\section*{Leaf Tip Covariance Analysis}"
    ]

    for cluster, tips in tip_groups.items():
        if len(tips) < 2:
            lines.append(f"\\subsection*{{{cluster}}}")
            lines.append("Not enough tips to compute covariance matrix.\\")
            continue

        coords_array = np.array(tips)
        cov_matrix = np.cov(coords_array.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)


        # Exports A LaTeX file containing all data
        lines.append(f"\\subsection*{{{cluster}}}")
        lines.append("Covariance matrix:")
        lines.append("\\[\\begin{bmatrix}" + \
                     f"{cov_matrix[0,0]:.2f} & {cov_matrix[0,1]:.2f} \\" + \
                     f"{cov_matrix[1,0]:.2f} & {cov_matrix[1,1]:.2f}" +
                     "\\end{bmatrix}\\]")
        lines.append("Eigenvalues (variances): \\" + ", \\".join(f"{val:.2f}" for val in eigenvalues) + "\\\\")
        lines.append("Eigenvectors (directions):")
        lines.append("\\[\\begin{bmatrix}" + \
                     f"{eigenvectors[0,0]:.2f} & {eigenvectors[0,1]:.2f} \\" + \
                     f"{eigenvectors[1,0]:.2f} & {eigenvectors[1,1]:.2f}" +
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



# Example usage:
tips, full_df = load_tips_from_csv(csv_path)
eps = suggest_eps(tips)
grouped_tips, tips_array, labels = group_tips_by_distance(tips, eps=eps, min_samples=2)
compute_and_export_covariance_matrix(grouped_tips, latex_output_path="covariance_report.tex")
plot_clusters_with_covariance(grouped_tips, tips_array, labels)