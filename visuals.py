import numpy as np
import matplotlib.pyplot as plt
import os

def plot_distribution_pie(outcome_counts, save_path=None):
    labels = list(outcome_counts.keys())
    counts = np.array([outcome_counts[k] for k in labels])
    colors = {"left": "red", "center": "orange", "right": "blue"}
    plt.figure()
    plt.pie(counts, labels=labels, colors=[colors[k] for k in labels],
            autopct='%1.1f%%', startangle=90)
    plt.title("Outcome Distribution (Pie Chart)")
    plt.axis('equal')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_distribution_bar(outcome_counts, save_path=None):
    labels = list(outcome_counts.keys())
    counts = np.array([outcome_counts[k] for k in labels])
    colors = {"left": "red", "center": "orange", "right": "blue"}
    plt.figure()
    plt.bar(labels, counts, color=[colors[k] for k in labels])
    plt.xlabel("Attractor")
    plt.ylabel("Frequency")
    plt.title("Outcome Distribution (Bar Chart)")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_heatmap(deviation_matrix, sigma_evol_values, attractor_keys, save_path=None):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(deviation_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, label="|Measured - Theoretical|")
    plt.xlabel("Attractor")
    plt.xticks(ticks=np.arange(len(attractor_keys)), labels=attractor_keys)
    plt.ylabel("σ_evol index")
    plt.yticks(ticks=np.arange(len(sigma_evol_values)), labels=[f"{s:.1f}" for s in sigma_evol_values])
    plt.title("Deviation from Born Rule (Heatmap)")
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_line_convergence(sigma_evol_values, measured_probs_dict, theoretical_probs, save_path=None):
    plt.figure(figsize=(8, 6))
    for key in measured_probs_dict:
        plt.plot(sigma_evol_values, measured_probs_dict[key], marker="o", label=f"Measured {key}")
        plt.hlines(theoretical_probs[key], xmin=sigma_evol_values[0], xmax=sigma_evol_values[-1],
                   linestyles="dashed", label=f"Theoretical {key}")
    plt.xlabel("σ_evol")
    plt.ylabel("Probability")
    plt.title("Convergence to Born Rule vs. σ_evol")
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
