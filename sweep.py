import numpy as np
import os
import matplotlib.pyplot as plt
from simulation import simulate_trials

def sweep_sigma_evol(sigma_evol_values, num_trials, initial_x0_range, config,
                     sigma_init, epsilon, attractors, use_geometry=True, save_folder=None):
    squared = {k: attractors[k]["weight"]**2 for k in attractors}
    total_sq = sum(squared.values())
    theoretical_probs = {k: squared[k] / total_sq for k in squared}
    attractor_keys = list(attractors.keys())
    
    deviation_matrix = np.zeros((len(sigma_evol_values), len(attractor_keys)))
    measured_probs_dict = {k: [] for k in attractor_keys}
    
    for i, sigma_evol in enumerate(sigma_evol_values):
        outcome_counts, outcomes, means = simulate_trials(num_trials, initial_x0_range,
                                                          config, sigma_init, epsilon,
                                                          sigma_evol, attractors, use_geometry=use_geometry)
        for j, key in enumerate(attractor_keys):
            measured_prob = outcome_counts[key] / num_trials
            deviation = abs(measured_prob - theoretical_probs[key])
            deviation_matrix[i, j] = deviation
            measured_probs_dict[key].append(measured_prob)
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(deviation_matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(im, label="|measured - theoretical|")
    plt.xlabel("Attractor")
    plt.xticks(ticks=np.arange(len(attractor_keys)), labels=attractor_keys)
    plt.ylabel("σigma_evol step")
    plt.yticks(ticks=np.arange(len(sigma_evol_values)), labels=[f"{s:.1f}" for s in sigma_evol_values])
    plt.title("Deviation from Born Probabilities")
    if save_folder:
        plt.savefig(os.path.join(save_folder, "sigma_evol_heatmap.png"), dpi=300)
    plt.show()
    
    plt.figure(figsize=(8, 6))
    for key in attractor_keys:
        plt.plot(sigma_evol_values, measured_probs_dict[key], marker="o", label=f"Measured {key}")
        plt.hlines(theoretical_probs[key], xmin=sigma_evol_values[0], xmax=sigma_evol_values[-1],
                   linestyles="dashed", label=f"Theoretical {key}")
    plt.xlabel("σigma_evol")
    plt.ylabel("Probability")
    plt.title("Convergence to Born Rule vs. σigma_evol")
    plt.legend()
    if save_folder:
        plt.savefig(os.path.join(save_folder, "sigma_evol_lineplot.png"), dpi=300)
    plt.show()
    
    return deviation_matrix, measured_probs_dict, theoretical_probs
