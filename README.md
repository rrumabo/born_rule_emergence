# Born Rule Emergence via Geometric Attractor Dynamics

This simulation investigates whether Born rule probabilities (∝ |a|²) can emerge as a statistical effect from a model involving Gaussian wavefunctions, attractor selection, and geometric heuristics.

## Features

- Configurable attractor weights and phases
- Noisy evolution simulating decoherence
- Heuristic evolution based on ⟨x⟩ and σ_evol
- Sweep experiments over σ_evol values
- Heatmaps and line plots for convergence visualization

## Structure

- `main.py`: Runs basic simulation with fixed parameters
- `evolution.py`: Contains wavefunction evolution logic
- `analysis.py`: Includes sweeping and statistical tools
- `plotting.py`: Handles bar, pie, and heatmap visualizations

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Output

Results (images and logs) are saved to a local folder on your desktop (e.g. `born_sim_results`).

## Goal

To test whether Born rule probabilities can emerge from a geometric + noise-based model, providing insights into quantum measurement interpretation beyond axiomatic assumptions.
