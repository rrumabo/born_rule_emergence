import numpy as np

def normalize_wavefunction(psi, x):
    norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi / norm

def gaussian_state(x, center, sigma):
    psi = np.exp(-((x - center)**2) / (2 * sigma**2))
    return psi

def add_noise(psi, epsilon):
    noise = (np.random.normal(0, epsilon, size=psi.shape) +
             1j * np.random.normal(0, epsilon, size=psi.shape))
    return psi + noise

def compute_mean(psi, x):
    return np.trapz(np.abs(psi)**2 * x, x)

def evolve_state(psi, x, attractors, sigma_evol, use_geometry=True):
    x_mean = compute_mean(psi, x)
    scores = {}
    for key, params in attractors.items():
        center = params["center"]
        weight = params["weight"]
        if use_geometry:
            score = (weight ** 2) * np.exp(- ((x_mean - center) ** 2) / (2 * sigma_evol ** 2))
        else:
            score = weight ** 2
        scores[key] = score
    total = sum(scores.values())
    probabilities = {key: score / total for key, score in scores.items()}
    keys = list(probabilities.keys())
    probs = np.array([probabilities[k] for k in keys])
    chosen = np.random.choice(keys, p=probs)
    chosen_params = attractors[chosen]
    new_center = chosen_params["center"]
    phase = chosen_params["phase"]
    sigma_final = 1.0
    psi_final = np.exp(-((x - new_center)**2) / (2 * sigma_final**2)) * np.exp(1j * phase)
    psi_final = normalize_wavefunction(psi_final, x)
    return chosen, probabilities, x_mean, psi_final

def simulate_trial(initial_x0, x, sigma_init, epsilon, attractors, sigma_evol, use_geometry=True):
    psi = gaussian_state(x, initial_x0, sigma_init)
    psi = normalize_wavefunction(psi, x)
    psi_noisy = add_noise(psi, epsilon)
    psi_noisy = normalize_wavefunction(psi_noisy, x)
    return evolve_state(psi_noisy, x, attractors, sigma_evol, use_geometry=use_geometry)

def simulate_trials(num_trials, initial_x0_range, config, sigma_init, epsilon, sigma_evol, attractors, use_geometry=True):
    L = config["L"]
    N = config["N"]
    x = np.linspace(-L/2, L/2, N)
    outcome_counts = {key: 0 for key in attractors.keys()}
    outcomes = []
    means = []
    for _ in range(num_trials):
        initial_x0 = np.random.uniform(initial_x0_range[0], initial_x0_range[1])
        chosen, probs, x_mean, psi_final = simulate_trial(initial_x0, x, sigma_init, epsilon, attractors, sigma_evol, use_geometry)
        outcome_counts[chosen] += 1
        outcomes.append(chosen)
        means.append(x_mean)
    return outcome_counts, outcomes, means
