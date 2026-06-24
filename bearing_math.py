import numpy as np

def calculate_bearing_performance(R, L, c, speed_rpm, load_n, viscosity_pas):
    """
    Computes fluid film geometry and dimensionless tribology metrics.
    """
    omega = 2 * np.pi * (speed_rpm / 60.0)  # rad/s
    P = load_n / (2 * R * L)                 # Projected pressure (Pa)
    
    # 1. Calculate Sommerfeld Number
    S = ((R / c) ** 2) * (viscosity_pas * (speed_rpm / 60.0)) / P
    
    # 2. Approximate Eccentricity Ratio (epsilon) using a standard analytical curve fit
    # In a full numerical app, we can use an iterative solver for Swift-Stieber boundaries
    epsilon = 1.0 - np.exp(-2.5 * S) if S > 0 else 0.99
    epsilon = min(max(epsilon, 0.01), 0.95)
    
    # Minimum film thickness
    h_min = c * (1.0 - epsilon)
    
    return {
        "Sommerfeld": S,
        "Eccentricity": epsilon,
        "Min Film Thickness (um)": h_min * 1e6,
        "Projected Pressure (MPa)": P / 1e6
    }
