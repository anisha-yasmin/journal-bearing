import numpy as np

def solve_reynolds_1d(R, L, c, speed_rpm, viscosity_pas, epsilon, mesh_pts=180):
    """
    Solves the 1D Reynolds equation around the bearing circumference
    using finite differences, enforcing the Reynolds cavitation boundary condition
    """
    omega = 2 * np.pi * (speed_rpm / 60.0)
    theta = np.linspace(0, 2 * np.pi, mesh_pts)
    dtheta = theta - theta
    
    # Film thickness profile: h = c * (1 + epsilon * cos(theta))
    h = c * (1.0 + epsilon * np.cos(theta))
    
    # FORCED 2D MATRIX CREATION: This syntax guarantees a 2D grid structure
    A = np.zeros(shape=(mesh_pts, mesh_pts), dtype=np.float64)
    B = np.zeros(shape=(mesh_pts,), dtype=np.float64)
    
    # Central difference coefficients
    for i in range(1, mesh_pts - 1):
        h_mid_plus = (h[i] + h[i+1]) / 2.0
        h_mid_minus = (h[i] + h[i-1]) / 2.0
        
        # Finite difference terms for left, center, and right nodes
        A[i, i-1] = (h_mid_minus**3) / (dtheta**2)
        A[i, i+1] = (h_mid_plus**3) / (dtheta**2)
        A[i, i] = -(h_mid_minus**3 + h_mid_plus**3) / (dtheta**2)
        
        # Right hand side forcing function (6 * mu * omega * R^2 * dh/dtheta)
        dh_dtheta = -c * epsilon * np.sin(theta[i])
        B[i] = 6 * viscosity_pas * omega * (R**2) * dh_dtheta

    # Enforce boundary conditions at the edges
    A = 1.0; B = 0.0
    A[-1, -1] = 1.0; B[-1] = 0.0
    
    # Solve initial pressure distribution
    P = np.linalg.solve(A, B)
    
    # Enforce Reynolds Cavitation condition (Swift-Stieber: P >= 0)
    for _ in range(10):
        for i in range(1, mesh_pts - 1):
            if P[i] < 0:
                P[i] = 0.0
                
    return theta, P

def find_equilibrium_eccentricity(R, L, c, speed_rpm, load_n, viscosity_pas):
    """
    Iterates through eccentricity ratios to find where the integrated fluid 
    pressure vector matches the external bearing load.
    """
    best_eps = 0.5
    min_force_error = float('inf')
    
    # Step through possible eccentricity ranges (0.05 to 0.95)
    for eps in np.linspace(0.05, 0.95, 50):
        theta, P = solve_reynolds_1d(R, L, c, speed_rpm, viscosity_pas, eps)
        
        # Integrate pressure forces along vertical and horizontal components
        axial_integration_factor = (2.0 / 3.0) * L
        
        fx = np.trapz(P * np.cos(theta), theta) * R * axial_integration_factor
        fy = np.trapz(P * np.sin(theta), theta) * R * axial_integration_factor
        total_fluid_force = np.sqrt(fx**2 + fy**2)
        
        error = abs(total_fluid_force - load_n)
        if error < min_force_error:
            min_force_error = error
            best_eps = eps
            
    # Regenerate the final matching pressure field profile
    theta, P_final = solve_reynolds_1d(R, L, c, speed_rpm, viscosity_pas, best_eps)
    h_min = c * (1.0 - best_eps)
    
    return best_eps, theta, P_final, h_min
