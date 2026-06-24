import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Hydrodynamic Tribology Suite", layout="centered")
st.title("Hydrodynamic Journal Bearing Solver")
st.markdown("Automated Reynolds lubrication modeling and fluid film profile generation.")
st.markdown("---")

# --- CORE PHYSICS SOLVER FUNCTIONS ---
def solve_reynolds_1d(R, L, c, speed_rpm, viscosity_pas, epsilon, mesh_pts=180):
    omega = 2 * np.pi * (speed_rpm / 60.0)
    theta = np.linspace(0, 2 * np.pi, mesh_pts)
    dtheta = theta - theta
    
    h = c * (1.0 + epsilon * np.cos(theta))
    
    # Initialize a standard Python list of lists for rows
    A_list = [[0.0] * mesh_pts for _ in range(mesh_pts)]
    B_list = [0.0] * mesh_pts
    
    # Boundary condition at node 0
    A_list = 1.0
    B_list = 0.0
    
    # Fill internal finite difference steps
    for i in range(1, mesh_pts - 1):
        h_mid_plus = (h[i] + h[i+1]) / 2.0
        h_mid_minus = (h[i] + h[i-1]) / 2.0
        
        A_list[i][i-1] = (h_mid_minus**3) / (dtheta**2)
        A_list[i][i+1] = (h_mid_plus**3) / (dtheta**2)
        A_list[i][i] = -(h_mid_minus**3 + h_mid_plus**3) / (dtheta**2)
        
        dh_dtheta = -c * epsilon * np.sin(theta[i])
        B_list[i] = 6 * viscosity_pas * omega * (R**2) * dh_dtheta

    # Boundary condition at the final node
    A_list[-1][-1] = 1.0
    B_list[-1] = 0.0
    
    # Convert to NumPy arrays only at the moment of solving
    A = np.array(A_list, dtype=np.float64)
    B = np.array(B_list, dtype=np.float64)
    
    P = np.linalg.solve(A, B)
    
    # Enforce Reynolds Cavitation condition
    for _ in range(10):
        for i in range(1, mesh_pts - 1):
            if P[i] < 0:
                P[i] = 0.0
                
    return theta, P

def find_equilibrium_eccentricity(R, L, c, speed_rpm, load_n, viscosity_pas):
    best_eps = 0.5
    min_force_error = float('inf')
    
    for eps in np.linspace(0.05, 0.95, 50):
        theta, P = solve_reynolds_1d(R, L, c, speed_rpm, viscosity_pas, eps)
        axial_integration_factor = (2.0 / 3.0) * L
        
        fx = np.trapz(P * np.cos(theta), theta) * R * axial_integration_factor
        fy = np.trapz(P * np.sin(theta), theta) * R * axial_integration_factor
        total_fluid_force = np.sqrt(fx**2 + fy**2)
        
        error = abs(total_fluid_force - load_n)
        if error < min_force_error:
            min_force_error = error
            best_eps = eps
            
    theta, P_final = solve_reynolds_1d(R, L, c, speed_rpm, viscosity_pas, best_eps)
    h_min = c * (1.0 - best_eps)
    
    return best_eps, theta, P_final, h_min

# --- SIDEBAR INPUT CONFIGURATION ---
st.sidebar.header("Mechanical Bounds")
R = st.sidebar.slider("Journal Radius (mm)", 10.0, 100.0, 40.0, 1.0) / 1000.0
L = st.sidebar.slider("Bearing Length (mm)", 10.0, 150.0, 50.0, 1.0) / 1000.0
c = st.sidebar.slider("Radial Clearance (microns)", 10, 200, 40, 5) / 1e6

st.sidebar.header("Operational Loading")
speed = st.sidebar.slider("Shaft Speed (RPM)", 500, 10000, 3000, 100)
load = st.sidebar.slider("Radial Load (N)", 500, 50000, 5000, 500)
visc = st.sidebar.selectbox("Lubricant Oil Choice", 
                            options=[0.015, 0.046, 0.068], 
                            format_func=lambda x: f"ISO VG {int(x*1000)} Dynamic Viscosity")

# --- SOLVER EXECUTION BUTTON ---
if st.sidebar.button("Execute Hydrodynamic Solver", use_container_width=True):
    eps, theta, P_profile, h_min = find_equilibrium_eccentricity(R, L, c, speed, load, visc)
    
    st.subheader("Fluid Film Diagnostics")
    col1, col2, col3 = st.columns(3)
    
    P_projected = load / (2 * R * L)
    S = ((R / c) ** 2) * (visc * (speed / 60.0)) / P_projected
    
    col1.metric("Sommerfeld Number (S)", f"{S:.3f}")
    col2.metric("Eccentricity Ratio (epsilon)", f"{eps:.2f}")
    col3.metric("Min Film Thickness", f"{h_min * 1e6:.1f} microns")
    
    if (h_min * 1e6) < 2.0:
        st.error("Operational Failure Risk: Oil film thickness dropped below 2.0 microns. High risk of scoring!")
    else:
        st.success("Stable Hydrodynamic Film: Fluid wedge successfully supports the radial shaft load.")

    # --- PLOT 1: 3D HYDRODYNAMIC PRESSURE FIELD ---
    st.markdown("### 3D Hydrodynamic Pressure Field Matrix")
    
    z = np.linspace(-L/2, L/2, 30)
    z_drop = 1.0 - (2.0 * z / L)**2
    
    P_matrix = np.outer(P_profile, z_drop) / 1e6
    THETA, Z = np.meshgrid(theta, z, indexing='ij')
    
    X_surf = R * np.cos(THETA)
    Y_surf = R * np.sin(THETA)
    
    fig_3d = go.Figure(data=[go.Surface(
        x=X_surf * 1000, y=Y_surf * 1000, z=Z * 1000,
        surfacecolor=P_matrix, colorscale='Viridis'
    )])
    fig_3d.update_layout(template="plotly_white", height=600, scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)", aspectmode='data'))
    st.plotly_chart(fig_3d, use_container_width=True)

    # --- PLOT 2: 2D CLEARANCE CROSS-SECTION ---
    st.markdown("### Shaft Eccentric Displacement & Wedge Clearance")
    
    angles = np.linspace(0, 2 * np.pi, 200)
    phi = np.pi / 4.0
    e_displacement = eps * (c * 1e6) 
    x_shaft_center = e_displacement * np.sin(phi)
    y_shaft_center = -e_displacement * np.cos(phi)
    
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(x=(c * 1e6) * np.cos(angles), y=(c * 1e6) * np.sin(angles), mode='lines', name='Bearing Sleeve Wall', line=dict(color='black', width=2)))
    fig_2d.add_trace(go.Scatter(x=x_shaft_center + (c * 1e6) * np.cos(angles), y=y_shaft_center + (c * 1e6) * np.sin(angles), mode='lines', fill='toself', name='Shifted Motor Shaft', line=dict(color='#1E88E5', width=2)))
    fig_2d.add_trace(go.Scatter(x=[0.0], y=[0.0], mode='markers', name='Sleeve Center', marker=dict(size=8, color='red')))
    fig_2d.add_trace(go.Scatter(x=[x_shaft_center], y=[y_shaft_center], mode='markers', name='Shaft Center', marker=dict(size=8, color='blue')))
    
    fig_2d.update_layout(template="plotly_white", height=500, width=500, xaxis=dict(title="Horizontal Clearance (microns)", scaleanchor="y", scaleratio=1), yaxis=dict(title="Vertical Clearance (microns)"))
    st.plotly_chart(fig_2d, use_container_width=True)
else:
    st.info("Adjust operating limits in the left panel and click execute to run the finite difference model loops.")
