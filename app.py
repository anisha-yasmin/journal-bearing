import streamlit as st
import numpy as np
import plotly.graph_objects as go
from bearing_math import find_equilibrium_eccentricity

st.set_page_config(page_title="Hydrodynamic Tribology", layout="centered")
st.title("Hydrodynamic Journal Bearing Solver")
st.markdown("Automated Reynolds lubrication modeling and fluid film profile generation.")
st.markdown("---")

# --- SIDEBAR INPUT CONFIGURATION ---
st.sidebar.header("Mechanical Bounds")
R = st.sidebar.slider("Journal Radius (mm)", 10.0, 100.0, 40.0, 1.0) / 1000.0
L = st.sidebar.slider("Bearing Length (mm)", 10.0, 150.0, 50.0, 1.0) / 1000.0
c = st.sidebar.slider("Radial Clearance (μm)", 10, 200, 40, 5) / 1e6

st.sidebar.header("Operational Loading")
speed = st.sidebar.slider("Shaft Speed (RPM)", 500, 10000, 3000, 100)
load = st.sidebar.slider("Radial Load (N)", 500, 50000, 5000, 500)
visc = st.sidebar.selectbox("Lubricant Oil Choice", 
                            options=[0.015, 0.046, 0.068], 
                            format_func=lambda x: f"ISO VG {int(x*1000)} Dynamic Viscosity")

# --- SOLVER EXECUTION (Safely wrapped inside a button state condition) ---
run_solver = st.sidebar.button("Execute Hydrodynamic Solver", use_container_width=True)

if run_solver:
    # This calculation ONLY runs when the button is clicked!
    eps, theta, P_profile, h_min = find_equilibrium_eccentricity(R, L, c, speed, load, visc)
    
    st.subheader("Fluid Film Diagnostics")
    col1, col2, col3 = st.columns(3)
    
    P_projected = load / (2 * R * L)
    S = ((R / c) ** 2) * (visc * (speed / 60.0)) / P_projected
    
    col1.metric("Sommerfeld Number (S)", f"{S:.3f}")
    col2.metric("Eccentricity Ratio (ε)", f"{eps:.2f}")
    col3.metric("Min Film Thickness", f"{h_min * 1e6:.1f} μm")
    
    if (h_min * 1e6) < 2.0:
        st.error("Operational Failure Risk: Oil film thickness dropped below 2.0 μm. High risk of scoring!")
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
    
    fig_2d.update_layout(template="plotly_white", height=500, width=500, xaxis=dict(title="Horizontal Clearance (μm)", scaleanchor="y", scaleratio=1), yaxis=dict(title="Vertical Clearance (μm)"))
    st.plotly_chart(fig_2d, use_container_width=True)
else:
    st.info("Adjust operating limits in the left panel and click execute to run the finite difference model loops.")
