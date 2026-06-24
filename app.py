import streamlit as st
import numpy as np
import plotly.graph_objects as go
from bearing_math import calculate_bearing_performance

st.set_page_config(page_title="Hydrodynamic Tribology Suite", layout="centered")
st.title("Hydrodynamic Journal Bearing Solver")
st.markdown("Automated Reynolds lubrication modeling and fluid film profile generation.")
st.markdown("---")

# Sidebar Configuration Layout
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

if st.sidebar.button("Execute Hydrodynamic Solver", use_container_width=True):
    metrics = calculate_bearing_performance(R, L, c, speed, load, visc)
    
    st.subheader("Fluid Film Diagnostics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Sommerfeld Number (S)", f"{metrics['Sommerfeld']:.3f}")
    col2.metric("Eccentricity Ratio (ε)", f"{metrics['Eccentricity']:.2f}")
    col3.metric("Min Film Thickness", f"{metrics['Min Film Thickness (um)']:.1f} μm")
    
    # 3D Pressure Profile Mockup Data Mapping Loop
    st.markdown("### 3D Hydrodynamic Pressure Field Matrix")
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(-L/2, L/2, 20)
    
    # Analytical Sommerfeld pressure curve configuration approximation
    eps = metrics['Eccentricity']
    p_theta = (6 * visc * (2*np.pi*speed/60) * (R/c)**2) * (eps * np.sin(theta) * (2 + eps*np.cos(theta))) / ((2 + eps**2) * (1 + eps*np.cos(theta))**2)
    p_theta = np.maximum(p_theta, 0) # Apply Reynolds boundary condition (No negative fluid tension)
    
    # Axial drop-off leakage function profile
    z_drop = 1 - (2 * z / L)**2
    
    # Core surface matrix mapping
    P_matrix = np.outer(p_theta, z_drop) / 1e6 # Convert to MPa
    THETA, Z = np.meshgrid(theta, z, indexing='ij')
    
    X_surf = R * np.cos(THETA)
    Y_surf = R * np.sin(THETA)
    
    fig = go.Figure(data=[go.Surface(x=X_surf*1000, y=Y_surf*1000, z=Z*1000, surfacecolor=P_matrix, colorscale='Viridis', showscale=True)])
    fig.update_layout(title="Fluid Pressure Over Bearing Surface Area (Color scale in MPa)", scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (Length mm)"), height=600)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Adjust operating limits in the left panel and click execute to map the fluid profile matrix.")
