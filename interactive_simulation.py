import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

st.set_page_config(page_title="IM-VRM Routing Simulation", layout="wide")

st.title("Interactive Geographic Route Simulation")
st.markdown("This dashboard calculates logistics metrics dynamically using the **exact U-shaped speed efficiency curves and payload mathematics** from `src/features/metrics.py`.")

# 1. Load Data
@st.cache_data
def load_data():
    file_path = os.path.join("outputs", "deep_aco_comparison.csv")
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    return df

df_original = load_data()

if df_original is None:
    st.error("Error: Could not find 'outputs/deep_aco_comparison.csv'. Please ensure the baseline pipeline has been executed.")
    st.stop()

# 2. Sidebar Parameters
st.sidebar.header("IM-VRM Parameters")
st.sidebar.markdown("*(Sourced from `config/default_config.yaml`)*")

vehicle_type = st.sidebar.radio(
    "Vehicle Configuration", 
    ["Light Duty (Taxi/Car)", "Heavy Duty (Truck)"],
    help="Changes the base fuel consumption rate and aerodynamic scaling."
)

if "Light Duty" in vehicle_type:
    BASE_FUEL_RATE_L_100KM = 7.0
    LOAD_PENALTY_FACTOR = 0.05
else:
    BASE_FUEL_RATE_L_100KM = 25.0
    LOAD_PENALTY_FACTOR = 0.08

payload_kg = st.sidebar.slider(
    "Payload (kg)", 
    min_value=0.0, 
    max_value=5000.0 if "Heavy" in vehicle_type else 2000.0, 
    value=0.0, 
    step=100.0,
    help="Extra vehicle payload. Plugs into formula: adjusted_rate = base_rate * (1 + penalty_factor * payload/100)"
)

speed_mult = st.sidebar.slider(
    "Target Speed Multiplier", 
    min_value=0.5, 
    max_value=3.0, 
    value=1.0, 
    step=0.1,
    help="Multiplies baseline average route speed. Determines the fuel efficiency based on the U-shaped curve."
)

congestion = st.sidebar.slider(
    "Congestion Level", 
    min_value=0.0, 
    max_value=2.0, 
    value=0.0, 
    step=0.5,
    help="Drops the average speed. Non-predictive algorithms get pushed into crawl speeds (<10km/h), heavily penalizing fuel efficiency."
)

# 3. Exact Mathematics from `metrics.py`
CO2_G_PER_LITER = 2600.0 if "Heavy" in vehicle_type else 2300.0

def get_efficiency_factor(v):
    """
    U-shaped efficiency factor based on speed (v) from metrics.py:
    < 10.0: 2.0 (Crawl)
    10-30: 1.4 (Heavy traffic)
    30-60: 1.0 (Optimum range)
    60-90: 1.25 (Sub-optimum)
    >= 90: 1.6 (Drag dominated)
    """
    eff = np.ones_like(v)
    eff[v < 10.0] = 2.0
    eff[(v >= 10.0) & (v < 30.0)] = 1.4
    eff[(v >= 30.0) & (v < 60.0)] = 1.0
    eff[(v >= 60.0) & (v < 90.0)] = 1.25
    eff[v >= 90.0] = 1.6
    return eff

def compute_metrics(dist_km, time_sec, original_fuel, original_co2, algo_type="predictive"):
    # Calculate baseline average speed in km/h
    time_sec = np.maximum(time_sec, 1.0)
    v_base = dist_km / (time_sec / 3600.0)
    
    # Apply congestion penalty to speed
    if algo_type == "non-predictive":
        v_new = v_base * speed_mult / (1.0 + congestion * 0.9)
    else:
        v_new = v_base * speed_mult / (1.0 + congestion * 0.2)
        
    v_new = np.maximum(v_new, 1.0)
    
    # New Time
    new_time_sec = (dist_km / v_new) * 3600.0
    
    # Get efficiency factors to compute a relative ratio
    eff_base = get_efficiency_factor(v_base)
    eff_new = get_efficiency_factor(v_new)
    
    # Relative scaling based on efficiency changes
    eff_ratio = eff_new / eff_base
    
    # Payload scaling
    payload_ratio = (1.0 + LOAD_PENALTY_FACTOR * (payload_kg / 100.0))
    
    # Base vehicle scaling ratio (if switching from default light to heavy)
    vehicle_ratio = BASE_FUEL_RATE_L_100KM / 7.0 
    
    # Calculate new Fuel and CO2 strictly relative to the true CSV baseline
    fuel_l = original_fuel * eff_ratio * payload_ratio * vehicle_ratio
    co2_g = original_co2 * eff_ratio * payload_ratio * vehicle_ratio * (CO2_G_PER_LITER / 2300.0)
    
    return new_time_sec, fuel_l, co2_g, v_new

# 5. Apply Scaling
df = df_original.copy()

# Greedy
g_time, g_fuel, g_co2, g_v = compute_metrics(df['greedy_dist'], df['greedy_time'], df['greedy_fuel'], df['greedy_co2'], "non-predictive")

# Classical ACO
a_time, a_fuel, a_co2, a_v = compute_metrics(df['aco_dist'], df['aco_time'], df['aco_fuel'], df['aco_co2'], "non-predictive")

# DeepACO
d_time, d_fuel, d_co2, d_v = compute_metrics(df['deep_aco_dist'], df['deep_aco_time'], df['deep_aco_fuel'], df['deep_aco_co2'], "predictive")


# 6. Calculate Averages
avg_greedy_time = g_time.mean() / 60 # minutes
avg_aco_time = a_time.mean() / 60
avg_deep_aco_time = d_time.mean() / 60

avg_greedy_co2 = g_co2.mean()
avg_aco_co2 = a_co2.mean()
avg_deep_aco_co2 = d_co2.mean()

avg_greedy_fuel = g_fuel.mean()
avg_aco_fuel = a_fuel.mean()
avg_deep_aco_fuel = d_fuel.mean()

avg_greedy_speed = g_v.mean()
avg_aco_speed = a_v.mean()
avg_deep_aco_speed = d_v.mean()

# 7. Display Metrics
st.subheader("Strict Mathematical Output Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="DeepACO Avg Route Time (mins)", 
        value=f"{avg_deep_aco_time:.1f}", 
        delta=f"{(avg_deep_aco_time - avg_greedy_time):.1f} vs Greedy", 
        delta_color="inverse"
    )
with col2:
    st.metric(
        label="DeepACO Avg CO2 Emissions (g)", 
        value=f"{avg_deep_aco_co2:,.0f}", 
        delta=f"{((avg_deep_aco_co2 - avg_greedy_co2)/avg_greedy_co2*100):.1f}% vs Greedy", 
        delta_color="inverse"
    )
with col3:
    st.metric(
        label="DeepACO Avg Moving Speed (km/h)", 
        value=f"{avg_deep_aco_speed:.1f}", 
        delta=f"{(avg_deep_aco_speed - avg_greedy_speed):.1f} vs Greedy", 
        delta_color="inverse"
    )

st.divider()

# 8. Interactive Visualization Charts
st.subheader("Interactive Algorithm Comparison")

chart_data = pd.DataFrame({
    'Algorithm': ['Greedy', 'Classical ACO', 'DeepACO (IM-VRM)'],
    'Time (mins)': [avg_greedy_time, avg_aco_time, avg_deep_aco_time],
    'CO2 (g)': [avg_greedy_co2, avg_aco_co2, avg_deep_aco_co2],
    'Fuel (L)': [avg_greedy_fuel, avg_aco_fuel, avg_deep_aco_fuel],
    'Speed (km/h)': [avg_greedy_speed, avg_aco_speed, avg_deep_aco_speed]
})

col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown("**Travel Duration Comparison (Lower is Better)**")
    st.bar_chart(data=chart_data, x="Algorithm", y="Time (mins)", color="Algorithm")

with col_chart2:
    st.markdown("**Carbon Footprint (Lower is Better)**")
    st.bar_chart(data=chart_data, x="Algorithm", y="CO2 (g)", color="Algorithm")

col_chart3, col_chart4 = st.columns(2)

with col_chart3:
    st.markdown("**Fuel Consumption (Lower is Better)**")
    st.bar_chart(data=chart_data, x="Algorithm", y="Fuel (L)", color="Algorithm")

with col_chart4:
    st.markdown("**Average Moving Speed**")
    st.bar_chart(data=chart_data, x="Algorithm", y="Speed (km/h)", color="Algorithm")

st.info("💡 **Insight:** Hover over the bars to see exact data. You can pan and zoom into the charts, and click the three dots (`...`) in the top right to download them as images!")

st.divider()
st.subheader("Raw Mathematical Output (Interactive Table)")
st.markdown("You can click on columns to sort, or download the output directly as a CSV.")

# Format the dataframe for display
display_df = chart_data.copy()
display_df['Time (mins)'] = display_df['Time (mins)'].round(1)
display_df['CO2 (g)'] = display_df['CO2 (g)'].round(0).astype(int)
st.dataframe(display_df, hide_index=True)
