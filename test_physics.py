import pandas as pd
import numpy as np
from vecto_loader import VectoLoader

# 1. Load Data
loader = VectoLoader()
vmap_path = "Engine/325kW.vmap"
vmod_path = "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod"
vmod_path_abs = "/root/ECMS_Python/" + vmod_path
vmap_path_abs = "/root/ECMS_Python/" + vmap_path

print("Loading VMAP...")
fuel_interp = loader.read_vmap(vmap_path_abs)

print("Checking Specific Points (VMAP verification)...")
# 500 rpm, 0 Nm -> 1355 g/h = 0.3764 g/s
val = fuel_interp([[500, 0]])[0] * 3600
print(f"500 RPM, 0 Nm: Map=1355 g/h, Interp={val:.2f} g/h")

# 500 rpm, 213.4 Nm -> 3412.291 g/h
val2 = fuel_interp([[500, 213.4]])[0] * 3600
print(f"500 RPM, 213.4 Nm: Map=3412.29 g/h, Interp={val2:.2f} g/h")


print("\nLoading VMOD...")
df = loader.read_vmod(vmod_path_abs)
if df is None:
    # Use raw pandas if loader fails or modifies too much
    df = pd.read_csv(vmod_path_abs, skiprows=2, encoding='latin1')
    # Rename basic cols for ease
    df = df.rename(columns={
        'time [s]': 'time',
        'n_ice_avg [1/min]': 'rpm', 
        'P_wheel_in [kW]': 'p_wheel',
        'P_aux_mech [kW]': 'p_aux',
        'FC-Map [g/h]': 'fc_vecto',
        'T_ice_fcmap [Nm]': 't_ice_vecto'
    })

print("Columns:", df.columns.tolist())

# 2. Physics Check: Reconstruct Engine Load
# My Code Logic: P_req = P_wheel / eta (ignoring aux?)
eta = 0.96
p_wheel = df['power_wheel_kw'].values
# Check for aux column
aux_col = 'P_aux_mech [kW]'
if aux_col in df.columns:
    p_aux = df[aux_col].values
else:
    p_aux = np.zeros_like(p_wheel)

# Vecto Torque/Power Check
rpm = df['rpm_ice'].values # 1/min
omega = rpm * 2 * np.pi / 60.0

# Calculate My T_req (As currently in p2_hybrid)
p_trans_in_my = np.where(p_wheel >= 0, p_wheel / eta, p_wheel * eta)
# Ignoring Aux in calculation as suspected
t_req_my = np.zeros_like(p_trans_in_my)
mask = omega > 1.0
t_req_my[mask] = (p_trans_in_my[mask] * 1000.0) / omega[mask]

# Calculate T_req WITH Aux
p_req_total = p_trans_in_my + p_aux
t_req_total = np.zeros_like(p_req_total)
t_req_total[mask] = (p_req_total[mask] * 1000.0) / omega[mask]

# Compare with VECTO T_ice
t_vecto_col = 'T_ice_fcmap [Nm]'
if t_vecto_col in df.columns:
    t_vecto = df[t_vecto_col].values
else:
    t_vecto = np.zeros_like(rpm)

print("\n--- Load Reconstruction Analysis ---")
print(f"Mean VECTO Torque: {np.mean(t_vecto):.2f} Nm")
print(f"Mean MY Torque (No Aux): {np.mean(t_req_my):.2f} Nm")
print(f"Mean MY Torque (With Aux): {np.mean(t_req_total):.2f} Nm")
print(f"Difference (VECTO - MyNoAux): {np.mean(t_vecto - t_req_my):.2f} Nm")

# 3. Fuel Calculation Check (Time Step Integration)
# Use VECTO Torque/RPM to lookup fuel in MY map and compare.
points = np.column_stack((rpm, t_vecto))
my_fuel_gs = fuel_interp(points)
my_fuel_gh = my_fuel_gs * 3600.0

fc_col = 'FC-Map [g/h]'
if fc_col in df.columns:
    vecto_fuel_gh = df[fc_col].values
else:
    vecto_fuel_gh = np.zeros_like(rpm)

print("\n--- Fuel Calculation Analysis ---")
print(f"Mean VECTO Fuel: {np.mean(vecto_fuel_gh):.2f} g/h")
print(f"Mean MY Fuel (using VECTO Torque): {np.mean(my_fuel_gh):.2f} g/h")

total_vecto_kg = np.sum(vecto_fuel_gh * df['dt [s]'].values / 3600.0) / 1000.0
total_my_kg = np.sum(my_fuel_gh * df['dt [s]'].values / 3600.0) / 1000.0

print(f"Total VECTO Fuel: {total_vecto_kg:.2f} kg")
print(f"Total MY Fuel (Re-calc): {total_my_kg:.2f} kg")

if abs(total_vecto_kg - total_my_kg) > 1.0:
    print("WARNING: Fuel Map interpolation discrepancy!")
else:
    print("Fuel Map Interpolation is accurate.")

# 4. Check My Simulation Load Fuel
# Fuel using MY Calculated Torque (Current Bug?)
points_my = np.column_stack((rpm, t_req_my))
fuel_my_load_gs = fuel_interp(points_my)
total_my_load_kg = np.sum(fuel_my_load_gs * df['dt [s]'].values) / 1000.0
print(f"Fuel using CURRENT SIMULATION Load (No Aux): {total_my_load_kg:.2f} kg")

points_total = np.column_stack((rpm, t_req_total))
fuel_total_load_gs = fuel_interp(points_total)
total_total_load_kg = np.sum(fuel_total_load_gs * df['dt [s]'].values) / 1000.0
print(f"Fuel using CORRECTED Load (With Aux): {total_total_load_kg:.2f} kg")
