import os
import pandas as pd
import numpy as np
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cycle_dir = os.path.join(base_dir, "Driving Cycle")
    
    cycles = []
    if os.path.exists(cycle_dir):
        for f in os.listdir(cycle_dir):
            if f.endswith(".vmod"):
                cycles.append(os.path.join(cycle_dir, f))
    
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    
    loader = VectoLoader()
    # Just need engine map for fuel interp
    eng_interp = loader.read_vmap(vmap_path)
    
    for cycle_path in cycles:
        cycle_name = os.path.basename(cycle_path).replace(".vmod", "")
        df = loader.read_vmod(cycle_path)
        
        # 1. Distance
        # velocity in km/h, time in s
        # dist_m = sum(v/3.6 * dt)
        if 'velocity_kmh' in df.columns:
            v_ms = df['velocity_kmh'] / 3.6
        else:
            # Fallback
            v_ms = df['v_act_kmh'] / 3.6 # typical vecto col
            
        # Time steps
        times = df['time'].values
        dts = np.diff(times, prepend=times[0])
        dts[0] = dts[1] if len(dts) > 1 else 1.0
        
        dist_m = np.sum(v_ms * dts)
        dist_km = dist_m / 1000.0
        
        # 2. Baseline Fuel (Pure ICE)
        # Using T_ice_fcmap [Nm] and rpm_ice
        # This assumes the reference load is the pure ICE load.
        t_ice = df['T_ice_fcmap [Nm]'].values
        rpm_ice = df['rpm_ice'].values
        
        # Vectorized lookup
        pts = np.column_stack((rpm_ice, t_ice))
        fuel_rate = eng_interp(pts) # g/s
        
        # Remove NaNs
        fuel_rate = np.nan_to_num(fuel_rate)
        
        # Idle check (rpm > 500)
        # Simplified: VECTO reference usually accounts for everything.
        
        total_fuel_g = np.sum(fuel_rate * dts)
        total_fuel_kg = total_fuel_g / 1000.0
        
        print(f"CYCLE: {cycle_name}")
        print(f"  Dist: {dist_km:.4f} km")
        print(f"  Base_Fuel: {total_fuel_kg:.4f} kg")

if __name__ == "__main__":
    main()
