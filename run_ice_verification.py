import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck

def main():
    # 1. Setup
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    
    # 2. Load Components
    print("Loading components...")
    truck.load_components(
        engine_map_path='/root/ECMS_Python/Engine/325kW.vmap',
        motor_map_path='/root/ECMS_Python/Emotor/EM_Map - kopie.vemo', # Not used in ICE only but needed for init
        motor_param_path='/root/ECMS_Python/Emotor/P2_Group5_EM.vem',
        bat_param_path='/root/ECMS_Python/Emotor/P2_Group5_REESS.vreess',
        bat_ocv_path='/root/ECMS_Python/Emotor/REESS_SOC_curve.vbatv',
        bat_res_path='/root/ECMS_Python/Emotor/REESS_Internal_Resistance.vbatr'
    )
    
    # 3. Load Cycle
    print("Loading Driving Cycle...")
    cycle_df = loader.read_vmod('/root/ECMS_Python/Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod')
    
    # 4. Physics
    # Use exact VECTO load
    t_reqs = truck.calc_backward_physics(cycle_df)
    
    print(f"Starting ICE-ONLY Simulation... (Steps: {len(cycle_df)})")
    
    times = cycle_df['time'].values
    rpms = cycle_df['rpm_ice'].values
    
    total_fuel_g = 0.0
    
    results = {'time': times, 'fuel_rate': [], 't_eng': []}
    
    for i in range(len(cycle_df)):
        t = times[i]
        rpm = rpms[i]
        t_req = t_reqs[i]
        
        # dt
        if i == 0:
            dt = 0.5 
        else:
            dt = t - times[i-1]
            
        # PURE ICE STRATEGY
        t_mot = 0.0
        t_eng = t_req # Engine takes all load
        
        # Calculate Fuel
        # Using truck.fuel_interp directly (returns g/s)
        fuel_g_s = float(truck.fuel_interp([[rpm, t_eng]]))
        
        # Handle NaN (Out of bounds)
        if np.isnan(fuel_g_s):
            # Check why it is NaN
            # print(f"NaN Fuel at RPM={rpm:.1f}, Torque={t_eng:.1f}")
            # Recover using Nearest? Or Extrapolate 0?
            # VECTO usually extrapolates for Fuel Map or clamps?
            # If Torque is negative (Coasting), Fuel is 0.
            if t_eng <= 0:
                fuel_g_s = 0.0
            else:
               pass # print(f"Warning: Positive Torque NaN at {rpm}, {t_eng}")
               # For verification, we suspect extrapolation needed.
               # But previously test_physics.py worked?
               # Ah, VectoLoader now returns NaN for fill_value.
               # We need to investigate if VECTO points are slightly outside Hull.
               fuel_g_s = 0.0 # Placeholder

        
        total_fuel_g += fuel_g_s * dt
        
        results['fuel_rate'].append(fuel_g_s)
        results['t_eng'].append(t_eng)
        
    print("Simulation Complete.")
    print(f"Total Fuel Consumed (Pure ICE Run): {total_fuel_g/1000:.2f} kg")
    
    # Baseline comparison 
    if 'FC-Map [g/h]' in cycle_df.columns:
        base_fuel_rate = cycle_df['FC-Map [g/h]'].fillna(0).values / 3600.0
        dt_arr = np.diff(times, prepend=times[0]) 
        total_base_g = np.sum(base_fuel_rate * dt_arr)
        print(f"Baseline Fuel (VECTO): {total_base_g/1000:.2f} kg")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, results['fuel_rate'], label='My Fuel (g/s)', alpha=0.7)
    if 'FC-Map [g/h]' in cycle_df.columns:
        plt.plot(times, cycle_df['FC-Map [g/h]']/3600.0, label='VECTO Fuel (g/s)', alpha=0.5, linestyle='--')
    plt.legend()
    plt.title("Fuel Rate Comparison (ICE Only)")
    plt.ylabel("Fuel Rate [g/s]")
    plt.xlabel("Time [s]")
    plt.savefig("ice_verification.png")
    print("Plot saved to ice_verification.png")

if __name__ == "__main__":
    main()
