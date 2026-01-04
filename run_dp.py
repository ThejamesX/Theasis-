import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from dp_optimizer import DPOptimizer

def run_dp_simulation(cycle_file=None, bat_capacity_kwh=120.0, output_prefix='dp'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cycle_file:
        vmod_path = cycle_file
    else:
        vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
        
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    # print("Loading Components...")
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    if bat_capacity_kwh is not None:
        truck.bat_params['Capacity'] = float(bat_capacity_kwh)
    
    # print("Loading Cycle...")
    cycle_df = loader.read_vmod(vmod_path)
    
    print(f"--- Running DP | Cap: {bat_capacity_kwh} kWh | Cycle: {os.path.basename(vmod_path)} ---")
    
    # UPDATED SETTINGS from User Request
    optimizer = DPOptimizer(truck, cycle_df, soc_grid_size=400)
    
    # print("Solving DP (Backward Sweep)...")
    # UPDATED SETTINGS from User Request (Target 0.51)
    J = optimizer.solve(start_soc=0.50, target_soc=0.51)
    
    # print("Reconstructing Optimal Path...")
    res = optimizer.reconstruct_path(start_soc=0.50)
    
    fuel_kg = res['total_fuel_kg']
    print(f"DP Result: {fuel_kg:.3f} kg")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(res['time'], res['soc'] * 100, label='DP Optimal', linewidth=2)
    plt.axhline(51.0, color='g', linestyle='--', label='Target')
    plt.ylabel('SOC [%]')
    plt.xlabel('Time [s]')
    plt.title(f'DP Optimal | Cap: {bat_capacity_kwh} | Fuel: {fuel_kg:.3f} kg')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f"{output_prefix}.png"
    plt.savefig(plot_filename)
    plt.close()
    
    return fuel_kg

def main():
    run_dp_simulation()
