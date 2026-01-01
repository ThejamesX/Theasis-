import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from dp_optimizer import DPOptimizer

def main():
    base_dir = "/root/ECMS_Python"
    vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    print("Loading Components...")
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    print("Loading Cycle...")
    cycle_df = loader.read_vmod(vmod_path)
    # Debug: short cycle?
    # cycle_df = cycle_df.iloc[:2000] # For speed debug
    
    print("Initializing DP Optimizer...")
    # 150 grid points for accuracy as requested
    optimizer = DPOptimizer(truck, cycle_df, soc_grid_size=150)
    
    print("Solving DP (Backward Sweep)...")
    # Solve for 0.50 Target
    J = optimizer.solve(start_soc=0.50, target_soc=0.5)
    
    print("Reconstructing Optimal Path...")
    res = optimizer.reconstruct_path(start_soc=0.50)
    
    print(f"\n=== DP RESULTS ===")
    print(f"Total Fuel: {res['total_fuel_kg']:.4f} kg")
    print(f"Final SOC:  {res['soc'][-1]:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(res['time'], res['soc'] * 100, label='DP Optimal', linewidth=2)
    plt.axhline(50.0, color='g', linestyle='--', label='Target')
    plt.ylabel('SOC [%]')
    plt.xlabel('Time [s]')
    plt.title('Dynamic Programming Optimal SOC Trajectory')
    plt.legend()
    plt.grid(True)
    plt.savefig('dp_result.png')
    print("Saved dp_result.png")

if __name__ == "__main__":
    main()
