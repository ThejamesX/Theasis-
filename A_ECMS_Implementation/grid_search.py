import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from A_ECMS_Implementation.aecms_controller import AECMS_Controller

def grid_search_aecms():
    # 1. Load Data
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    
    # Paths
    base_dir = parent_dir
    vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")
    
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    cycle_df = loader.read_vmod(vmod_path)
    
    # Physics
    t_req_arr = truck.calc_backward_physics(cycle_df)
    dts = np.diff(cycle_df['time'].values, prepend=0)
    dts[0] = 0.5
    rpms = cycle_df['rpm_ice'].values
    steps = len(cycle_df)
    q_max_as = (120.0 * 3.6e6) / truck.get_ocv(0.5)
    
    target_soc = 0.50
    
    # 2. Grid Setup
    # 10x10 grid? 100 runs.
    kp_dis_vals = np.linspace(10, 30, 20)
    kp_chg_vals = np.linspace(0.01, 5, 20)
    
    results = []
    
    print(f"Starting Grid Search (Total {len(kp_dis_vals)*len(kp_chg_vals)} runs)...")
    
    for kd in kp_dis_vals:
        for kc in kp_chg_vals:
            ctrl = AECMS_Controller(truck, kp_dis=kd, kp_chg=kc, target_soc=target_soc)
            soc = target_soc
            total_fuel = 0.0
            
            # Fast Sim Loop
            for i in range(steps):
                res = ctrl.decide_split(t_req_arr[i], rpms[i], soc)
                fuel_rate = res[4]
                p_chem = res[3]
                
                total_fuel += fuel_rate * dts[i]
                
                u_oc = truck.get_ocv(soc)
                dsoc = -( (p_chem/u_oc) * dts[i] ) / q_max_as
                soc += dsoc
                
            final_soc = soc
            soc_dev = abs(final_soc - target_soc) * 100
            fuel_kg = total_fuel / 1000.0
            
            # Penalized Metric? Cost = Fuel + 100 * Dev?
            # Or just raw metrics
            print(f"Kd={kd:.1f}, Kc={kc:.1f} -> Fuel={fuel_kg:.2f}kg, Dev={soc_dev:.2f}%")
            
            results.append({
                'kp_dis': kd,
                'kp_chg': kc,
                'fuel': fuel_kg,
                'dev': soc_dev
            })

    # 3. Analyze
    df = pd.DataFrame(results)
    df.to_csv('aecms_grid_results.csv', index=False)
    
    # Best Fuel with Dev < 1%
    valid = df[df['dev'] < 1.0]
    if not valid.empty:
        best = valid.loc[valid['fuel'].idxmin()]
        print("\n--- BEST VALID RESULT (Dev < 1%) ---")
        print(best)
    else:
        print("\nNo run satisfied deviation < 1%. Best overall fuel:")
        best = df.loc[df['fuel'].idxmin()]
        print(best)
        
    # 4. Plot Heatmap
    # Reshape
    pivot = df.pivot(index='kp_dis', columns='kp_chg', values='fuel')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, cmap='viridis_r', interpolation='nearest', origin='lower',
               extent=[kp_chg_vals.min(), kp_chg_vals.max(), kp_dis_vals.min(), kp_dis_vals.max()])
    plt.colorbar(label='Fuel [kg]')
    plt.xlabel('Kp Charge')
    plt.ylabel('Kp Discharge')
    plt.title('A-ECMS Fuel Consumption Grid Search')
    plt.savefig('aecms_grid_heatmap.png')
    print("Saved heatmap to aecms_grid_heatmap.png")
    
    # 5. 3D Surface Plot (Mesh Grid)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(kp_chg_vals, kp_dis_vals)
    Z = pivot.values
    
    # Plot Surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis_r', edgecolor='none', alpha=0.9)
    # View from the other side
    ax.view_init(elev=30, azim=135)
    
    # Labels
    ax.set_xlabel('Kp Charge')
    ax.set_ylabel('Kp Discharge')
    ax.set_zlabel('Fuel Consumed [kg]')
    ax.set_title(f'A-ECMS Fuel Optimization Landscape\nBest: {best["fuel"]:.4f} kg')
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Fuel [kg]')
    
    plt.savefig('aecms_3d_surface.png')
    print("Saved 3D surface plot to aecms_3d_surface.png")

if __name__ == "__main__":
    grid_search_aecms()
