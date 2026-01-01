import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from A_ECMS_Implementation.aecms_controller import AECMS_Controller

def calibrate_kp():
    # 1. Load Data
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    
    # Paths (Assumed relative to parent)
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
    
    # Pre-calc Physics
    # Reuse truck logic or manual? truck logic is safer
    t_req_arr = truck.calc_backward_physics(cycle_df)
    
    # Sim Params
    dts = np.diff(cycle_df['time'].values, prepend=0)
    dts[0] = 0.5
    rpms = cycle_df['rpm_ice'].values
    steps = len(cycle_df)
    q_max_as = (120.0 * 3.6e6) / truck.get_ocv(0.5) # Approx
    
    # 2. Sweep Kp
    k_values = np.linspace(0.2, 20, 80)
    results = []
    
    target_soc = 0.50
    
    print(f"Starting Kp Calibration. {len(k_values)} runs...")
    
    trajectories = {} # Store a few for plotting
    
    for kp in k_values:
        print(f"Testing Kp={kp:.2f}...", end='')
        
        ctrl = AECMS_Controller(truck,kp_dis=kp, kp_chg=kp, target_soc=target_soc)
        soc = target_soc
        total_fuel = 0.0
        soc_hist = [soc]
        
        for i in range(steps):
            t_req = t_req_arr[i]
            rpm = rpms[i]
            dt = dts[i]
            
            res = ctrl.decide_split(t_req, rpm, soc)
            # res: t_eng, t_mot, cost, p_chem, fuel_rate
            fuel_rate = res[4]
            p_chem = res[3]
            
            total_fuel += fuel_rate * dt
            
            # Update SOC
            u_oc = truck.get_ocv(soc)
            i_bat = p_chem / u_oc
            dsoc = -(i_bat * dt) / q_max_as
            soc += dsoc
            soc_hist.append(soc)
            
        final_soc = soc
        soc_dev = abs(final_soc - target_soc) * 100 # %
        fuel_kg = total_fuel / 1000.0
        
        print(f" Fuel={fuel_kg:.2f} kg, Dev={soc_dev:.2f}%")
        results.append({
            'kp': kp, 
            'fuel': fuel_kg, 
            'dev': soc_dev
        })
        
        # Save trajectories for Low, Mid, High
        if kp == k_values[0]: trajectories['low'] = (kp, soc_hist)
        if kp == k_values[10]: trajectories['mid'] = (kp, soc_hist)
        if kp == k_values[-1]: trajectories['high'] = (kp, soc_hist)

    # 3. Plotting
    res_df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Graph 1: SOC Deviation
    axes[0].plot(res_df['kp'], res_df['dev'], 'b-o')
    axes[0].axhline(0, color='k', linestyle='--')
    axes[0].set_title('Final SOC Deviation vs Kp')
    axes[0].set_ylabel('Abs Deviation [%]')
    axes[0].grid(True)
    
    # Graph 2: Fuel
    axes[1].plot(res_df['kp'], res_df['fuel'], 'g-o')
    axes[1].set_title('Total Fuel vs Kp')
    axes[1].set_ylabel('Fuel [kg]')
    axes[1].grid(True)
    
    # Graph 3: Trajectories
    time_axis = np.linspace(0, len(cycle_df), len(trajectories['low'][1]))
    axes[2].plot(time_axis, np.array(trajectories['low'][1])*100, label=f'Kp={trajectories["low"][0]:.2f}')
    axes[2].plot(time_axis, np.array(trajectories['mid'][1])*100, label=f'Kp={trajectories["mid"][0]:.2f}')
    axes[2].plot(time_axis, np.array(trajectories['high'][1])*100, label=f'Kp={trajectories["high"][0]:.2f}')
    axes[2].axhline(target_soc*100, color='r', linestyle='--', label='Target')
    axes[2].set_title('SOC Trajectories')
    axes[2].set_ylabel('SOC [%]')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('aecms_calibration.png')
    print("Calibration Complete. Saved 'aecms_calibration.png'.")
    
if __name__ == "__main__":
    calibrate_kp()
