import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller
from pecms_supervisor import HorizonPredictor, PECMS_Supervisor

def calibrate_kgrav():
    # 1. Setup
    base_dir = "/root/ECMS_Python"
    vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    cycle_df = loader.read_vmod(vmod_path)
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req

    v_nom = truck.ocv_curve(50).item()
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom
    
    # 2. Params
    k_candidates = np.linspace(0, 0.0015, 70)
        
    results = []
    
    print(f"Starting K_grav Calibration. Candidates: {k_candidates}")
    
    predictor = HorizonPredictor(cycle_df)
    total_dist = predictor.dist_arr[-1]
    
    for k_val in k_candidates:
        print(f"Testing K_grav = {k_val} ...")
        
        # Reset Controller
        controller = ECMS_Controller(truck, q_lhv=42700.0)
        supervisor = PECMS_Supervisor(truck, controller, total_dist, q_max_as, k_grav=k_val)
        
        # Simulation
        soc = 0.50
        times = cycle_df['time'].values
        dts = cycle_df['dt'].values
        rpms = cycle_df['rpm_ice'].values
        t_reqs = cycle_df['t_req_hybrid_in'].values
        
        total_fuel = 0.0
        
        for i in range(len(cycle_df)):
            if i % 3 == 0:
                horizon = predictor.get_horizon(i)
                curr_dist = predictor.dist_arr[i]
                opt_s, _ = supervisor.get_optimal_s(curr_dist, soc, horizon)
                controller.s_dis = opt_s
                controller.s_chg = opt_s * (1.9950 / 2.0886)
                
            t_eng, t_mot, h_cost, p_chem, fuel_rate = controller.decide_split(t_reqs[i], rpms[i], soc)
            
            # Physics Update
            u_oc = truck.get_ocv(soc)
            i_bat = p_chem / u_oc # Approx for SOC update loop. Ideally matches plant.
            # Use p2_hybrid calc_battery_dynamics? 
            # Or assume decide_split result p_chem is accurate enough? 
            # Controller uses: p_chem = u * i. Rigid.
            
            # Use standard update to be safe
            d_soc = - (i_bat * dts[i]) / q_max_as
            soc += d_soc
            
            total_fuel += fuel_rate * dts[i]
            
        final_soc = soc
        
        # Correct for SOC deviation
        # Fuel_equiv = Fuel + s * Q_delta / Q_lhv
        # Use average S ~ 2.0
        fuel_equiv = total_fuel + (0.50 - final_soc) * q_max_as * 600 * 2.0 / 42700000 
        # Very rough correction. Better to check if strictly sustaining.
        # But comparisons are valid if final SOC is close.
        
        print(f"  -> Fuel: {total_fuel/1000:.3f} kg, Final SOC: {final_soc:.4f}")
        results.append({'k': k_val, 'fuel': total_fuel/1000, 'soc': final_soc})
        
    # Best
    best_run = min(results, key=lambda x: x['fuel'])
    print("\nCalibration Complete.")
    print(f"Best K_grav: {best_run['k']} (Fuel: {best_run['fuel']:.3f} kg)")
    
    # Save results
    res_df = pd.DataFrame(results)
    res_df.to_csv("Calibration/kgrav_results.csv", index=False)

if __name__ == "__main__":
    calibrate_kgrav()
