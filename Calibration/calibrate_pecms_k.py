import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add Parent Dir to Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'P_ECMS'))
sys.path.append(os.path.join(parent_dir, 'A_ECMS_Implementation'))

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller
from P_ECMS.horizon_predictor import HorizonPredictor
from P_ECMS.new_horizon_predictor import NewHorizonPredictor
from P_ECMS.pecms_supervisor import PECMS_Supervisor
from A_ECMS_Implementation.aecms_controller import AECMS_Controller

def run_calibration():
    # 1. Setup
    print("Initializing Calibration for P-ECMS Slope Factor (k)...")
    base_dir = parent_dir
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
    if cycle_df is None: return

    # Physics
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req
    
    v_nom = truck.ocv_curve(50).item()
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom
    
    # 2. Calibration Range
    k_values = np.linspace(0, 1, 25)
    
    results_log = []
    
    target_soc = 0.50
    
    print(f"Testing k_slope values: {k_values}")
    
    for k in k_values:
        print(f"\n--- Running Simulation for k_slope = {k} ---")
        
        # Reset Controller & Supervisor per run
        controller = ECMS_Controller(truck, q_lhv=42700.0) 
        
        # Predictor (Re-init to be safe)
        predictor = NewHorizonPredictor(cycle_df, spatial_step=50.0)
        
        # Supervisor with current k
        supervisor = PECMS_Supervisor(truck, controller, q_max_as, target_soc=target_soc, k_slope=k)
        
        # Simulation Loop
        soc = target_soc
        total_fuel_g = 0.0
        
        dts = cycle_df['dt'].values
        rpms = cycle_df['rpm_ice'].values
        t_reqs = cycle_df['t_req_hybrid_in'].values
        
        steps = len(cycle_df)
        
        for i in range(steps):
            dt = dts[i]
            rpm = rpms[i]
            tr = t_reqs[i]
            
            if i % 3 == 0:
                horizon = predictor.get_horizon(i)
                curr_dist = predictor.dist_arr[i]
                opt_s, curr_target, ratio = supervisor.get_optimal_s(curr_dist, soc, horizon)
                controller.s_dis = opt_s
                controller.s_chg = opt_s * ratio

            t_eng, t_mot, h_cost, p_chem, fuel_g_s = controller.decide_split(tr, rpm, soc)
            
            # SOC Update
            v_oc = truck.ocv_curve(soc * 100.0).item()
            i_bat = p_chem / v_oc
            dot_soc = - i_bat / q_max_as
            soc += dot_soc * dt
            soc = max(0.0, min(1.0, soc))
            
            total_fuel_g += fuel_g_s * dt
            
        # Result for this k
        fuel_kg = total_fuel_g / 1000.0
        final_soc = soc
        dsoc = final_soc - target_soc
        
        # Equivalent Fuel (Correction)
        # Using avg LHV and Engine Eff approx or just raw
        # Generic correction: m_eq = m_fuel + (E_batt_used / Q_lhv / eta_avg)
        # E_batt_used = (SOC_start - SOC_end) * Q_batt
        # If SOC_end > SOC_start (dsoc > 0), we gained energy -> Subtract fuel
        # If SOC_end < SOC_start (dsoc < 0), we used energy -> Add fuel
        
        e_batt_kwh = cap_kwh
        e_delta_kwh = (target_soc - final_soc) * e_batt_kwh # Positive if we used battery
        # Approx Engine Eff 0.40, LHV 11.86 kWh/kg
        # m_add = E_delta / (11.86 * 0.40)
        m_equiv_kg = fuel_kg + (e_delta_kwh / (11.86 * 0.40))
        
        print(f"Result k={k}: Fuel={fuel_kg:.3f} kg, FinalSOC={final_soc:.4f}, EquivFuel={m_equiv_kg:.3f} kg")
        
        results_log.append({
            'k_slope': k,
            'fuel_kg': fuel_kg,
            'final_soc': final_soc,
            'equiv_fuel_kg': m_equiv_kg
        })
        
    # 3. Save & Plot
    df_res = pd.DataFrame(results_log)
    print("\nCalibration Results:")
    print(df_res)
    
    df_res.to_csv(os.path.join(current_dir, 'pecms_k_results.csv'), index=False)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('k_slope')
    ax1.set_ylabel('Fuel Consumed [kg]', color=color)
    ax1.plot(df_res['k_slope'], df_res['fuel_kg'], marker='o', color=color, label='Physical Fuel')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Final SOC [-]', color=color)
    ax2.plot(df_res['k_slope'], df_res['final_soc'], marker='x', linestyle='--', color=color, label='Final SOC')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Reference Line
    ax2.axhline(0.50, color='gray', linestyle=':', alpha=0.5)
    
    plt.title('P-ECMS Calibration: Slope Factor (k)')
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'pecms_k_calibration.png'))
    print("Plot saved to pecms_k_calibration.png")

if __name__ == "__main__":
    run_calibration()
