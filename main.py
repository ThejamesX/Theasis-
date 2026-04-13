import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller

def run_ecms_simulation(strategy='AECMS', cycle_file=None, bat_capacity_kwh=120.0, output_prefix='ecms'):
    # 1. Paths
    # Using absolute paths as requested or safer relative if running from root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cycle_file:
        vmod_path = cycle_file
    else:
        vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
        
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    # Note: "EM_Map - kopie.vemo" might be tricky with spaces if not handled well, but python open() handles it.
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    # New Resistance file check
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    # 2. Initialize
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    
    print(f"--- Running {strategy} | Cap: {bat_capacity_kwh} kWh | Cycle: {os.path.basename(vmod_path)} ---")
    # Pass resistance path (it handles if missing)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    # Override Capacity
    if bat_capacity_kwh is not None:
        truck.bat_params['Capacity'] = float(bat_capacity_kwh)

    # 3. Load Cycle
    cycle_df = loader.read_vmod(vmod_path)
    if cycle_df is None:
        print("Failed to load cycle.")
        return None

    # 4. Backward Physics
    # Add T_req column to cycle vector
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req
    
    # 5. Simulation Loop
    target_soc = 0.70 # Start at 70%
    # Q_LHV = 42700 J/g
    controller = ECMS_Controller(truck, q_lhv=42700.0) 
    
    # Capacity handling strictly
    # VECTO "Capacity" is typically kWh.
    # Q_max [As] = (kWh * 3600 * 1000) / V_nom
    v_nom = truck.ocv_curve(70).item() # 70% SOC voltage
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom
    # print(f"DEBUG: V_nom={v_nom:.2f} V, Cap={cap_kwh:.2f} kWh, Q_max={q_max_as:.2f} As")

    # --- P-ECMS Strategy Selection ---
    # --- Strategy Selection ---
    # Options: 'ECMS', 'A-ECMS', 'LINEAR', 'GRAVITY', 'ENERGY', 'PECMS'
    STRATEGY = strategy
    
    # Imports
    sys.path.append(os.path.join(base_dir, 'P_ECMS')) 
    sys.path.append(os.path.join(base_dir, 'A_ECMS_Implementation'))
    
  
    from P_ECMS.new_horizon_predictor import NewHorizonPredictor
  
    from P_ECMS.pecms_supervisor import PECMS_Supervisor
    from P_ECMS.pecms_supervisor import PECMS_Supervisor
    from A_ECMS_Implementation.aecms_controller import AECMS_Controller

    # Default Predictor (can be overwritten)
    predictor = NewHorizonPredictor(cycle_df, spatial_step=50.0)
    
    supervisor = None
    
    # 1. Standard ECMS (Fixed)
    if STRATEGY == 'ECMS':
        # print("Strategy: Standard ECMS (Fixed Factors)")
        # Base Controller already initialized above
        pass 
        
    # 2. A-ECMS (Adaptive Proportional)
    elif STRATEGY in ['AECMS', 'A-ECMS']:
        # print("Strategy: A-ECMS (Proportional Feedback)")
        # Replace base controller with Adaptive one
        controller = AECMS_Controller(truck, kp_dis=30, kp_chg=0.01, target_soc=target_soc)  
     #3. P-ECMS Variants (Supervisor + Base Controller)
        # Use New PECMS Supervisor (Updated Init)
    elif STRATEGY == 'PECMS':
        # print("Strategy: P-ECMS (Constant Reference)") 
            
        controller.s_dis = 2.3395
        controller.s_chg = 1.7538
        predictor = NewHorizonPredictor(cycle_df, spatial_step=50.0)
        supervisor = PECMS_Supervisor(truck, controller, q_max_as, target_soc=target_soc)
        
    else:
        raise ValueError(f"Unknown Strategy: {STRATEGY}")
    
    # Storage
    results = {
        'time': cycle_df['time'].values,
        'velocity_kmh': cycle_df['velocity_kmh'].values,
        'altitude_m': cycle_df['altitude_m'].values if 'altitude_m' in cycle_df.columns else np.zeros_like(cycle_df['time'].values),
        'rpm_ice': cycle_df['rpm_ice'].values,
        't_req': t_req,
        'soc': [],
        'soc_target': [], 
        't_ice': [],
        't_em': [],
        'fuel_rate': [],
        'p_chem': [],
        'cost_inst': [],
        's_factor': []
    }
    
    soc = target_soc
    curr_target = target_soc # Initial
    
    # print(f"Starting Simulation... (Steps: {len(cycle_df)})")
    
    times = cycle_df['time'].values
    dts = cycle_df['dt'].values # New column from VECTO
    rpms = cycle_df['rpm_ice'].values
    t_reqs = cycle_df['t_req_hybrid_in'].values
    
    total_fuel_g = 0.0
    
    # Running Dist for logging
    curr_dist = 0
    
    for i in range(len(cycle_df)):
        t = times[i]
        rpm = rpms[i]
        tr = t_reqs[i]
        
        # dt from file (Exact VECTO Step)
        dt = dts[i]
        vel = cycle_df['velocity_kmh'].values[i] if 'velocity_kmh' in cycle_df.columns else 0.0
            
        # Linear Target Update
        curr_dist_val = predictor.dist_arr[i] if hasattr(predictor, 'dist_arr') else 0
        total_dist_val = predictor.dist_arr[-1] if hasattr(predictor, 'dist_arr') else 1
        curr_lin_target = 0.70 - (curr_dist_val / total_dist_val) * (0.70 - 0.30)
        
        if hasattr(controller, 'target_soc'):
            controller.target_soc = curr_lin_target

        # --- Update Logic ---
        if supervisor is not None:
             supervisor.target_soc = curr_lin_target
             if hasattr(supervisor, 'soc_nominal'):
                 supervisor.soc_nominal = curr_lin_target
             
             # P-ECMS Update (Every 3 steps? Or every step for accuracy? P-ECMS typically periodic)
             # Let's keep every 3 steps to save compute, matches legacy
             if i % 3 == 0:
                horizon = predictor.get_horizon(i)
                curr_dist = predictor.dist_arr[i]
                
                # Unpack target
                opt_s, curr_target, ratio = supervisor.get_optimal_s(curr_dist, soc, horizon)
                
                # Update Controller
                controller.s_dis = opt_s
                controller.s_chg = opt_s * ratio # maintain ratio
        else:
            # ECMS / A-ECMS
            curr_target = curr_lin_target # Update reference target for plotting
            
        # Logging print (reduced)
        # if i % 5000 == 0:
        #     print(f"Step {i}: s={controller.s_dis:.4f}, SOC={soc:.4f}, Target={curr_target:.4f}")
            
        # Call Controller (Uses updated or internal s)
        t_eng, t_mot, h_cost_watts, p_chem_watts, fuel_g_s = controller.decide_split(tr, rpm, soc, vel)
        
        # Store s and target
        results['s_factor'].append(controller.s_dis)
        results['soc_target'].append(curr_target)
        
        # Update SOC strictly (Eq 13)
        v_oc = truck.ocv_curve(soc * 100.0).item()
        
        # Protect div zero? Voc won't be 0.
        i_bat = p_chem_watts / v_oc
        
        # Eq 13: dot_soc = - i_bat / q_max_as
        dot_soc = - i_bat / q_max_as
        
        soc = soc + dot_soc * dt
        
        # Clamp
        soc = max(0.0, min(1.0, soc))
        
        # Store
        results['soc'].append(soc)
        results['t_ice'].append(t_eng)
        results['t_em'].append(t_mot)
        results['fuel_rate'].append(fuel_g_s)
        results['p_chem'].append(p_chem_watts)
        results['cost_inst'].append(h_cost_watts)
        
        total_fuel_g += fuel_g_s * dt
        
    # print("Simulation Complete.")
    # --- Plotting ---
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 11})
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. SOC
    axes[0].plot(results['time'], np.array(results['soc']) * 100, label='SOC [%]', color='tab:blue', linewidth=1.5)
    axes[0].plot(results['time'], np.array(results['soc_target']) * 100, label='Target SOC [%]', color='black', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('SOC [%]')
    axes[0].set_title(f'Strategy: {STRATEGY} | Cap: {bat_capacity_kwh} | Fuel: {total_fuel_g/1000:.2f} kg', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.7)
    
    # 2. Torque
    axes[1].plot(results['time'], results['t_ice'], label='Engine Torque', color='tab:red', alpha=0.8, linewidth=1)
    axes[1].plot(results['time'], results['t_em'], label='Motor Torque', color='tab:green', alpha=0.8, linewidth=1)
    axes[1].set_ylabel('Torque [Nm]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    plot_filename = os.path.join(out_dir, f"{output_prefix}.png")
    
    plt.savefig(plot_filename, dpi=300)
    plt.close(fig) # Close to release memory
    
    fuel_kg = total_fuel_g / 1000.0
    print(f"Saved {plot_filename} | Fuel: {fuel_kg:.3f} kg")
    
    return fuel_kg, results

def main():
    # Backward Comp for manual run
    run_ecms_simulation()

if __name__ == "__main__":
    main()
