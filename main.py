import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller

def main():
    # 1. Paths
    # Using absolute paths as requested or safer relative if running from root
    base_dir = "/root/ECMS_Python"
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
    
    print("Loading components...")
    # Pass resistance path (it handles if missing)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    # 3. Load Cycle
    print("Loading Driving Cycle...")
    cycle_df = loader.read_vmod(vmod_path)
    if cycle_df is None:
        print("Failed to load cycle.")
        return

    # 4. Backward Physics
    print("Calculating Physics...")
    # Add T_req column to cycle vector
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req
    
    # 5. Simulation Loop
    target_soc = 0.50
    # Q_LHV = 42700 J/g
    controller = ECMS_Controller(truck, q_lhv=42700.0) 
    rule_based_controller = None # RuleBasedController(truck) <- File deleted by user
    
    # Capacity handling strictly
    # VECTO "Capacity" is typically kWh.
    # Q_max [As] = (kWh * 3600 * 1000) / V_nom
    v_nom = truck.ocv_curve(50).item() # 50% SOC voltage
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom
    print(f"DEBUG: V_nom={v_nom:.2f} V, Cap={cap_kwh:.2f} kWh, Q_max={q_max_as:.2f} As")

    # --- P-ECMS Strategy Selection ---
    # --- Strategy Selection ---
    # Options: 'ECMS', 'A-ECMS', 'LINEAR', 'GRAVITY', 'ENERGY'
    STRATEGY = 'A-ECMS' 
    
    # Imports
    sys.path.append(os.path.join(base_dir, 'P_ECMS')) 
    sys.path.append(os.path.join(base_dir, 'A_ECMS_Implementation'))
    
    from P_ECMS.horizon_predictor import HorizonPredictor
    from P_ECMS.energy_supervisor import EnergyBalanceSupervisor
    from P_ECMS.linear_supervisor import LinearSupervisor
    from P_ECMS.gravity_supervisor import GravitySupervisor
    from A_ECMS_Implementation.aecms_controller import AECMS_Controller

    predictor = HorizonPredictor(cycle_df) 
    total_dist = predictor.dist_arr[-1]
    
    supervisor = None
    
    # 1. Standard ECMS (Fixed)
    if STRATEGY == 'ECMS':
        print("Strategy: Standard ECMS (Fixed Factors)")
        # Base Controller already initialized above
        pass 
        
    # 2. A-ECMS (Adaptive Proportional)
    elif STRATEGY == 'A-ECMS':
        print("Strategy: A-ECMS (Proportional Feedback)")
        # Replace base controller with Adaptive one
        # Uses tuned gains K_dis=20, K_chg=10.1 from calibration
        controller = AECMS_Controller(truck, kp_dis=16.5, kp_chg=2, target_soc=target_soc)
        
    # 3. P-ECMS Variants (Supervisor + Base Controller)
    elif STRATEGY == 'LINEAR':
        print("Strategy: P-ECMS (Linear Reference)")
        supervisor = LinearSupervisor(truck, controller, total_dist, q_max_as, start_soc=target_soc, end_soc=target_soc)
        
    elif STRATEGY == 'GRAVITY':
        print("Strategy: P-ECMS (Gravity-Aware, K=0.0003)")
        supervisor = GravitySupervisor(truck, controller, total_dist, q_max_as, start_soc=target_soc, end_soc=target_soc, k_grav=0.0003)
        
    elif STRATEGY == 'ENERGY':
        print("Strategy: P-ECMS (Energy Balance)")
        supervisor = EnergyBalanceSupervisor(truck, controller, total_dist, q_max_as, start_soc=target_soc, end_soc=target_soc)
        
    else:
        raise ValueError(f"Unknown Strategy: {STRATEGY}")
    
    # Storage
    results = {
        'time': cycle_df['time'].values,
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
    
    print(f"Starting Simulation... (Steps: {len(cycle_df)})")
    
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
            

        # --- Update Logic ---
        if supervisor is not None:
             # P-ECMS Update (Every 3 steps)
            if i % 3 == 0:
                horizon = predictor.get_horizon(i)
                curr_dist = predictor.dist_arr[i]
                
                # Unpack target
                opt_s, curr_target = supervisor.get_optimal_s(curr_dist, soc, horizon)
                
                # Update Controller
                controller.s_dis = opt_s
                controller.s_chg = opt_s * (1.9950 / 2.0886) # maintain ratio
        else:
            # ECMS / A-ECMS
            curr_target = target_soc # Constant target for plotting
            
        # Logging print
        if i % 1000 == 0:
            print(f"Step {i}: s={controller.s_dis:.4f}, SOC={soc:.4f}, Target={curr_target:.4f}")
            
        # Call Controller (Uses updated or internal s)
        t_eng, t_mot, h_cost_watts, p_chem_watts, fuel_g_s = controller.decide_split(tr, rpm, soc)
        
        # Store s and target
        results['s_factor'].append(controller.s_dis)
        results['soc_target'].append(curr_target)
        
        # if i == 0 or i == 1000:
        #     print(f"DEBUG MAIN Step {i}: p_chem={p_chem_watts:.4f}, Fuel={fuel_g_s:.4f}, SOC_start={soc:.6f}")
        
        if i % 1000 == 0:
            print(f"Step {i}: TR={tr:.1f}, TR_eff={t_eng+t_mot:.1f}, T_eng={t_eng:.1f}, T_mot={t_mot:.1f}, Fuel={fuel_g_s:.4f} g/s")
        
        # Update SOC strictly (Eq 13)
        # We need I_bat.
        # P_chem = V_oc * I_bat -> I_bat = P_chem / V_oc
        v_oc = truck.ocv_curve(soc * 100.0).item()
        
        # Protect div zero? Voc won't be 0.
        i_bat = p_chem_watts / v_oc
        
        # Eq 13: dot_soc = - i_bat / Q_max
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
        
    print("Simulation Complete.")
    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    
    # 1. Velocity & Altitude
    axes[0].plot(results['time'], cycle_df['velocity_kmh'], label='Speed [km/h]', color='black')
    axes[0].set_ylabel('Speed [km/h]')
    axes[0].set_title(f'Strategy: {STRATEGY} | Fuel: {total_fuel_g/1000:.2f} kg')
    axes[0].grid(True)
    
    # Twin axis for Altitude
    if 'altitude_m' in cycle_df.columns:
        ax0_alt = axes[0].twinx()
        ax0_alt.plot(results['time'], cycle_df['altitude_m'], label='Altitude [m]', color='gray', alpha=0.5, linestyle='--')
        ax0_alt.set_ylabel('Altitude [m]', color='gray')
    
    # 2. SOC
    axes[1].plot(results['time'], np.array(results['soc']) * 100, label='SOC [%]', color='blue')
    axes[1].plot(results['time'], np.array(results['soc_target']) * 100, label='Target SOC [%]', color='red', linestyle='--')
    axes[1].set_ylabel('SOC [%]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    # 3. Torque
    axes[2].plot(results['time'], results['t_ice'], label='Engine Torque', color='red', alpha=0.7)
    axes[2].plot(results['time'], results['t_em'], label='Motor Torque', color='green', alpha=0.7)
    axes[2].set_ylabel('Torque [Nm]')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)
    
    # 4. Equivalence Factor (s)
    axes[3].plot(results['time'], results['s_factor'], label='Equiv Factor (s)', color='purple')
    for y in [2.0886, 2.40]: # Visual reference lines
        axes[3].axhline(y, color='gray', linestyle=':', alpha=0.5)
    axes[3].set_ylabel('s [-]')
    axes[3].set_xlabel('Time [s]')
    axes[3].legend(loc='upper right')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig('ecms_results.png')
    print("Plot saved to ecms_results.png")
    
    # Compare with Baseline
    baseline_fuel_kg = 36.55 # VECTO Reference
    fuel_kg = total_fuel_g / 1000.0
    savings = (1 - fuel_kg/baseline_fuel_kg) * 100
    print(f"Total Fuel Consumed (Hybrid): {fuel_kg:.2f} kg")
    print(f"Baseline Fuel (VECTO): {baseline_fuel_kg:.2f} kg")
    print(f"Potential Savings: {savings:.2f}%")
    print(f"Final SOC: {soc:.4f}")

if __name__ == "__main__":
    main()
