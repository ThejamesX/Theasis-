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
    # Options: 'LINEAR', 'GRAVITY', 'ENERGY'
    STRATEGY = 'GRAVITY' 
    
    # Imports
    sys.path.append(os.path.join(base_dir, 'P_ECMS')) 
    from P_ECMS.horizon_predictor import HorizonPredictor
    from P_ECMS.energy_supervisor import EnergyBalanceSupervisor
    from P_ECMS.linear_supervisor import LinearSupervisor
    from P_ECMS.gravity_supervisor import GravitySupervisor

    predictor = HorizonPredictor(cycle_df) 
    total_dist = predictor.dist_arr[-1]
    
    supervisor = None
    
    if STRATEGY == 'LINEAR':
        print("Strategy: Normal P-ECMS (Linear Reference)")
        supervisor = LinearSupervisor(truck, controller, total_dist, q_max_as, start_soc=target_soc, end_soc=target_soc)
        
    elif STRATEGY == 'GRAVITY':
        print("Strategy: Gravity-Aware P-ECMS (K_grav=0.0003)")
        supervisor = GravitySupervisor(truck, controller, total_dist, q_max_as, start_soc=target_soc, end_soc=target_soc, k_grav=0.0003)
        
    elif STRATEGY == 'ENERGY':
        print("Strategy: Energy Balance P-ECMS (Potential Energy)")
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
            

        # --- P-ECMS Update (Every 3s or 3 steps?) ---
        if i % 3 == 0:
            # Get Horizon
            horizon = predictor.get_horizon(i)
            # Get Optimal s
            curr_dist = predictor.dist_arr[i]
            
            # Unpack target
            opt_s, curr_target = supervisor.get_optimal_s(curr_dist, soc, horizon)
            
            # Update Controller
            controller.s_dis = opt_s
            controller.s_chg = opt_s * (1.9950 / 2.0886) # maintain ratio
            
            # Print occasionally
            if i % 1000 == 0:
                print(f"Step {i}: s_opt={opt_s:.4f}, SOC={soc:.4f}, Target={curr_target:.4f}")
            
        # Call Controller (Now uses updated s)
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
    print(f"Total Fuel Consumed (Hybrid): {total_fuel_g/1000:.2f} kg")
    
    # Baseline comparison 
    if 'FC-Map [g/h]' in cycle_df.columns:
        base_fuel_rate = cycle_df['FC-Map [g/h]'].fillna(0).values / 3600.0
        dt_arr = np.diff(times, prepend=times[0]) 
        total_base_g = np.sum(base_fuel_rate * dt_arr)
        print(f"Baseline Fuel (VECTO): {total_base_g/1000:.2f} kg")
        savings = (1 - total_fuel_g/total_base_g) * 100.0
        print(f"Potential Savings: {savings:.2f}%")
        print(f"Final SOC: {soc:.4f}")
        
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Top: Speed & Altitude
    ax1 = axs[0]
    ax1.plot(results['time'], cycle_df['velocity_kmh'], 'b-', label='Speed [km/h]')
    ax1.set_ylabel('Speed [km/h]', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Altitude if available
    if 'altitude_m' in cycle_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(results['time'], cycle_df['altitude_m'], 'g--', label='Altitude', alpha=0.5)
        ax2.set_ylabel('Altitude [m]', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
    ax1.set_title("Driving Cycle")
    
    # Middle: Torque Split
    ax3 = axs[1]
    ax3.plot(results['time'], results['t_ice'], 'r', label='Engine Torque', linewidth=1)
    ax3.plot(results['time'], results['t_em'], 'c', label='Motor Torque', linewidth=1)
    ax3.plot(results['time'], t_reqs, 'k--', label='Total Req', alpha=0.3)
    ax3.set_ylabel('Torque [Nm]')
    ax3.legend()
    ax3.set_title("Torque Split (ECMS)")
    
    # Bottom: SOC
    ax4 = axs[2]
    ax4.plot(results['time'], np.array(results['soc']) * 100, 'm-', label='SOC [%]')
    # Plot Dynamic Target
    if len(results['soc_target']) == len(results['time']):
        ax4.plot(results['time'], np.array(results['soc_target']) * 100, 'r--', label='SOC Target (Energy Balance)', linewidth=2)
    else:
        ax4.axhline(y=target_soc*100, color='k', linestyle='--', label='Target')
        
    ax4.set_ylabel('SOC [%]')
    ax4.set_xlabel('Time [s]')
    ax4.grid(True)
    ax4.legend()
    ax4.set_title("Battery State of Charge")
    
    plt.tight_layout()
    plt.savefig('ecms_results.png')
    print("Plot saved to ecms_results.png")

if __name__ == "__main__":
    main()
