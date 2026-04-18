import os
import sys
import numpy as np

# Ensure the root directory is in sys.path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.append(base_dir)

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller

def run_test():
    # 1. Initialize Loader and Truck
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    
    # Paths to components
    vmap_path = os.path.join(base_dir, "Engine", "325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor", "P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor", "EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor", "P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor", "REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor", "REESS_Internal_Resistance.vbatr")

    print("Loading truck components...")
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    # 2. Load the specific cycle
    vmod_path = os.path.join(base_dir, "Driving Cycle", "RegionalDeliveryEMSReferenceLoad.vmod")
    print(f"Loading cycle: {vmod_path}...")
    cycle_df = loader.read_vmod(vmod_path)
    if cycle_df is None:
        print("Failed to load cycle.")
        return
        
    print("Calculatingb ackward physics...")
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req
    
    # 3. Setup ECMS Controller with given factors 
    s_chg = 1.649367088607595
    s_dis = 1.907594936708861
    print(f"Initializing ECMS Controller with s_chg={s_chg}, s_dis={s_dis}...")
    controller = ECMS_Controller(truck, s_chg=s_chg, s_dis=s_dis, q_lhv=42700.0)
    
    # Simulation setup
    initial_soc = 0.70 # Default starting SOC in standard runs
    soc = initial_soc
    
    # Capacity calculation
    v_nom = truck.ocv_curve(70).item()
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom

    times = cycle_df['time'].values
    dts = cycle_df['dt'].values
    rpms = cycle_df['rpm_ice'].values
    t_reqs = cycle_df['t_req_hybrid_in'].values
    
    total_fuel_g = 0.0
    
    print(f"--- Output ---")
    print(f"Initial SOC: {soc:.4f}")
    
    # 4. Run loop
    for i in range(len(cycle_df)):
        vel = cycle_df['velocity_kmh'].values[i] if 'velocity_kmh' in cycle_df.columns else 0.0
        tr = t_reqs[i]
        rpm = rpms[i]
        dt = dts[i]
        
        # Determine split
        t_eng, t_mot, h_cost_watts, p_chem_watts, fuel_g_s = controller.decide_split(tr, rpm, soc, vel)
        
        # Update SOC
        v_oc = truck.ocv_curve(soc * 100.0).item()
        i_bat = p_chem_watts / v_oc
        dot_soc = - i_bat / q_max_as
        
        soc += dot_soc * dt
        soc = max(0.0, min(1.0, soc))
        
        total_fuel_g += fuel_g_s * dt

    print(f"Final SOC: {soc:.4f}")
    print(f"Total Fuel: {total_fuel_g/1000.0:.4f} kg")
    
    # Check target
    target_soc = 0.30
    dist_to_target = soc - target_soc
    print(f"Difference to DP Target (0.30): {dist_to_target:.4f}")
    if soc < target_soc:
        print("Result: YES, the ECMS final SOC is lower than 0.30.")
    else:
        print("Result: NO, the ECMS final SOC is higher than or equal to 0.30.")

if __name__ == '__main__':
    run_test()
