import os
import sys
import numpy as np
from scipy.optimize import minimize
import logging

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller

# Global Instances for speed
loader = None
truck = None
cycle_df = None
v_nom = None
cap_kwh = None
q_max = None
t_reqs = None
rpms = None
dts = None

def setup_simulation():
    global loader, truck, cycle_df, t_reqs, rpms, dts, q_max
    
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
    
    print("Loading Driving Cycle...")
    cycle_df = loader.read_vmod(vmod_path)
    
    # Pre-calc physics
    if 'T_ice_fcmap [Nm]' in cycle_df.columns:
        t_reqs = cycle_df['T_ice_fcmap [Nm]'].values
    else:
        t_reqs = truck.calc_backward_physics(cycle_df)
    
    rpms = cycle_df['rpm_ice'].values
    times = cycle_df['time'].values
    dts = np.diff(times, prepend=times[0])
    dts[0] = 0.5
    
    # Q Max
    v_nom = truck.get_ocv(0.5)
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max = (cap_kwh * 3.6e6) / v_nom

def simulation_objective(x):
    """
    Objective function for optimizer.
    x = [s_dis, s_chg]
    Returns: Fuel + Penalty
    """
    s_dis, s_chg = x
    
    # Controller
    controller = ECMS_Controller(truck, s_dis=s_dis, s_chg=s_chg, q_lhv=42700.0)
    soc = 0.50
    total_fuel_g = 0.0
    
    # Fast Loop
    for i in range(len(dts)):
        # logic
        t_eng, t_mot, _, p_chem, fuel_gs = controller.decide_split(t_reqs[i], rpms[i], soc)
        
        total_fuel_g += fuel_gs * dts[i]
        
        # SOC
        v_oc = truck.get_ocv(soc)
        if v_oc > 0:
            i_bat = p_chem / v_oc
            d_soc = - i_bat / q_max
            soc += d_soc * dts[i]
        soc = max(0.0, min(1.0, soc))
        
    fuel_kg = total_fuel_g / 1000.0
    soc_error = soc - 0.50
    
    # Penalty Function
    # We want exact SOC=0.50.
    # Deviation of 0.01 (1%) is bad.
    # Penalty weight: 1000. 0.01 -> 10kg penalty.
    penalty = 10000.0 * (soc_error ** 2)
    
    cost = fuel_kg + penalty
    print(f"Eval: s_dis={s_dis:.4f}, s_chg={s_chg:.4f} -> Fuel={fuel_kg:.4f}kg, SOC={soc:.4f}, Cost={cost:.4f}")
    
    return cost

def main():
    setup_simulation()
    
    # Initial Guess from Coarse Grid
    x0 = [2.075, 2.051] 
    
    print("\nStarting Nelder-Mead Optimization for Exact Charge Sustenance...")
    print(f"Initial Guess: {x0}")
    
    # Optimization
    res = minimize(
        simulation_objective, 
        x0, 
        method='Nelder-Mead', 
        tol=1e-4,
        options={'maxiter': 50, 'disp': True}
    )
    
    print("\n=== Fine-Tuning Complete ===")
    print(f"Status: {res.message}")
    print(f"Iterations: {res.nit}")
    print(f"Optimal Factors: s_dis={res.x[0]:.6f}, s_chg={res.x[1]:.6f}")
    
    # Final Confirmation Run
    cost = simulation_objective(res.x)
    
if __name__ == "__main__":
    main()
