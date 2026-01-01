import os
import sys
import numpy as np

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck

def main():
    base_dir = "/root/ECMS_Python"
    # Map paths
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    print("Loading Components...")
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    # Load Cycle
    vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
    cycle_df = loader.read_vmod(vmod_path)
    
    # 1. Check Engine Map
    print("\n--- Engine Map Check (g/s or g/h?) ---")
    pts = [[2000, 500]] # RPM, Torque
    fuel = truck.fuel_interp(pts)
    print(f"(2000 rpm, 500 Nm) -> {fuel[0]:.4f}")
    # Power = 500 * 2000 * 2pi/60 = 104 kW.
    # BSFC ~ 200 g/kWh.
    # Fuel Rate = 104 * 200 = 20800 g/h = 5.7 g/s.
    # If output is ~20000, it's g/h. If ~5, it's g/s.
    
    # 2. Check Motor Map
    print("\n--- Motor Map Check (P_el kW) ---")
    # Motoring
    p_mech = 500 * 2000 * 2 * np.pi / 60.0 / 1000.0 # 104.7 kW
    pts_mot = [[2000, 500]]
    p_el = truck.em_eff_interp(pts_mot)[0]
    print(f"Motoring: (2000 rpm, 500 Nm) -> P_mech={p_mech:.1f} kW, P_el={p_el:.4f} kW")
    # Generating
    pts_gen = [[2000, -500]]
    p_el_gen = truck.em_eff_interp(pts_gen)[0]
    print(f"Generating: (2000 rpm, -500 Nm) -> P_mech=-{p_mech:.1f} kW, P_el={p_el_gen:.4f} kW")
    
    # 4. Cycle Energy Analysis
    print("\n--- Cycle Energy Analysis ---")
    if 'T_ice_fcmap [Nm]' not in cycle_df.columns:
        print("T_ice_fcmap not found? Using calculation.")
        t_reqs = truck.calc_backward_physics(cycle_df)
    else:
        t_reqs = cycle_df['T_ice_fcmap [Nm]'].values
        
    rpms = cycle_df['rpm_ice'].values
    times = cycle_df['time'].values
    dts = np.diff(times, prepend=times[0])
    dts[0] = 0.5
    
    # Energy [kWh]
    # P [kW] = T [Nm] * rpm * 2pi/60 / 1000
    p_mech_kw = t_reqs * rpms * 2 * np.pi / 60.0 / 1000.0
    energy_step_kwh = p_mech_kw * dts / 3600.0
    
    total_pos = np.sum(energy_step_kwh[energy_step_kwh > 0])
    total_neg = np.sum(energy_step_kwh[energy_step_kwh < 0])
    net_energy = np.sum(energy_step_kwh)
    
    print(f"Total Positive Energy (Propulsion): {total_pos:.2f} kWh")
    print(f"Total Negative Energy (Braking):    {total_neg:.2f} kWh")
    print(f"Net Energy Required:                {net_energy:.2f} kWh")
    
    # Fuel Estimate
    # Assume 40% eff -> 10 kWh fuel per 4 kWh mech.
    # 36 kg fuel ~ 1500 MJ ~ 400 kWh chem.
    # 400 kWh chem -> 160 kWh mech.
    # If Net Energy is 160 kWh, then 36 kg is reasonable.
    
if __name__ == "__main__":
    main()
