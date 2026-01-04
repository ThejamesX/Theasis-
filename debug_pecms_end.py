import os
import sys
import numpy as np
import pandas as pd
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller

# Add paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'P_ECMS')) 
from P_ECMS.pecms_supervisor import PECMS_Supervisor
from P_ECMS.new_horizon_predictor import NewHorizonPredictor

class DebugPECMS_Supervisor(PECMS_Supervisor):
    def get_optimal_s_debug(self, current_dist, current_soc, horizon_data, step_idx, total_steps):
        # Only debug near end
        is_debug = (total_steps - step_idx) < 500
        
        # 1. Target SOC logic (Copy-paste logic to trace it)
        try:
             # Just call base helper? calculate_horizon_energy_delta is internal
             # But self.total_dist_m might be missing
             pass
        except:
             pass
             
        # Call base implementation to get 'candidates' logic... 
        # But we need to see INSIDE the loop. 
        # So we must reimplement/copy the loop for debugging.
        
        # --- RE-IMPLEMENTATION FOR DEBUG LOGGING ---
        calculate_horizon_energy_delta = self.calculate_horizon_energy_delta(horizon_data)
        soc_adj = 0.25 * calculate_horizon_energy_delta
        
        # Check landing logic presence
        if hasattr(self, 'total_dist_m') and self.total_dist_m is not None:
             dist_remaining = self.total_dist_m - current_dist
             if dist_remaining < 2500.0 and dist_remaining > 0:
                 fade = dist_remaining / 2500.0
                 soc_adj = soc_adj * fade
                 if is_debug and (step_idx % 50 == 0):
                     print(f"[DEBUG] Step {step_idx} | Landing Fade={fade:.3f}")

        soc_target = max(0.35, min(0.75, self.soc_nominal + soc_adj))
        
        # Physics
        vels_kmh = horizon_data.get('vel_kmh', np.zeros_like(horizon_data['rpms']))
        vels = vels_kmh / 3.6
        grades = horizon_data.get('grades', np.zeros_like(vels))
        dts = horizon_data.get('dts', np.ones_like(vels))
        
        accels = np.zeros_like(vels)
        if len(vels) > 1:
            dv = np.diff(vels) 
            dt_steps = dts[:-1]
            dt_steps[dt_steps < 0.01] = 0.01 
            accels[:-1] = dv / dt_steps
            accels[-1] = accels[-2] 
        else:
             accels[:] = 0.0 
        f_aero = 0.5 * self.rho * self.cd * self.area * (vels**2)
        theta = np.arctan(grades)
        f_roll = self.cr * self.mass * 9.81 * np.cos(theta)
        f_grade = self.mass * 9.81 * np.sin(theta)
        f_acc = self.mass * accels
        p_load_kw = ((f_aero + f_roll + f_grade + f_acc) * vels) / 1000.0
        p_trans_in = np.where(p_load_kw >= 0, p_load_kw / self.eta_trans, p_load_kw * self.eta_trans)
        rpms = horizon_data['rpms']
        omega = rpms * 2 * np.pi / 60.0
        omega[omega < 1.0] = 1.0 
        t_reqs_calc = (p_trans_in * 1000.0) / omega
        
        candidates = [
            self.last_opt_s,
            self.last_opt_s - 1 * self.delta_s,
            self.last_opt_s + 1 * self.delta_s,
            self.last_opt_s - 2 * self.delta_s,
            self.last_opt_s + 2 * self.delta_s,
        ]
        
        best_s = self.last_opt_s
        min_error = float('inf')
        
        orig_s_dis = self.controller.s_dis
        orig_s_chg = self.controller.s_chg
        
        debug_logs = []
        
        steps = len(rpms)
        for s in candidates:
            if s < self.s_min: s = self.s_min
            if s > self.s_max: s = self.s_max
            
            sim_soc = current_soc
            self.controller.s_dis = s
            self.controller.s_chg = s * self.ratio
            
            sim_valid = True
            for k in range(steps):
                try:
                    res = self.controller.decide_split(t_reqs_calc[k], rpms[k], sim_soc)
                    p_chem = res[3] 
                    u_oc = self.veh.get_ocv(sim_soc)
                    i_bat = p_chem / u_oc
                    d_soc = - (i_bat * dts[k]) / self.q_max_as
                    sim_soc += d_soc
                except:
                    sim_valid = False
                    break
            
            if sim_valid:
                error = abs(sim_soc - soc_target)
                debug_logs.append(f"s={s:.4f} -> SOC={sim_soc:.6f} (Err={error:.6f})")
                if error < min_error:
                    min_error = error
                    best_s = s
        
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        self.last_opt_s = best_s
        
        if is_debug and (step_idx % 20 == 0):
             print(f"Step {step_idx} | Current SOC: {current_soc:.6f} | Target: {soc_target:.6f}")
             print("   Candidates: " + " | ".join(debug_logs))
             print(f"   Chosen: {best_s:.4f}")
             
        return best_s, soc_target, self.ratio

def run_debug():
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
    
    # Calc Physics (Crucial Step Missing)
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req
    
    # Calc dist manually like main.py
    v_mps = cycle_df['velocity_kmh'] / 3.6
    dts = cycle_df['dt'] if 'dt' in cycle_df.columns else np.diff(cycle_df['time'], prepend=0)
    total_dist_m = (v_mps * dts).sum() 
    cycle_df['dist_accum_m'] = np.cumsum(v_mps * dts) # Ensure this exists
    
    predictor = NewHorizonPredictor(cycle_df, spatial_step=50.0)
    controller = ECMS_Controller(truck, q_lhv=42700.0)
    
    # Init Supervisor (Correct way)
    v_nom = truck.ocv_curve(50).item()
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom
    
    supervisor = DebugPECMS_Supervisor(truck, controller, q_max_as, target_soc=0.50)
    
    soc = 0.50
    curr_dist = 0
    total_steps = len(cycle_df)
    
    print("Starting Debug Simulation...")
    
    for i in range(total_steps):
        # t_req = truck.calc_backward_physics_step(cycle_df.iloc[[i]]).item() # Non-existent method
        t_req = cycle_df['t_req_hybrid_in'].iloc[i] # Use pre-calc
        
        # Hack for speed: just check if we update
        
        if i % 3 == 0:
            horizon = predictor.get_horizon(i)
            curr_dist = predictor.dist_arr[i]
            # Use DEBUG method
            opt_s, target, ratio = supervisor.get_optimal_s_debug(curr_dist, soc, horizon, i, total_steps)
            controller.s_dis = opt_s
            controller.s_chg = opt_s * ratio
            
        # Run step (Mock or Real)
        # We need real SOC evolution
        # Simplified step:
        t_req_val = cycle_df['T_ice_fcmap [Nm]'].iloc[i] # Use VECTO for ref
        rpm = cycle_df['rpm_ice'].iloc[i]
        dt = dts[i]
        
        res = controller.decide_split(t_req_val, rpm, soc)
        p_chem = res[3]
        v_oc = truck.ocv_curve(soc * 100.0).item()
        i_bat = p_chem / v_oc
        soc += (-i_bat / q_max_as) * dt
        soc = max(0.0, min(1.0, soc))

if __name__ == "__main__":
    run_debug()
