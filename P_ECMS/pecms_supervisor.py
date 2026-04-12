import numpy as np
import pandas as pd

class PECMS_Supervisor:
    """
    Implements P-ECMS Supervisory with Physics-based Look-Ahead.
    Optimized for Real-Time execution using Local Neighborhood Search.
    
    Logic:
    1. Retrieve Speed Vector from Horizon (Spatial Nodes).
    2. Calculate Forces (Aero, Roll, Grade, Accel) -> P_wheel -> T_req internally.
    3. Simulate multiple s-candidates locally centered on previous optimum.
    """
    def __init__(self, vehicle, controller, q_max_as, target_soc=0.50, k_slope=1):
        self.veh = vehicle
        self.controller = controller # Instance of ECMS_Controller
        self.target_soc = target_soc
        self.q_max_as = q_max_as
        self.k_slope = k_slope
        self.soc_nominal = target_soc

        
        # --- Physics Parameters (Extracted from Vehicle) ---
        self.mass = getattr(self.veh, 'M_veh', 40000.0)
        self.cd = getattr(self.veh, 'Cd', 0.6)
        self.area = getattr(self.veh, 'A_front', 10.0)
        self.cr = getattr(self.veh, 'C_roll', 0.006)
        self.rho = getattr(self.veh, 'Rho_air', 1.2)
        # Assuming Trans efficiency is needed for Load -> Engine Torque
        self.eta_trans = getattr(self.veh, 'eta_trans', 0.96)
        
        self.s_dis_0 = 2.3395
        self.s_chg_0 = 1.7538
        self.ratio = self.s_chg_0 / self.s_dis_0
        
        # --- Optimization State Memory ---
        # Initialize with the hardcoded base value
        self.last_opt_s = self.s_dis_0
        
        # Params (Bounds)
        self.delta_s = 0.025
        self.last_target = self.target_soc
        
        # "min 1.6 for charge and max 2.6 for discharge based on the ratio"
        # s_chg >= 1.6  =>  s_dis * ratio >= 1.6  =>  s_dis >= 1.6 / ratio
        # s_dis <= 2.6
        self.s_max = 2.6
        self.s_min = 1.4 / self.ratio



    def get_optimal_s(self, current_dist, current_soc, horizon_data):

        # 1. Update Target SOC with Slope Adjustment
        soc_adj = 0 
        self.target_soc = self.soc_nominal + soc_adj
        
    
        self.target_soc = max(0.30, min(0.75, self.target_soc))
        
        # 2. Internal Physics Calculation
        # Force Calculation using Speed Vector + Grade + Spatial Accel
        vels_kmh = horizon_data.get('vel_kmh', np.zeros_like(horizon_data['rpms']))
        vels = vels_kmh / 3.6 # m/s
        grades = horizon_data.get('grades', np.zeros_like(vels))
        
        dts = horizon_data.get('dts', np.ones_like(vels))
        
        # Gradient accel (Spatial/Forward Diff)
        accels = np.zeros_like(vels)
        dv = np.diff(vels) 
        dt_steps = dts[:-1]
        dt_steps[dt_steps < 0.01] = 0.01 
        accels[:-1] = dv / dt_steps
        accels[-1] = accels[-2] 
        
        # Forces
        f_aero = 0.5 * self.rho * self.cd * self.area * (vels**2)
        theta = np.arctan(grades)
        f_roll = self.cr * self.mass * 9.81 * np.cos(theta)
        f_grade = self.mass * 9.81 * np.sin(theta)
        f_acc = self.mass * accels
        
        f_total = f_aero + f_roll + f_grade + f_acc
        
        # Power at Wheel [kW]
        p_load_kw = (f_total * vels) / 1000.0
        
        # P_trans_in
        p_trans_in_arr = np.where(p_load_kw >= 0, p_load_kw / self.eta_trans, p_load_kw * self.eta_trans)
        
        # T_req from VECTO RPM (Baseline Strategy)
        rpms = horizon_data['rpms']
        omega = rpms * 2 * np.pi / 60.0
        omega[omega < 1.0] = 1.0 
        
        t_reqs_calc = (p_trans_in_arr * 1000.0) / omega
        
        # 3. Optimization (Local Search)
        steps = len(rpms)
        
        candidates = [
            self.last_opt_s, # Check current first (Stability Bias)
            self.last_opt_s - 1 * self.delta_s,
            self.last_opt_s + 1 * self.delta_s,
            self.last_opt_s - 2 * self.delta_s,
            self.last_opt_s + 2 * self.delta_s,
        ]
        
        best_s = self.last_opt_s
        min_error = float('inf')
        
        orig_s_dis = self.controller.s_dis
        orig_s_chg = self.controller.s_chg
        
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
                    
                except Exception as e:
                    print(f"DEBUG PECMS EXCEPTION: {e}")
                    sim_valid = False
                    break
            
            if sim_valid:
                error = abs(sim_soc - self.target_soc)
                if error < min_error:
                    min_error = error
                    best_s = s
        
        # Restore
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        
        # Memory
        self.last_opt_s = best_s
        
        return best_s, self.target_soc, self.ratio
