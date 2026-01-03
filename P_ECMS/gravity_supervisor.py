import numpy as np

class GravitySupervisor:
    """
    Implements P-ECMS with Gravity Awareness (Altitude Look-Ahead).
    Optimized for Real-Time execution using Local Neighborhood Search.
    
    Target SOC = Base_Target + K_grav * (Alt_future_avg - Alt_current)
    """
    def __init__(self, vehicle, controller, q_max_as, target_soc=0.50, k_grav=2.5e-4):
        self.veh = vehicle
        self.controller = controller
        self.q_max_as = q_max_as
        self.target_soc = target_soc
        self.k_grav = k_grav
        
        # Optimization State Memory
        self.last_opt_s = self.controller.s_dis 
        
        # Pre-calculate Charge/Discharge Ratio
        if self.controller.s_dis != 0:
            self.ratio = self.controller.s_chg / self.controller.s_dis
        else:
            self.ratio = 0.95 
            
        # Optimization Parameters
        self.delta_s = 0.05
        self.s_min = 1.4
        self.s_max = 3.5

    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        dist_covered = horizon_data['dist_covered']
        
        # 1. Update Target SOC with Gravity Adjustment
        
        # Base
        soc_target = self.target_soc 
        
        # Gravity Adj
        if 'alts' in horizon_data and len(horizon_data['alts']) > 0:
            alts = horizon_data['alts']
            alt_curr = alts[0]
            alt_avg = np.mean(alts)
             # Logic from PECMS update (User confirmed "Same logic")
             # soc_adj = k * (avg - curr)
            soc_adj = self.k_grav * (alt_avg - alt_curr)
            soc_target += soc_adj
        
        # Clamp Target SOC
        soc_target = max(0.35, min(0.75, soc_target))
        
        # 2. Local Search Candidates
        candidates = [
            self.last_opt_s - 2 * self.delta_s,
            self.last_opt_s - 1 * self.delta_s,
            self.last_opt_s,
            self.last_opt_s + 1 * self.delta_s,
            self.last_opt_s + 2 * self.delta_s
        ]
        
        rpms = horizon_data['rpms']
        t_reqs = horizon_data['t_reqs']
        dts = horizon_data['dts']
        steps = len(rpms)
        
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
                    res = self.controller.decide_split(t_reqs[k], rpms[k], sim_soc)
                    p_chem = res[3]
                    u_oc = self.veh.get_ocv(sim_soc)
                    i_bat = p_chem / u_oc
                    d_soc = - (i_bat * dts[k]) / self.q_max_as
                    sim_soc += d_soc
                except Exception:
                    sim_valid = False
                    break
            
            if sim_valid:
                error = abs(sim_soc - soc_target)
                if error < min_error:
                    min_error = error
                    best_s = s
        
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        
        self.last_opt_s = best_s
        
        return best_s, soc_target, self.ratio
