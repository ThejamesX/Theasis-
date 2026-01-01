import numpy as np

class GravitySupervisor:
    """
    Implements P-ECMS with Gravity Awareness (Altitude Look-Ahead).
    Target SOC = Linear_Ref + K_grav * (Alt_future_avg - Alt_current)
    """
    def __init__(self, vehicle, controller, total_dist_m, q_max_as, start_soc=0.50, end_soc=0.50, k_grav=2.5e-4):
        self.veh = vehicle
        self.controller = controller
        self.total_dist = total_dist_m
        self.q_max_as = q_max_as
        self.start_soc = start_soc
        self.end_soc = end_soc
        self.k_grav = k_grav
        
        self.s_candidates = np.linspace(1.4, 3.5, 10) 

    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        dist_covered = horizon_data['dist_covered']
        
        # 1. Base Linear Target
        dist_future = current_dist + dist_covered
        if dist_future > self.total_dist: dist_future = self.total_dist
            
        slope = (self.start_soc - self.end_soc) / self.total_dist
        soc_target = self.start_soc - dist_future * slope
        
        # 2. Gravity Adjustment
        if 'alts' in horizon_data and len(horizon_data['alts']) > 0:
            alts = horizon_data['alts']
            alt_curr = alts[0]
            alt_avg = np.mean(alts)
            delta_alt = alt_curr - alt_avg
            
            soc_adj = self.k_grav * delta_alt
            soc_target -= soc_adj
        
        # Clamp
        soc_target = max(0.35, min(0.75, soc_target))
        
        # 3. Shooting Method
        rpms = horizon_data['rpms']
        t_reqs = horizon_data['t_reqs']
        dts = horizon_data['dts']
        steps = len(rpms)
        
        best_s = self.controller.s_dis
        min_error = float('inf')
        
        orig_s_dis = self.controller.s_dis
        orig_s_chg = self.controller.s_chg
        ratio = 1.9950 / 2.0886
        
        for s in self.s_candidates:
            sim_soc = current_soc
            self.controller.s_dis = s
            self.controller.s_chg = s * ratio
            
            for k in range(steps):
                try:
                    res = self.controller.decide_split(t_reqs[k], rpms[k], sim_soc)
                    p_chem = res[3]
                    u_oc = self.veh.get_ocv(sim_soc)
                    i_bat = p_chem / u_oc
                    d_soc = - (i_bat * dts[k]) / self.q_max_as
                    sim_soc += d_soc
                except:
                    sim_soc = -999
                    break
            
            error = abs(sim_soc - soc_target)
            if error < min_error:
                min_error = error
                best_s = s
        
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        
        return best_s, soc_target
