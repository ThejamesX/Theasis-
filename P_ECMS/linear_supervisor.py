import numpy as np

class LinearSupervisor:
    """
    Implements normal P-ECMS with Constant Reference.
    (Previously Linear Reference, but Linear calculation removed as per user request).
    """
    def __init__(self, vehicle, controller, q_max_as, target_soc=0.50):
        self.veh = vehicle
        self.controller = controller
        self.q_max_as = q_max_as
        self.target_soc = target_soc
        
        # Candidate range
        self.s_candidates = np.linspace(1.4, 3.5, 10) 

    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        """
        Determines the optimal s to hit Constant Reference.
        """
        soc_target = self.target_soc
        
        # Clamp target (safety bounds)
        soc_target = max(0.35, min(0.75, soc_target))
        
        # 2. Shooting Method
        rpms = horizon_data['rpms']
        t_reqs = horizon_data['t_reqs'] # Use vector from horizon (ignoring physics calc for Linear) or should I use physics? 
        # "remove linear calculation from prediction files" implies just the target.
        # LinearSupervisor traditionally relies on VECTO signals. I will keep it as is.
        dts = horizon_data['dts']
        steps = len(rpms)
        
        best_s = self.controller.s_dis
        min_error = float('inf')
        
        orig_s_dis = self.controller.s_dis
        orig_s_chg = self.controller.s_chg
        
        # Ratio logic needs update if using ratio
        # LinearSupervisor (old) used ratio=1.995/2.0886 hardcoded?
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
        
        return best_s, soc_target, ratio
