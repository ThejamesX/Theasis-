import numpy as np

class EnergyBalanceSupervisor:
    """
    Implements the Potential Energy Balance Reference Generator.
    Equation: Delta_SOC = K_gain * (m * g * Delta_h) / E_batt
    """
    def __init__(self, vehicle, controller, q_max_as, target_soc=0.50, k_gain=0.6):
        self.veh = vehicle
        self.controller = controller
        self.q_max_as = q_max_as
        self.target_soc = target_soc
        
        # Energy Capacity
        v_nom = self.veh.get_ocv(0.50)
        self.bat_energy_joules = q_max_as * v_nom
        
        # Physics
        self.M_veh = getattr(self.veh, 'M_veh', 40000.0)
        self.g = 9.81
        self.k_gain = k_gain 
        
        # Bounds
        self.soc_min = 0.35
        self.soc_max = 0.75
        
        self.s_candidates = np.linspace(1.5, 3.5, 10)

    def get_target_soc(self, current_soc, horizon_data):
        """
        Calculates SOC Target based on Potential Energy.
        """
        alts = horizon_data.get('alts', [])
        
        # Base
        soc_target = self.target_soc
        
        if len(alts) > 0:
            curr_alt = alts[0]
            avg_future_alt = np.mean(alts)
            
            delta_h = avg_future_alt - curr_alt
            
            # Physics Conversion
            pe_delta = self.M_veh * self.g * delta_h
            soc_shift = self.k_gain * (pe_delta / self.bat_energy_joules)
            
            soc_target += soc_shift
        
        # Clamp
        soc_target = max(self.soc_min, min(self.soc_max, soc_target))
        return soc_target

    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        target = self.get_target_soc(current_soc, horizon_data)
        
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
            
            error = abs(sim_soc - target)
            if error < min_error:
                min_error = error
                best_s = s
        
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        
        return best_s, target, ratio
