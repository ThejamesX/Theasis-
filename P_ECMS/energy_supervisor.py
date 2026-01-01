import numpy as np

class EnergyBalanceSupervisor:
    """
    Implements the Potential Energy Balance Reference Generator.
    Equation: Delta_SOC = K_gain * (m * g * Delta_h) / E_batt
    """
    def __init__(self, vehicle, controller, total_dist_m, q_max_as, start_soc=0.50, end_soc=0.50, k_gain=0.6):
        self.veh = vehicle
        self.controller = controller
        self.total_dist = total_dist_m
        self.q_max_as = q_max_as
        self.start_soc = start_soc
        self.end_soc = end_soc
        
        # Energy Capacity in Joules
        # Q_as = E_j / V_nom ? No.
        # Energy = Q_as * V_nom
        v_nom = self.veh.get_ocv(0.50)
        self.bat_energy_joules = q_max_as * v_nom
        print(f"DEBUG EnergySupervisor: V_nom={v_nom:.2f} V, E_batt={self.bat_energy_joules/3.6e6:.2f} kWh")
        
        # Physics
        self.M_veh = self.veh.M_veh 
        self.g = 9.81
        self.k_gain = k_gain # Tunable factor (0.5 - 0.7 safer)
        
        # Bounds
        self.soc_min = 0.35
        self.soc_max = 0.75
        
        self.s_candidates = np.linspace(1.5, 3.5, 10)

    def get_target_soc(self, current_soc, horizon_data):
        """
        Calculates SOC Target based on Potential Energy.
        """
        alts = horizon_data.get('alts', [])
        
        if len(alts) == 0:
            return self.start_soc
            
        # 1. Horizon Analysis
        curr_alt = alts[0]
        avg_future_alt = np.mean(alts)
        
        delta_h = avg_future_alt - curr_alt
        
        # 2. Physics Conversion
        # Delta_SOC = K * (m * g * delta_h) / E_batt
        # Potential Energy Delta [J]
        pe_delta = self.M_veh * self.g * delta_h
        
        soc_shift = self.k_gain * (pe_delta / self.bat_energy_joules)
        
        # 3. Target
        # Logic: 
        # If delta_h > 0 (Uphill ahead), avg > curr. pe_delta > 0.
        # soc_shift > 0. Target increases. (Pre-charge). Correct.
        # If delta_h < 0 (Downhill ahead). pe_delta < 0. Target drops. Correct.
        
        soc_target = self.start_soc + soc_shift
        
        # Debug large shifts
        if abs(soc_shift) > 0.05 and np.random.rand() < 0.01:
             print(f"DEBUG EnergyTarget: dH={delta_h:.1f}m, Shift={soc_shift:.4f}, Target={soc_target:.4f}")
        
        # Clamp
        soc_target = max(self.soc_min, min(self.soc_max, soc_target))
        
        return soc_target

    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        """
        Uses Shooting Method to hit the ENERGY-AWARE target.
        """
        # 1. Get Target
        target = self.get_target_soc(current_soc, horizon_data)
        
        # 2. Shooting Method (Same as before but targeting 'target')
        # ... logic reused from pecms_supervisor ...
        
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
                # We need to simulate battery dynamics approx
                # controller.decide_split...
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
        
        return best_s, target
