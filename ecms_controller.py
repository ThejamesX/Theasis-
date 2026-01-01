import numpy as np

class ECMS_Controller:
    def __init__(self, vehicle_model, s_dis= 2.0886, s_chg= 1.9950, q_lhv=42700.0):
        self.veh = vehicle_model
        # Static Factors
        self.s_dis = s_dis
        self.s_chg = s_chg
        self.q_lhv_kj_g = q_lhv / 1000.0 # ~42.7
        
        # Pre-allocate candidates?
        # self.t_mot_candidates = np.linspace(-1000, 1000, 50) 
    
    def get_equivalence_factor(self, soc):
        return self.s_dis # Not used in static mode
        
    def decide_split(self, t_req, rpm, soc):
        """
        Optimization step minimizing Hamiltonian H (Eq 23).
        With Static Factors (self.s_dis, self.s_chg) and Torque Saturation.
        """
        if rpm < 600:
            pass


        # 1. Saturate t_req to Hybrid System Capability
        # Anything beyond min/max is Friction Brakes or unachievable
        t_sys_min, t_sys_max = self.veh.get_system_limits(rpm)
        
        # If t_req is very negative (-6000), t_req_hybrid becomes -1350.
        # The braking logic (Friction) handles the rest (-4650).
        t_req_hybrid = max(t_sys_min, min(t_sys_max, t_req))
        
        # Motor Candidates
        t_mot_min_phys, t_mot_max_phys = self.veh.get_limits(rpm)
        
        # SOC HARD LIMITS (20% - 80%)
        # If SOC < 0.20: No Discharge Allowed (t_mot <= 0)
        # If SOC > 0.80: No Charge Allowed (t_mot >= 0)
        
        # Apply SOC Constraints to Physical Limits
        if soc <= 0.20:
            t_mot_max_phys = min(0.0, t_mot_max_phys) # Clamp pos torque to 0
        if soc >= 0.80:
            t_mot_min_phys = max(0.0, t_mot_min_phys) # Clamp neg torque to 0
            
        # Engine Limits constraint code follows...
        # t_sys_min = t_eng_min + t_mot_min
        # We can deduce t_eng limits roughly or add a method. 
        # For now using consistent values:
        t_eng_max = 2400.0
        t_eng_min = -300
        
        # Constraint: t_eng_min <= (t_req - t_mot) <= t_eng_max
        # => t_mot <= t_req - t_eng_min
        # => t_mot >= t_req - t_eng_max
        
        t_mot_max_constr = t_req_hybrid - t_eng_min
        t_mot_min_constr = t_req_hybrid - t_eng_max
        
        # Intersect with Physical Motor Limits (which are now SOC constrained)
        t_mot_max = min(t_mot_max_phys, t_mot_max_constr)
        t_mot_min = max(t_mot_min_phys, t_mot_min_constr)
        
        if t_mot_min > t_mot_max:
             # Should not happen if t_req_hybrid was clamped to sys limits properly
            # Fallback to single point (best effort)
            candidates = np.array([t_req_hybrid - t_eng_max]) # Force engine max?
            # Re-check against physical (SOC constrained) limits
            c = candidates[0]
            if c > t_mot_max_phys: c = t_mot_max_phys
            if c < t_mot_min_phys: c = t_mot_min_phys
            candidates = np.array([c])
        else:
            candidates = np.linspace(t_mot_min, t_mot_max, 50)
        
        t_mots = candidates
        t_engs = t_req_hybrid - t_mots
        
        # 1. Fuel Power (Eq 21)
        rpms = np.full_like(t_engs, rpm)
        points_eng = np.column_stack((rpms, t_engs))
        fuel_rates_gs = self.veh.fuel_interp(points_eng) # g/s
        
        p_fuel_watts = fuel_rates_gs * (self.q_lhv_kj_g * 1000.0)
        
        # 2. Battery Chemical Power (P_chem)
        points_mot = np.column_stack((rpms, t_mots))
        p_elec_kw = self.veh.em_eff_interp(points_mot) # kW terminal
        
        # Calculate P_chem using strictly Eq 11, 13 physics
        _, p_chem_watts = self.veh.calc_battery_dynamics(p_elec_kw, 0.0, soc)
        
        # Filter NaNs (Battery Limits AND Engine Limits)
        valid_batt = ~np.isnan(p_chem_watts)
        valid_fuel = ~np.isnan(p_fuel_watts)
        valid_mask = valid_batt & valid_fuel
        
        h_costs = np.full_like(p_fuel_watts, np.inf)
        
        if np.any(valid_mask):
            # Static Equivalence Factors
            # Discharge (P_chem > 0) -> s_dis
            # Charge (P_chem < 0) -> s_chg
            
            p_c = p_chem_watts[valid_mask]
            # Vectorized selection of s based on sign of P_chem
            s_factors = np.where(p_c >= 0, self.s_dis, self.s_chg)
            
            # Eq 23: H = P_fuel + s * P_chem
            h = p_fuel_watts[valid_mask] + s_factors * p_c
            h_costs[valid_mask] = h
        
        # Min
        best_idx = np.argmin(h_costs)
        
        if np.isinf(h_costs[best_idx]):
            # Fallback
            idx_0 = np.argmin(np.abs(t_mots))
            fuel_fallback = float(self.veh.fuel_interp([[rpm, t_engs[idx_0]]]))
            p_fuel_fb = fuel_fallback * self.q_lhv_kj_g * 1000.0
            return t_engs[idx_0], t_mots[idx_0], p_fuel_fb, 0.0, fuel_fallback
            
        return t_engs[best_idx], t_mots[best_idx], h_costs[best_idx], p_chem_watts[best_idx], fuel_rates_gs[best_idx]
