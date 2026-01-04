from pecms_supervisor import PECMS_Supervisor

class APECMS_Supervisor(PECMS_Supervisor):
    """
    A-P-ECMS: Adaptive P-ECMS.
    Combines:
    1. Predictive Target SOC (from P-ECMS)
    2. Adaptive Feedback Control (from A-ECMS) - Replaces Local Search
    """
    def __init__(self, vehicle, controller, q_max_as, target_soc=0.50, kp_dis=16.5, kp_chg=2.0):
        # Initialize parent (P-ECMS) to get physics & logic methods
        super().__init__(vehicle, controller, q_max_as, target_soc)
        
        self.kp_dis = kp_dis
        self.kp_chg = kp_chg
        
    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        # 1. Predictive Target SOC Calculation (Same as P-ECMS)
        calculate_horizon_energy_delta = self.calculate_horizon_energy_delta(horizon_data)
        soc_adj = 0.25 * calculate_horizon_energy_delta
        
        # Note: If P-ECMS has landing logic, we should probably have it here too?
        # User requested "Predicted Target SOC Generation".
        # I will check if 'total_dist_m' is available in self (it was added then removed?)
        # For now, I stick to the pure prediction logic currently in P-ECMS.
        
        soc_target = self.soc_nominal + soc_adj
        soc_target = max(0.35, min(0.75, soc_target))
        
        # 2. Adaptive Control (Feedback)
        # Replaces local search loop
        
        error = soc_target - current_soc
        
        # Formula: s = s0 + Kp * error
        # High SOC (Error < 0) -> Lower s -> Discharge
        # Low SOC (Error > 0) -> Higher s -> Charge/Save
        
        s_dis_new = self.s_dis_0 + self.kp_dis * error
        s_chg_new = self.s_chg_0 + self.kp_chg * error
        
        # Clamping (Stability)
        s_dis_new = max(1.0, min(3.5, s_dis_new))
        s_chg_new = max(1.0, min(3.5, s_chg_new))
        
        # 3. Output
        # Supervisor interface expects: best_s, soc_target, ratio
        # Main.py uses:
        # controller.s_dis = best_s
        # controller.s_chg = best_s * ratio
        
        ratio_new = s_chg_new / s_dis_new
        
        return s_dis_new, soc_target, ratio_new
