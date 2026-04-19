import sys
import os
import numpy as np

# Add parent directory to path to import core modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ecms_controller import ECMS_Controller

class AECMS_Controller(ECMS_Controller):
    """
    Adaptive ECMS with Proportional Feedback Control.
    Base factors are pre-calculated.
    Real-time factors adapt to SOC deviation.
    """
    WILLANS_P_MAX_WATTS = 325000.0

    def _sample_engine_points_for_willans(self, n_rpm=24, n_torque=24):
        """Sample physically valid engine points for Willans fit up to 325 kW."""
        sample_rpms = np.linspace(600.0, 2200.0, n_rpm)
        point_sets = []

        for rpm in sample_rpms:
            _, t_eng_max = self.veh.get_eng_limits(float(rpm))
            if not np.isfinite(t_eng_max) or t_eng_max <= 0.0:
                continue

            omega = float(rpm) * 2.0 * np.pi / 60.0
            if omega <= 0.0:
                continue

            # Enforce both engine-torque envelope and rated power cap.
            t_pow_cap = self.WILLANS_P_MAX_WATTS / omega
            t_cap = min(float(t_eng_max), float(t_pow_cap))
            if t_cap <= 0.0:
                continue

            torques = np.linspace(0.0, t_cap, n_torque)
            rpms = np.full_like(torques, float(rpm), dtype=float)
            point_sets.append(np.column_stack((rpms, torques)))

        if not point_sets:
            return np.array([]), np.array([])

        pts = np.vstack(point_sets)
        fuel_rates_gs = self.veh.fuel_interp(pts)
        omega = pts[:, 0] * 2.0 * np.pi / 60.0
        p_mech_watts = pts[:, 1] * omega

        valid_mask = (
            (p_mech_watts > 0.0)
            & (~np.isnan(fuel_rates_gs))
            & (p_mech_watts <= self.WILLANS_P_MAX_WATTS + 1e-6)
        )
        return p_mech_watts[valid_mask], fuel_rates_gs[valid_mask]

    def __init__(self, vehicle_model, kp_dis=65.508475, kp_chg=10.000000, target_soc=0.50):
        # Initialize parent
        super().__init__(vehicle_model)
        
        self.kp_dis = kp_dis
        self.kp_chg = kp_chg
        self.target_soc = target_soc
        
        # --- Method 1: Willans Line (Marginal Cost) ---
        p_fit, f_fit = self._sample_engine_points_for_willans()
        
        if len(p_fit) > 10:
            slope, intercept = np.polyfit(p_fit, f_fit, 1)
            willans_k = slope
        else:
            intercept = 0.0
            willans_k = 0.00005
            
        q_lhv = 42700.0 # J/g
        eta_marg = 1.0 / (willans_k * q_lhv)
        
        # Efficiencies
        self.eta_mot_avg = 0.91 
        self.eta_inv_avg = 0.95
        self.eta_batt_avg = 0.97
        eta_elec_path = self.eta_mot_avg * self.eta_inv_avg * self.eta_batt_avg
        
        base_factor_dim = willans_k * q_lhv
        
        s_dis_willans = base_factor_dim / eta_elec_path
        s_chg_willans = base_factor_dim * eta_elec_path

        # Expose Willans calibration values for reporting/plotting.
        self.willans_k = willans_k
        self.willans_intercept = intercept
        self.s_dis_willans = s_dis_willans
        self.s_chg_willans = s_chg_willans
        self.willans_p_fit_watts = p_fit
        self.willans_f_fit_gs = f_fit
        self.willans_p_max_watts = self.WILLANS_P_MAX_WATTS
        
        # --- Method 2: Average Efficiency (Bulk Cost) ---
        # Previous manual tuned value
        eta_eng_avg_man = 0.4461
        
        s_dis_manual = 1.0 / (eta_eng_avg_man * eta_elec_path)
        s_chg_manual = eta_elec_path / eta_eng_avg_man
        
        # --- Final: Average of Both ---
        self.s_dis_0 = (s_dis_willans + s_dis_manual) / 2.0
        self.s_chg_0 = (s_chg_willans + s_chg_manual) / 2.0
        
        print(f"A-ECMS Calibration:")
        print(f"  Willans (Marginal): k={willans_k:.2e}, s_d={s_dis_willans:.3f}, s_c={s_chg_willans:.3f}")
        print(f"  Manual  (Average) : eta={eta_eng_avg_man:.3f}, s_d={s_dis_manual:.3f}, s_c={s_chg_manual:.3f}")
        print(f"  FINAL (Average)   : s_dis_0={self.s_dis_0:.4f}, s_chg_0={self.s_chg_0:.4f}")
        
    def decide_split(self, t_req, rpm, soc, v_kmh=None):
        """
        Adapts s_dis/s_chg and calls parent optimization.
        """
        # P-Control
        error = self.target_soc - soc
        
        # Adaptation
        # If SOC < Target (Error > 0): SOC is low.
        # s_dis should increase (penalize discharge)
        # s_chg should increase (incentivize charge)
        # Both move in the same direction with error, but scaled differently.
        
        self.s_dis = self.s_dis_0 + self.kp_dis * error
        self.s_chg = self.s_chg_0 + self.kp_chg * error
        
        # Clamp
        self.s_dis = max(0.5, min(5.0, self.s_dis))
        self.s_chg = max(0.5, min(5.0, self.s_chg))
        
        return super().decide_split(t_req, rpm, soc, v_kmh)
