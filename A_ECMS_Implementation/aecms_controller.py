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
    def __init__(self, vehicle_model, kp_dis=10, kp_chg=10.1, target_soc=0.50):
        # Initialize parent
        super().__init__(vehicle_model)
        
        self.kp_dis = kp_dis
        self.kp_chg = kp_chg
        self.target_soc = target_soc
        
        # --- Method 1: Willans Line (Marginal Cost) ---
        # Sample Engine Map
        sample_rpms = np.linspace(600, 2200, 20)
        sample_torques = np.linspace(0, 2500, 20)
        rr, tt = np.meshgrid(sample_rpms, sample_torques)
        pts = np.column_stack((rr.flatten(), tt.flatten()))
        
        fuel_rates_gs = self.veh.fuel_interp(pts)
        omega = pts[:, 0] * 2 * np.pi / 60.0
        p_mech_watts = pts[:, 1] * omega
        
        valid_mask = (p_mech_watts > 0) & (~np.isnan(fuel_rates_gs))
        p_fit = p_mech_watts[valid_mask]
        f_fit = fuel_rates_gs[valid_mask]
        
        if len(p_fit) > 10:
            slope, intercept = np.polyfit(p_fit, f_fit, 1)
            willans_k = slope
        else:
            willans_k = 0.00005
            
        q_lhv = 42700.0 # J/g
        eta_marg = 1.0 / (willans_k * q_lhv)
        
        # Efficiencies
        self.eta_mot_avg = 0.93 
        self.eta_inv_avg = 0.95
        self.eta_batt_avg = 0.98 
        eta_elec_path = self.eta_mot_avg * self.eta_inv_avg * self.eta_batt_avg
        
        base_factor_dim = willans_k * q_lhv
        
        s_dis_willans = base_factor_dim / eta_elec_path
        s_chg_willans = base_factor_dim * eta_elec_path
        
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
        
    def decide_split(self, t_req, rpm, soc):
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
        
        return super().decide_split(t_req, rpm, soc)
