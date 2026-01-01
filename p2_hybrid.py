import pandas as pd
import numpy as np
import os

class P2HybridTruck:
    def __init__(self, vecto_loader, chassis_params=None):
        self.loader = vecto_loader
        self.fuel_interp = None
        self.em_eff_interp = None
        self.em_params = {}
        self.bat_params = {}
        self.ocv_curve = None
        self.r_int_curve = None
        
        # Default Trans efficiency if not provided
        self.eta_trans = 0.96 
        
        # Fallback Resistance (Ohms) if file missing.
        # 120kWh battery ~ 600V -> 200Ah. R ~ 0.05 - 0.1 Ohm for pack?
        # VECTO usually uses mOhm in files? No, usually Ohm or just V/A.
        # Fixed Standard: 0.25 Ohm as per User Request.
        # Fixed Standard: 0.25 Ohm as per User Request.
        self.fallback_r_int = 0.25
        
        # Physics Parameters (Typical 40t LH Truck)
        self.M_veh = 40000.0 # [kg] (Gross Mass) or read from cycle?
        self.Cd = 0.6        # Check typical VECTO
        self.A_front = 10.0  # [m^2]
        self.C_roll = 0.006  # Rolling coeff
        self.Rho_air = 1.2   # Air density

    def load_components(self, engine_map_path, motor_map_path, motor_param_path, bat_param_path, bat_ocv_path, bat_res_path=None):
        """
        Loads all component maps and parameters.
        """
        self.fuel_interp = self.loader.read_vmap(engine_map_path)
        self.em_eff_interp = self.loader.read_vemo(motor_map_path)
        self.em_params = self.loader.read_vem(motor_param_path)
        self.bat_params = self.loader.read_vreess(bat_param_path)
        self.ocv_curve = self.loader.read_vbatv(bat_ocv_path)
        
        if bat_res_path and os.path.exists(bat_res_path):
            self.r_int_curve = self.loader.read_vbatr(bat_res_path)
        else:
            print("Internal Resistance file not found or not provided. Using fallback constant.")
            self.r_int_curve = None
        
        print("Components loaded successfully.")

    def get_ocv(self, soc):
        """
        Returns Open Circuit Voltage for a given SOC (0-1).
        """
        if self.ocv_curve is None:
            # Fallback if no curve loaded (e.g. testing)
            return 700.0 
            
        return float(self.ocv_curve([soc * 100.0])) # distinct correction: fraction -> %

    def calc_backward_physics(self, cycle_df):
        """
        Calculates the required torque at the hybrid input shaft.
        
        IMPROVED LOGIC: Use 'T_ice_fcmap [Nm]' from VECTO file if available.
        This provides the EXACT load the engine saw in the baseline, accounting for
        all Aux loads, Gearbox losses, Axle losses, and Inertias.
        
        Fallback: Calculate from Power Wheel (less accurate).
        """
        if cycle_df is None:
            return None
            
        # Check for VECTO columns
        if 'T_ice_fcmap [Nm]' in cycle_df.columns:
            print("Using exact Load from VECTO column: T_ice_fcmap [Nm]")
            t_req = cycle_df['T_ice_fcmap [Nm]'].values
            return t_req
            
        print("Exact VECTO Torque not found, calculating from Physics (Approximation)...")
        
        # Physics:
        omega = cycle_df['rpm_ice'] * 2 * np.pi / 60.0
        p_wheel_kw = cycle_df['power_wheel_kw'] 
        
        p_trans_in = np.where(p_wheel_kw >= 0, p_wheel_kw / self.eta_trans, p_wheel_kw * self.eta_trans)
        
        # Add Aux loads if found
        p_aux_col = next((c for c in cycle_df.columns if 'P_aux_mech' in c), None)
        if p_aux_col:
            p_trans_in += cycle_df[p_aux_col].values
        
        p_trans_in_watts = p_trans_in * 1000.0
        
        t_req = np.zeros_like(p_trans_in)
        mask = omega > 1.0 # rad/s
        t_req[mask] = p_trans_in_watts[mask] / omega[mask]
        
        return t_req

    def calc_battery_dynamics(self, p_mot_kw, p_aux_kw, soc, dt=1.0):
        """
        Calculates Battery Dynamics strictly according to Equations 11 and 13.
        
        Args:
            p_mot_kw: Electric motor power request [kW]. (Positive = consuming).
            p_aux_kw: Aux load [kW].
            soc: Current State of Charge [0..1].
            dt: Time step [s].
            
        Returns:
            i_bat: Battery current [A].
            p_chem_watts: Chemical power [W].
        """
        # Eq 11 Inputs
    def calc_battery_dynamics(self, p_elec_kw, dt, soc):
        """
        Calculates battery current and fuel equivalent cost.
        soc: fractional (0-1)
        """
        # SOC to %
        soc_pct = soc * 100.0

        # Convert to Watts
        p_bat_watts = p_elec_kw * 1000.0
        
        # STRICT SIGN CONVENTION logic from previous reasoning:
        # Eq 11 ((-U + sqrt)/2R) yields negative current for positive term inside sqrt.
        # Physics requires P < 0 for discharge for this formula to yield I > 0 (Discharge).
        # Standard VECTO P_mot > 0 is Discharge.
        # So we flip sign for the formula input.
        p_bat_eqn = -1.0 * p_bat_watts
        
        # 1. Get OCV and R_int
        u_bat_oc = self.ocv_curve([soc_pct]) # [V]
        
        if self.r_int_curve:
            r_bat = self.r_int_curve([soc_pct])
        else:
            r_bat = self.fallback_r_int # Uses 0.25 Ohm Standard
            
        # Discriminant
        # Term = u_oc^2 - 4 * r * P_bat_eqn
        discriminant = u_bat_oc**2 - 4 * r_bat * p_bat_eqn
        
        # Prepare Output
        if isinstance(discriminant, np.ndarray):
            i_bat = np.full_like(discriminant, np.nan)
            valid = discriminant >= 0
            
            # Eq 11: (-U + sqrt(D)) / 2R
            if np.any(valid):
                # Handle scalar R/U (0-d array or 1-element) vs Array Candidates
                r_val = r_bat
                if hasattr(r_bat, 'ndim') and r_bat.ndim > 0 and r_bat.shape == valid.shape:
                     r_val = r_bat[valid]
                elif hasattr(r_bat, 'item'): # Extract scalar
                     r_val = r_bat.item()
                
                u_val = u_bat_oc
                if hasattr(u_bat_oc, 'ndim') and u_bat_oc.ndim > 0 and u_bat_oc.shape == valid.shape:
                     u_val = u_bat_oc[valid]
                elif hasattr(u_bat_oc, 'item'):
                     u_val = u_bat_oc.item()
                
                # Calculate Current for VALID points
                # i = (-U + sqrt(D)) / 2R
                sqrt_d = np.sqrt(discriminant[valid])
                i_bat[valid] = (-u_val + sqrt_d) / (2 * r_val)
                
        else:
            # Scalar Case (if single p_elec passed)
            if discriminant >= 0:
                i_bat = (-u_bat_oc + np.sqrt(discriminant)) / (2 * r_bat)
            else:
                i_bat = np.nan
        
        # P_chem (Watts) = U_oc * I_bat
        p_chem_watts = u_bat_oc * i_bat
        
        return i_bat, p_chem_watts

    def get_limits(self, rpm):
        """
        Returns dynamic limits for MOTOR based on current params.
        """
        t_em_max = self.em_params.get('OverloadTorque', 1050) # found 1050 in file
        t_em_min = -t_em_max 
        return t_em_min, t_em_max

    def get_system_limits(self, rpm):
        """
        Returns dynamic limits for the HYBRID SYSTEM (Engine + Motor).
        Used to saturate t_req from VECTO before optimization.
        """
        t_mot_min, t_mot_max = self.get_limits(rpm)
        
        # Engine Limits (Approximate from 325kW.vmap analysis)
        # Drag ~ -235 Nm (from file check) to -300 Nm
        # Max ~ 2400 Nm (User Request)
        t_eng_min = -300.0
        t_eng_max = 2400.0 
        
        t_sys_min = t_eng_min + t_mot_min 
        t_sys_max = t_eng_max + t_mot_max
        
        return t_sys_min, t_sys_max
