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
        self.fld_drive_interp = None
        self.fld_drag_interp = None
        self.ice_max_interp = None

    def load_components(self, engine_map_path, motor_map_path, motor_param_path, bat_param_path, bat_ocv_path, bat_res_path=None):
        """
        Loads all component maps and parameters.
        """
        import pandas as pd
        import numpy as np
        from scipy.interpolate import interp1d
        
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
            
        # Load Electric Motor Limits (EM_fld.vemp)
        em_fld_path = os.path.join(os.path.dirname(motor_map_path), "EM_fld.vemp")
        if os.path.exists(em_fld_path):
            fld_df = pd.read_csv(em_fld_path, skipinitialspace=True)
            fld_rpm = fld_df.iloc[:, 0].values
            fld_tq_drive = fld_df.iloc[:, 1].values
            fld_tq_drag = fld_df.iloc[:, 2].values
            self.fld_drive_interp = interp1d(fld_rpm, fld_tq_drive, kind='linear', bounds_error=False, fill_value=(fld_tq_drive[0], 0.0))
            self.fld_drag_interp = interp1d(fld_rpm, fld_tq_drag, kind='linear', bounds_error=False, fill_value=(fld_tq_drag[0], 0.0))
            
        # Load ICE Limits from VMAP
        try:
            df_vmap = pd.read_csv(engine_map_path, sep=',', header=0)
            df_vmap.columns = ['rpm', 'torque', 'fuel']
            rpm_raw = df_vmap['rpm'].values
            torque_raw = df_vmap['torque'].values
            drpm = np.diff(rpm_raw)
            block_ends = np.where(np.abs(drpm) > 5.0)[0]
            block_ends = np.append(block_ends, len(rpm_raw)-1)
            max_tq_curve = []
            for i, end_idx in enumerate(block_ends):
                start_idx = 0 if i == 0 else block_ends[i-1] + 1
                rp_block = rpm_raw[start_idx:end_idx+1]
                tq_block = torque_raw[start_idx:end_idx+1]
                if tq_block.max() > 0:
                    max_tq_curve.append((rp_block[np.argmax(tq_block)], tq_block.max()))
            max_tq_curve = np.array(max_tq_curve)
            self.ice_max_interp = interp1d(max_tq_curve[:,0], max_tq_curve[:,1], kind='linear', bounds_error=False, fill_value=(max_tq_curve[0,1], 0.0))
        except Exception as e:
            print("Failed to load ICE Max Torque from VMAP:", e)
        
        print("Components loaded successfully.")

    def get_ocv(self, soc):
        """
        Returns Open Circuit Voltage for a given SOC (0-1).
        """
        if self.ocv_curve is None:
            # Fallback if no curve loaded (e.g. testing)
            return 700.0 
            
        # Ensure input is scalar
        if hasattr(soc, "item"): 
            soc_val = soc.item()
        else:
            soc_val = soc
            
        res = self.ocv_curve(soc_val * 100.0) # interp1d handles scalar
        
        # Ensure output is native float
        if hasattr(res, "item"):
            return res.item()
        return float(res)

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
        discriminant = u_bat_oc**2 - 4 * r_bat * p_bat_watts
        
        # Prepare Output
        if isinstance(discriminant, np.ndarray):
            i_bat = np.full_like(discriminant, np.nan)
            valid = discriminant >= 0
            
            # Eq 11: (U - sqrt(D)) / 2R
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
                sqrt_d = np.sqrt(discriminant[valid])
                i_bat[valid] = (u_val - sqrt_d) / (2 * r_val)
                
        else:
            # Scalar Case (if single p_elec passed)
            if discriminant >= 0:
                i_bat = (u_bat_oc - np.sqrt(discriminant)) / (2 * r_bat)
            else:
                i_bat = np.nan
        
        # P_chem (Watts) = U_oc * I_bat
        p_chem_watts = u_bat_oc * i_bat
        
        return i_bat, p_chem_watts

    def get_limits(self, rpm):
        """
        Returns dynamic limits for MOTOR based on current params and EM_fld.
        """
        if self.fld_drive_interp is not None and self.fld_drag_interp is not None:
            t_em_max = float(self.fld_drive_interp(rpm))
            t_em_min = float(self.fld_drag_interp(rpm))
        else:
            t_em_max = self.em_params.get('OverloadTorque', 1050)
            t_em_min = -t_em_max 
        return t_em_min, t_em_max

    def get_eng_limits(self, rpm):
        """
        Returns dynamic limits for Engine based on current params and 325kW.vmap.
        """
        t_eng_min = -300.0  # Typical drag
        if self.ice_max_interp is not None:
            t_eng_max = float(self.ice_max_interp(rpm))
        else:
            t_eng_max = 2400.0 
        return t_eng_min, t_eng_max

    def get_system_limits(self, rpm):
        """
        Returns dynamic limits for the HYBRID SYSTEM (Engine + Motor).
        Used to saturate t_req from VECTO before optimization.
        """
        t_mot_min, t_mot_max = self.get_limits(rpm)
        t_eng_min, t_eng_max = self.get_eng_limits(rpm)
        
        t_sys_min = t_eng_min + t_mot_min 
        t_sys_max = t_eng_max + t_mot_max
        
        return t_sys_min, t_sys_max
