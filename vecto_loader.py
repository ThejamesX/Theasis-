import json
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import os

class VectoLoader:
    def __init__(self):
        pass

    def read_vmod(self, file_path):
        """
        Reads the VECTO result file (.vmod).
        Identifies the header end by looking for the line starting with "time".
        """
        try:
            # VECTO vmod files often have a header. The data starts with "time".
            # We can find the header length or just let pandas try to find the header.
            # However, VECTO headers can be variable length. Padas 'header'='infer' might struggle if there are comment lines above.
            # A robust way: read lines until "time" is found.
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            header_row = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("time"):
                    header_row = i
                    break
            
            df = pd.read_csv(file_path, skiprows=header_row)
            
            # Clean column names (remove units in brackets if desired, but for now specific columns are requested)
            # The user requested: time, P_wheel_kW, n_eng_rpm, v_act_kmh
            # Column names in sample: "time [s]", "P_wheel_in [kW]", "n_ice_avg [1/min]", "v_act [km/h]"
            # NOTE: "P_wheel_in [kW]" is usually the load at the wheel hub. "P_trac [kW]" or "P_wheel_kW" might be used.
            # Let's map them to standard names.
            
            rename_map = {
                'time [s]': 'time',
                'dt [s]': 'dt',
                'v_act [km/h]': 'velocity_kmh',
                'n_ice_avg [1/min]': 'rpm_ice',
                'P_wheel_in [kW]': 'power_wheel_kw',
                'altitude [m]': 'altitude_m',
                'grad [%]': 'grade_pct'
            }
            
            # Try to rename, keep only matches
            df = df.rename(columns=rename_map)
            
            # Note: If n_ice_avg is not appropriate (e.g. engine off), we might need wheel speed.
            # But task says "Trust the P_wheel_kW and n_eng_rpm from the .vmod file".
            
            return df
        except Exception as e:
            print(f"Error reading .vmod file: {e}")
            return None

    def read_vmap(self, file_path):
        """
        Reads fuel consumption map (.vmap). Format: CSV
        columns: engine speed [rpm], torque [Nm], fuel consumption [g/h]
        Returns: Interpolator function f(rpm, torque) -> fuel_g_s
        """
        try:
            # .vmap usually csv with header
            df = pd.read_csv(file_path)
            # Check column names. Sample: "engine speed [rpm], torque [Nm], fuel consumption [g/h]"
            # Normalize names
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Extract columns based on keywords
            rpm_col = next(c for c in df.columns if 'speed' in c or 'rpm' in c)
            tq_col = next(c for c in df.columns if 'torque' in c)
            fuel_col = next(c for c in df.columns if 'fuel' in c)
            
            # Create interpolator. Since data is likely scattered or structured grid, 
            # LinearNDInterpolator is safe for scattered, but if it is grid, RegularGrid is faster.
            # VECTO maps are usually regular grids. Let's check unique values.
            
            rpms = df[rpm_col].unique()
            torques = df[tq_col].unique()
            
            # If it's a perfect grid
            if len(df) == len(rpms) * len(torques):
                print(f"DEBUG: Using RegularGridInterpolator (Grid Size: {len(rpms)}x{len(torques)})")
                df = df.sort_values(by=[rpm_col, tq_col])
                fuel_grid = df[fuel_col].values.reshape(len(rpms), len(torques))
                fuel_grid = fuel_grid / 3600.0 # g/h -> g/s
                interp = RegularGridInterpolator((np.sort(rpms), np.sort(torques)), fuel_grid, bounds_error=False, fill_value=None)
            else:
                print(f"DEBUG: Using Robust Linear+Nearest Interpolator (Data Len: {len(df)})")
                from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
                points = list(zip(df[rpm_col], df[tq_col]))
                values = df[fuel_col]
                values = values / 3600.0 # g/h -> g/s
                
                lin_interp = LinearNDInterpolator(points, values, fill_value=np.nan)
                near_interp = NearestNDInterpolator(points, values)
                
                # Wrapper to use Linear, then fill NaNs with Nearest
                def robust_interp(query_points):
                    res = lin_interp(query_points)
                    if np.any(np.isnan(res)):
                        # If scalar, and nan, replace
                        if np.isscalar(res) and np.isnan(res):
                            return near_interp(query_points)
                        # If array, replace mask
                        mask = np.isnan(res)
                        if np.any(mask):
                            # Ensure query_points is array matching shape
                            q_arr = np.array(query_points)
                            # Handle scalar query case wrapped in list
                            if q_arr.ndim == 1 and len(q_arr) == 2 and not np.isscalar(res): 
                                # Single point query returning 1-element array?
                                # LinearNDInterpolator behavior depends on input
                                pass 
                            
                            # Simple approach: predict nearest for ALL, then fill
                            # Optimized: only for mask
                            # query_points might be (N, 2)
                            if q_arr.ndim == 2:
                                res[mask] = near_interp(q_arr[mask])
                            else:
                                # Fallback for complex shapes or single-point list input [[x,y]]
                                res_near = near_interp(query_points)
                                res[mask] = res_near[mask]
                    return res
                    
                interp = robust_interp
                
            return interp
            
        except Exception as e:
            print(f"Error reading .vmap: {e}")
            return None

    def read_vemo(self, file_path):
        """
        Reads Electric Motor map (.vemo).
        Sample format:
        n [rpm] , T [Nm] , P_el [kW]
        """
        try:
            df = pd.read_csv(file_path)
            df.columns = [c.lower().strip() for c in df.columns]
            
            rpm_col = next(c for c in df.columns if 'n' in c or 'rpm' in c)
            tq_col = next(c for c in df.columns if 't' in c and 'nm' in c) # 'T [Nm]'
            p_col = next(c for c in df.columns if 'p_el' in c)
            
            points = list(zip(df[rpm_col], df[tq_col]))
            values = df[p_col] # kW
            
            from scipy.interpolate import LinearNDInterpolator
            # Fill value? If dragging motor (0 torque), power is loss.
            # We can let it return NaN and handle in controller or fill with nearest.
            # Using LinearNDInterpolator
            interp = LinearNDInterpolator(points, values, fill_value=np.nan) 
            
            return interp
        except Exception as e:
            print(f"Error reading .vemo: {e}")
            return None

    def read_vem(self, file_path):
        """
        Reads Motor Parameters (.vem JSON).
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data['Body'] # Return the Body dict containing inertia, max torque etc
        except Exception as e:
            print(f"Error reading .vem: {e}")
            return None

    def read_vreess(self, file_path):
        """
        Reads Battery Parameters (.vreess JSON) and potentially linked files if needed.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data['Body']
        except Exception as e:
            print(f"Error reading .vreess: {e}")
            return None
    
    def read_vbatv(self, file_path):
        """
        Reads OCV curve (.vbatv CSV).
        Sample: SOC, V
        """
        try:
            # Use sep=None which tries to sniff, or just ',' as requested.
            # skipinitialspace=True helps with spaces after commas provided in user screenshot "SOC, V"
            df = pd.read_csv(file_path, sep=',', skipinitialspace=True)
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Identify columns by keywords
            soc_col = next((c for c in df.columns if 'soc' in c or '%' in c), df.columns[0])
            v_col = next((c for c in df.columns if 'v' in c or 'volt' in c), df.columns[1])
            
            # Returns interpolator f(soc) -> ocv
            return interp1d(df[soc_col], df[v_col], kind='linear', fill_value="extrapolate")
        except Exception as e:
            print(f"Error reading .vbatv: {e}")
            return None

    def read_vbatr(self, file_path):
        """
        Reads Internal Resistance curve (.vbatr CSV).
        Sample: SOC, R [Ohm]
        """
        try:
            df = pd.read_csv(file_path, sep=',', skipinitialspace=True)
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Identify columns
            soc_col = next((c for c in df.columns if 'soc' in c or '%' in c), df.columns[0])
            r_col = next((c for c in df.columns if 'r' in c or 'ohm' in c or 'res' in c), df.columns[1])
            
            # Returns interpolator f(soc) -> r_int
            return interp1d(df[soc_col], df[r_col], kind='linear', fill_value="extrapolate")
        except Exception as e:
            print(f"Error reading .vbatr: {e}")
            return None
