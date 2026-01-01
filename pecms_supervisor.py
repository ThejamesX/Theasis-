import numpy as np
import pandas as pd

class HorizonPredictor:
    """
    Look-ahead module to extract future driving conditions.
    """
    def __init__(self, cycle_df, horizon_dist_m=2000.0, resample_step=10):
        """
        Args:
            cycle_df: DataFrame with 'dist_m', 'time', 'rpm_ice', 't_req_hybrid_in'.
                      Must have cumulative distance column 'dist_accum_m' or we calc it.
            horizon_dist_m: Look ahead distance [m].
            resample_step: Resample step for optimization (e.g. 10 means every 10th row).
                           VECTO is 1Hz? Then every 10s.
        """
        self.cycle_df = cycle_df.copy()
        
        # Ensure cumulative distance exists
        if 'dist_accum_m' not in self.cycle_df.columns:
            # Approx distance from speed? Or use VECTO if available
            # Speed [km/h] / 3.6 * dt = dist [m]
            # vecto_loader renames to 'velocity_kmh'
            # Use 'dt' from file if available, else calc
            if 'dt' in self.cycle_df.columns:
                 dts = self.cycle_df['dt'].values
            else:
                 times = self.cycle_df['time'].values
                 dts = np.diff(times, prepend=times[0])
                 dts[0] = 0.5
            
            v_mps = self.cycle_df['velocity_kmh'] / 3.6
            dist_step = v_mps * dts
            self.cycle_df['dist_accum_m'] = np.cumsum(dist_step)
            
        self.dist_arr = self.cycle_df['dist_accum_m'].values
        self.time_arr = self.cycle_df['time'].values
        # Store dt array for horizon extraction
        if 'dt' in self.cycle_df.columns:
            self.dt_arr = self.cycle_df['dt'].values
        else:
             times = self.cycle_df['time'].values
             self.dt_arr = np.diff(times, prepend=times[0])
             self.dt_arr[0] = 0.5
             
        self.rpm_arr = self.cycle_df['rpm_ice'].values
        self.treq_arr = self.cycle_df['t_req_hybrid_in'].values
        if 'altitude_m' in self.cycle_df.columns:
            self.alt_arr = self.cycle_df['altitude_m'].values
        else:
            self.alt_arr = np.zeros_like(self.time_arr)
        
        self.horizon_dist = horizon_dist_m
        self.resample_step = resample_step
        self.N = len(cycle_df)

    def get_horizon(self, current_idx):
        """
        Returns vectors for the next 2km: t, dt, rpm, t_req.
        """
        curr_dist = self.dist_arr[current_idx]
        target_dist = curr_dist + self.horizon_dist
        
        # Find index where dist >= target_dist
        # Search sorted is fast
        end_idx = np.searchsorted(self.dist_arr, target_dist)
        
        # Clamp to end of cycle
        if end_idx >= self.N:
            end_idx = self.N - 1
            
        # Extract slice
        # Use stepping for speed
        sl = slice(current_idx, end_idx, self.resample_step)
        
        ts = self.time_arr[sl]
        rpms = self.rpm_arr[sl]
        t_reqs = self.treq_arr[sl]
        dts = self.dt_arr[sl]
        alts = self.alt_arr[sl]
        
        # Check lengths (resampling might drop last dt if not careful?)
        # Array slicing should be consistent.
            
        return {
            'times': ts,
            'dts': dts,
            'rpms': rpms,
            't_reqs': t_reqs,
            'alts': alts,
            'dist_covered': self.dist_arr[end_idx] - curr_dist
        }

class PECMS_Supervisor:
    """
    Implements the Shooting Method supervisory control with Gravity Awareness.
    """
    def __init__(self, vehicle, controller, total_dist_m, q_max_as, start_soc=0.50, end_soc=0.50, k_grav=0.0):
        self.veh = vehicle
        self.controller = controller # Instance of ECMS_Controller
        self.total_dist = total_dist_m
        self.start_soc = start_soc
        self.end_soc = end_soc
        self.q_max_as = q_max_as
        self.k_grav = k_grav # Gravity gain
        
        # Candidate range
        # Centered around typical efficient values
        self.s_candidates = np.linspace(1.4, 3.5, 10) 
        
    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        """
        Determines the optimal s that hits the Reference SOC at the end of the horizon.
        """
        dist_covered = horizon_data['dist_covered']
        
        # Helper to calc linear target
        def get_linear_target(d):
            if d > self.total_dist: d = self.total_dist
            slope = (self.start_soc - self.end_soc) / self.total_dist
            return self.start_soc - d * slope

        if dist_covered < 10.0:
            return self.controller.s_dis, get_linear_target(current_dist) # No horizon
            
        # 1. Calc Target SOC (Linear Depletion)
        # SOC_ref(d) = SOC_start - d * (SOC_start - SOC_end) / D_total
        # We want target AT END of horizon
        dist_future = current_dist + dist_covered

        if dist_future > self.total_dist:
            dist_future = self.total_dist
        
        slope = (self.start_soc - self.end_soc) / self.total_dist
        soc_target = self.start_soc - dist_future * slope
        
        # --- Gravity Consideration ---
        # SOC_target(t) = SOC_nominal + K_grav * (Alt_future_avg - Alt_current)
        if 'alts' in horizon_data and len(horizon_data['alts']) > 0:
            alts = horizon_data['alts']
            alt_curr = alts[0]
            alt_avg = np.mean(alts)
            
            delta_alt = alt_avg - alt_curr
            # K_grav units: SOC per Meter? 
            # If K_grav=0.0001. Delta=100m. Adj=0.01 (1%).
            
            soc_adj = self.k_grav * delta_alt
            soc_target += soc_adj
            
        # Clamp target (safety bounds)
        soc_target = max(0.35, min(0.75, soc_target))
        
        rpms = horizon_data['rpms']
        t_reqs = horizon_data['t_reqs']
        dts = horizon_data['dts']
        steps = len(rpms)
        
        best_s = self.controller.s_dis
        min_error = float('inf')
        
        # Store original factors to restore later
        orig_s_dis = self.controller.s_dis
        orig_s_chg = self.controller.s_chg
        
        # Ratio
        ratio = 1.9950 / 2.0886
        
        # 2. Shooting Method
        for s in self.s_candidates:
            sim_soc = current_soc
            
            # Set virtual s
            self.controller.s_dis = s
            self.controller.s_chg = s * ratio
            
            # Fast Simulation
            for k in range(steps):
                # Optimization step
                t_req = t_reqs[k]
                rpm = rpms[k]
                dt = dts[k]
                
                # decide_split returns: t_eng, t_mot, cost, p_chem, fuel_rate
                # Warning: decide_split logic might clamp SOC if we used real SOC limits.
                # Here we strictly trust the result P_chem.
                
                try:
                    res = self.controller.decide_split(t_req, rpm, sim_soc)
                    p_chem = res[3] # index 3
                    
                    # Update SOC
                    # I = P_chem / OCV
                    u_oc = self.veh.get_ocv(sim_soc)
                    i_bat = p_chem / u_oc
                    
                    # dSOC
                    d_soc = - (i_bat * dt) / self.q_max_as
                    sim_soc += d_soc
                    
                except Exception:
                    # Infeasible?
                    sim_soc = -999
                    break
            
            # Evaluate
            error = abs(sim_soc - soc_target)
            if error < min_error:
                min_error = error
                best_s = s
                
        # Restore controller state
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        
        return best_s, soc_target
