import numpy as np
import pandas as pd

class HorizonPredictor:
    """
    Look-ahead module to extract future driving conditions.
    """
    def __init__(self, cycle_df, horizon_dist_m=2000.0, resample_step=10):
        """
        Args:
            cycle_df: DataFrame. Must contain mapped VECTO columns.
            horizon_dist_m: Look ahead distance [m]. (Default 2000m)
            resample_step: Optimization downsampling. (Default 10)
        """
        self.cycle_df = cycle_df.copy()
        
        # Ensure cumulative distance
        if 'dist_accum_m' not in self.cycle_df.columns:
            if 'dt' in self.cycle_df.columns:
                 dts = self.cycle_df['dt'].values
            else:
                 times = self.cycle_df['time'].values
                 dts = np.diff(times, prepend=times[0])
                 dts[0] = 0.5 # approx first
            
            v_mps = self.cycle_df['velocity_kmh'] / 3.6
            dist_step = v_mps * dts
            self.cycle_df['dist_accum_m'] = np.cumsum(dist_step)
            
        self.dist_arr = self.cycle_df['dist_accum_m'].values
        self.time_arr = self.cycle_df['time'].values
        
        # dt
        if 'dt' in self.cycle_df.columns:
            self.dt_arr = self.cycle_df['dt'].values
        else:
             times = self.cycle_df['time'].values
             self.dt_arr = np.diff(times, prepend=times[0])
             self.dt_arr[0] = 0.5
        
        # Physics Quantities
        self.vel_kmh_arr = self.cycle_df['velocity_kmh'].values
        self.rpm_arr = self.cycle_df['rpm_ice'].values
        self.treq_arr = self.cycle_df['t_req_hybrid_in'].values
        
        if 'grade_pct' in self.cycle_df.columns:
            self.grade_arr = self.cycle_df['grade_pct'].values / 100.0 # Convert % to decimal
        else:
            self.grade_arr = np.zeros_like(self.time_arr)
            
        if 'altitude_m' in self.cycle_df.columns:
            self.alt_arr = self.cycle_df['altitude_m'].values
        else:
            self.alt_arr = np.zeros_like(self.time_arr)
        
        self.horizon_dist = horizon_dist_m
        self.resample_step = resample_step
        self.N = len(cycle_df)

    def get_horizon(self, current_idx):
        """
        Returns vectors for the next horizon.
        """
        curr_dist = self.dist_arr[current_idx]
        target_dist = curr_dist + self.horizon_dist
        
        # Find index
        end_idx = np.searchsorted(self.dist_arr, target_dist)
        if end_idx >= self.N:
            end_idx = self.N - 1
            
        # Extract slice
        sl = slice(current_idx, end_idx, self.resample_step)
        
        return {
            'times': self.time_arr[sl],
            'dts': self.dt_arr[sl],
            'vel_kmh': self.vel_kmh_arr[sl],
            'grades': self.grade_arr[sl],
            'alts': self.alt_arr[sl],
            'rpms': self.rpm_arr[sl],
            't_reqs': self.treq_arr[sl],
            'dist_covered': self.dist_arr[end_idx] - curr_dist
        }
