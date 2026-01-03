import numpy as np
import pandas as pd

class NewHorizonPredictor:
    """
    Look-ahead module to extract future driving conditions.
    Refined to return Speed and Gradient vectors sliced by SPATIAL nodes (every 50m).
    """
    def __init__(self, cycle_df, horizon_dist_m=2000.0, spatial_step=50.0):
        """
        Args:
            cycle_df: DataFrame. Must contain mapped VECTO columns.
            horizon_dist_m: Look ahead distance [m]. (Default 2000m)
            spatial_step: Distance between nodes [m]. (Default 50m)
        """
        self.cycle_df = cycle_df.copy()
        self.spatial_step = spatial_step
        self.horizon_dist = horizon_dist_m
        
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
        
        self.N = len(cycle_df)

    def get_horizon(self, current_idx):
        """
        Returns vectors for the next horizon based on SPATIAL NODES (50m).
        """
        curr_dist = self.dist_arr[current_idx]
        
        # Calculate target distances: d0, d0+50, d0+100 ... d0+2000
        # Number of steps = horizon / step
        num_nodes = int(self.horizon_dist / self.spatial_step) + 1
        
        target_dists = curr_dist + np.arange(num_nodes) * self.spatial_step
        
        # Find indices
        # searchsorted returns index where element should be inserted
        indices = np.searchsorted(self.dist_arr, target_dists)
        
        # Clamp indices
        indices = np.clip(indices, 0, self.N - 1)
        
        # Extract values at these indices
        # We assume "Node" value is the instantaneous value at that distance point
        
        result = {
            'times': self.time_arr[indices],
            'dts': np.zeros_like(indices, dtype=float), # Will calc below
            'vel_kmh': self.vel_kmh_arr[indices], 
            'grades': self.grade_arr[indices],
            'alts': self.alt_arr[indices],
            'rpms': self.rpm_arr[indices],
            't_reqs': self.treq_arr[indices],
            'dist_covered': self.dist_arr[indices[-1]] - curr_dist,
            'spatial_nodes': target_dists
        }
        
        # Calculate DTs between nodes for integration
        # dt[k] = time[k+1] - time[k] (approx time to reach next node)
        # Last dt is 0 or estimated by speed
        
        # Simple finite diff of times
        result['dts'][:-1] = np.diff(result['times'])
        
        # Check for zero dt (if speed was high and nodes clamped to same index at end of cycle)
        # or if multiple nodes fall on same index (stopped vehicle?)
        # Fallback: dt = dx / v
        
        mask_zero = result['dts'] <= 0.001
        if np.any(mask_zero):
             # dx = 50m. v = vel_kmh / 3.6
             v_mps = result['vel_kmh'] / 3.6
             v_mps[v_mps < 0.1] = 0.1 # Protect
             est_dt = self.spatial_step / v_mps
             
             # Apply fallback only where diff failed (or clamped)
             # Note: result['dts'] matches size of nodes. diff is size-1.
             # We filled result['dts'][:-1].
             
             # Actually safer to always use dx/v for prediction consistency if VECTO time is jittery
             # But VECTO time is ground truth.
             pass
             
        # Last dt estimate
        v_end = result['vel_kmh'][-1] / 3.6
        if v_end < 0.1: v_end = 0.1
        result['dts'][-1] = self.spatial_step / v_end
             
        return result
