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
        Vrací vektory pro horizont. 
        MODIFIKACE: Používá FIXNÍ RPM (aktuální hodnotu) pro celý horizont.
        """
        curr_dist = self.dist_arr[current_idx]
        
        # 1. Výpočet cílových vzdáleností (Nodes)
        num_nodes = int(self.horizon_dist / self.spatial_step) + 1
        target_dists = curr_dist + np.arange(num_nodes) * self.spatial_step
        
        boundary_indices = np.searchsorted(self.dist_arr, target_dists)
        boundary_indices = np.clip(boundary_indices, 0, self.N - 1)
        
        # Přečtení AKTUÁLNÍCH otáček (na začátku horizontu)
        current_rpm = self.rpm_arr[current_idx] 
        
        # Inicializace polí
        res_vel = np.zeros(num_nodes)
        res_grade = np.zeros(num_nodes)
        
        # ZDE JE ZMĚNA: Místo nul vytvoříme pole plné aktuální hodnoty RPM
        res_rpm = np.full(num_nodes, current_rpm) 
        
        res_treq = np.zeros(num_nodes)
        
        # Ostatní stavové veličiny
        res_alt = self.alt_arr[boundary_indices]
        res_time = self.time_arr[boundary_indices]
        
        # 4. Smyčka pro průměrování (ostatní veličiny jako Grade/Speed se stále průměrují)
        for k in range(num_nodes):
            idx_start = boundary_indices[k]
            if k < num_nodes - 1:
                idx_end = boundary_indices[k+1]
            else:
                idx_end = min(idx_start + 1, self.N - 1)
            
            if idx_end <= idx_start:
                res_vel[k] = self.vel_kmh_arr[idx_start]
                res_grade[k] = self.grade_arr[idx_start]
                res_treq[k] = self.treq_arr[idx_start]
            else:
                res_vel[k] = np.mean(self.vel_kmh_arr[idx_start:idx_end])
                res_grade[k] = np.mean(self.grade_arr[idx_start:idx_end])
                res_treq[k] = np.mean(self.treq_arr[idx_start:idx_end])
            
            # RPM zde už neřešíme, je nastaveno fixně nahoře
            # (Případně můžete zde pro jistotu přepsat: res_rpm[k] = current_rpm)

        # 5. Sestavení výsledku
        result = {
            'times': res_time,
            'dts': np.zeros(num_nodes, dtype=float),
            'vel_kmh': res_vel,
            'grades': res_grade,
            'alts': res_alt,
            'rpms': res_rpm,    # <-- Toto pole nyní obsahuje samé konstanty
            't_reqs': res_treq,
            'dist_covered': self.dist_arr[boundary_indices[-1]] - curr_dist,
            'spatial_nodes': target_dists,
            'remaining_distance': self.dist_arr[-1] - curr_dist
        }
        
        # ... (zbytek funkce pro výpočet dts zůstává stejný)
        result['dts'][:-1] = np.diff(result['times'])
        v_end = result['vel_kmh'][-1] / 3.6
        if v_end < 0.1: v_end = 0.1
        result['dts'][-1] = self.spatial_step / v_end

        return result