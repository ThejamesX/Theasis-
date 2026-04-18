import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

class DPOptimizer:
    def __init__(self, truck, cycle_df, soc_grid_size=250, bat_capacity_kwh =120.0):
        self.truck = truck
        self.cycle_df = cycle_df
        
        # Grid Setup
        self.soc_min = 0.3
        self.soc_max = 0.7
        self.soc_grid = np.linspace(self.soc_min, self.soc_max, soc_grid_size)
        self.ns = soc_grid_size
        self.N = len(cycle_df)
        self.bat_capacity_kwh = bat_capacity_kwh
        
        # Time Steps
        times = cycle_df['time'].values
        self.dts = np.diff(times, prepend=times[0])
        self.dts[0] = 0.5 # Fix first step
        
        # Physics inputs
        if 'T_ice_fcmap [Nm]' in cycle_df.columns:
            self.t_reqs = cycle_df['T_ice_fcmap [Nm]'].values
        else:
            self.t_reqs = truck.calc_backward_physics(cycle_df)
            
        self.rpms = cycle_df['rpm_ice'].values
        
        # Create Vectorized Maps for Speed
        self._prepare_maps()
        
    def _prepare_maps(self):
        """
        Creates efficient interpolators and inverse maps.
        """
        print("Pre-computing Inverse Motor Map...")
        # Motor Map: (n, T) -> P_el
        # We need: (n, P_el) -> T
        # Grid range based on expected operation
        n_range = np.linspace(0, 4000, 50)
        t_range = np.linspace(-1500, 1500, 100)
        
        N_grid, T_grid = np.meshgrid(n_range, t_range, indexing='ij')
        
        # Evaluate P_el for this grid using truck's interpolator
        # truck.em_eff_interp takes (n, t)
        pts = np.column_stack((N_grid.ravel(), T_grid.ravel()))
        P_el_flat = self.truck.em_eff_interp(pts)
        P_el_grid = P_el_flat.reshape(N_grid.shape)
        
        # Inverse: For each n slice, P_el is monotonic-ish? 
        # Actually P_el ~ T * w. It is monotonic with T for fixed w.
        # So we can interp T(P) for each n.
        
        # However, LinearNDInterpolator is slow to construct every time.
        # Let's try to make a 2D interpolator (n, P) -> T.
        # Points: (n, p_el) -> value: t
        
        valid_mask = ~np.isnan(P_el_flat)
        inv_pts = np.column_stack((pts[valid_mask, 0], P_el_flat[valid_mask]))
        inv_vals = pts[valid_mask, 1]
        
        # This might be heavy? LinearND for inverse.
        # Alternative: We solve T_mot = P_elec / (w * eta) dynamically?
        # No, map is cleaner.
        self.inv_mot_interp = LinearNDInterpolator(inv_pts, inv_vals, fill_value=np.nan)
        print("Inverse Map Ready.")

    def solve(self, start_soc=0.70, target_soc=0.30):
        print("Starting DP Backward Sweep (Control Discretization)...")
        start_time = time.time()
        self._infeasible_rows_total = 0

        # Persist target configuration for reconstruction/output logic.
        self.dp_target_soc = target_soc
        
        # Grid Setup
        # SOC Grid (States)
        self.J_next = np.full(self.ns, np.inf)
        
        # Terminal cost can be disabled by passing target_soc=None.
        if target_soc is None:
            self.J_next = np.zeros(self.ns)
        else:
            penalty_weight = 1e6
            self.J_next = penalty_weight * (self.soc_grid - target_soc)**2
        
        # Control Variable Grid (T_mot)
        # We'll adapt it dynamically or use fixed grid? 
        # Fixed grid is easier for vectorization.
        self.u_control_grid = np.linspace(-1500, 1500, 201) # T_mot candidates
        self.nu = len(self.u_control_grid)
        
        # Storage for Optimal Control (We store the optimal T_mot index or value?)
        # Storing value is easier, or index into u_grid.
        # But u_grid is constant. so index.
        # u_opt[k, i] = index of best control
        self.u_opt_idx = np.zeros((self.N, self.ns), dtype=np.int32)
        
        
        # Pre-calc Physics constants
        # Dynamic Capacity from file (kWh) -> Coulombs (As)
        # FORCE SMALL BATTERY (14 kWh) as per Sync Request
        cap_kwh = self.bat_capacity_kwh
        v_nom = self.truck.get_ocv(0.7)
        cap_coulombs = (cap_kwh * 3.6e6) / v_nom 
        q_max = cap_coulombs
        print(f"DP Physics: Cap={cap_kwh:.2f} kWh (Forced), V_nom={v_nom:.1f} V, Q={q_max:.1f} As")
        
        # Vectorized State Grid (Rows)
        # SOC_i: (Ns, 1)
        SOC_i = self.soc_grid.reshape(-1, 1)
        
        # Vectorized Control Grid (Cols)
        # T_mot_u: (1, Nu)
        T_mot_u = self.u_control_grid.reshape(1, -1)
        
        # Pre-calc Constant Parameter Curves for Speed
        # OCV and Rint depend on SOC.
        # We compute them for the State Grid once (assuming they don't change much with small dSOC in one step)
        voc_grid = self.truck.ocv_curve(self.soc_grid * 100).reshape(-1, 1)
        if self.truck.r_int_curve:
            r_grid = self.truck.r_int_curve(self.soc_grid * 100).reshape(-1, 1)
        else:
            r_grid = np.full((self.ns, 1), self.truck.fallback_r_int)

        from scipy.interpolate import interp1d

        # Add velocity array for standstill check
        if 'velocity_kmh' in self.cycle_df.columns:
            velocities = self.cycle_df['velocity_kmh'].values
        else:
            velocities = np.ones(self.N) * 10.0 # Dummy

        for k in range(self.N - 1, -1, -1):
            dt = self.dts[k]
            t_req = self.t_reqs[k]
            w_rpm = self.rpms[k]
            v_kmh = velocities[k]
            
            # --- Apply System Limits and Standstill to Match ECMS ---
            t_sys_min, t_sys_max = self.truck.get_system_limits(w_rpm)
            t_req = max(t_sys_min, min(t_sys_max, t_req))
            
            # --- 1. Calculate Next SOC using ECMS Logic ---
            
            # P_elec from T_mot (Control)
            # Map (n, T) -> P_el (kW)
            # Optimize: Calc P_elec for the Control Grid once (1D array)
            T_mot_u_eff = T_mot_u.copy()
            t_mot_min_phys, t_mot_max_phys = self.truck.get_limits(w_rpm)
            t_eng_min_phys, t_eng_max_phys = self.truck.get_eng_limits(w_rpm)
            
            if v_kmh < 0.1:
                safe_0 = max(t_mot_min_phys, min(t_mot_max_phys, 0.0))
                T_mot_u_eff = np.full_like(T_mot_u, safe_0)
            
            pts_mot = np.column_stack((np.full(self.nu, w_rpm), T_mot_u_eff.ravel()))
            P_el_1d = self.truck.em_eff_interp(pts_mot) # kW. (Nu,)
            P_el_u = P_el_1d.reshape(1, -1) # Broadcastable
            
            # Use Helper (Standard Physics)
            I_bat, mask_feas = self._calc_current_standard(voc_grid, r_grid, P_el_u)
            
            # dSOC = - I * dt / Q 
            # I in Amps.
            dSOC = - (I_bat * dt) / q_max
            
            # Next SOC
            SOC_next = SOC_i + dSOC
            
            # --- 2. Calculate Fuel Cost ---
            # Wait until standstill constraint to define T_eng_u
            T_eng_u = t_req - T_mot_u_eff # (1, Nu)
            
            # Fuel Map (n, T_eng) -> Fuel (1, Nu)
            # Again, depends only on Control, not SOC.
            pts_eng = np.column_stack((np.full(self.nu, w_rpm), T_eng_u.ravel()))
            Fuel_1d = self.truck.fuel_interp(pts_eng) # g/s
            # Calculate Idle Fuel (Fuel at 0 Torque) for this RPM
            pts_idle = np.column_stack((np.full(self.nu, w_rpm), np.zeros(self.nu)))
            idle_fuel_1d = self.truck.fuel_interp(pts_idle)

            # Some map corners (especially low RPM/standstill) may return NaN fuel.
            # Allow a conservative fallback only near zero engine torque to avoid full-step infeasibility.
            near_zero_teng = np.abs(T_eng_u.ravel()) <= 1.0
            idle_fuel_safe = np.where(np.isfinite(idle_fuel_1d), idle_fuel_1d, 0.0)
            Fuel_1d = np.where(np.isnan(Fuel_1d) & near_zero_teng, idle_fuel_safe, Fuel_1d)
            
            Fuel_cost = (Fuel_1d * dt).reshape(1, -1) # (1, Nu)
            
            # Apply Constraints
            # 1. Motor Power Infeasible (Delta < 0)
            # 2. SOC Next out of bounds (0.3 to 0.7)
            mask_soc = (SOC_next >= self.soc_min) & (SOC_next <= self.soc_max)
            
            # --- NEW: Dynamic Component Limits based on Map Curves ---
            t_mot_min_phys, t_mot_max_phys = self.truck.get_limits(w_rpm)
            t_eng_min_phys, t_eng_max_phys = self.truck.get_eng_limits(w_rpm)
            
            # --- Apply Standstill constraint to Control Grid (like ECMS) ---
            # (already applied earlier for T_mot_u_eff)
            
            mask_mot_lim = (T_mot_u_eff >= t_mot_min_phys) & (T_mot_u_eff <= t_mot_max_phys)
            mask_eng_lim = (T_eng_u >= t_eng_min_phys) & (T_eng_u <= t_eng_max_phys)
            mask_torque_lim = mask_mot_lim & mask_eng_lim
            
            # 3. Component Limits (NaN in maps)
            
            Total_Cost = Fuel_cost.copy() 
            
            # Handle Infeasibles
            Total_Cost = np.broadcast_to(Total_Cost, (self.ns, self.nu)).copy()
            Total_Cost[~mask_feas] = np.inf
            Total_Cost[~mask_soc] = np.inf
            Total_Cost[~np.broadcast_to(mask_torque_lim, (self.ns, self.nu))] = np.inf
            Total_Cost[np.isnan(Total_Cost)] = np.inf 
            
            # --- 3. Value Function Interpolation ---
            # Evaluate J_next at SOC_next
            # Use interp1d (fast linear interp)
            # Flatten SOC_next (Ns * Nu)
            SOC_next_flat = SOC_next.ravel()
            
            # extrapolation? usually inf, but we enforce bounds mask_soc. 
            # So just valid points.
            J_future = np.interp(SOC_next_flat, self.soc_grid, self.J_next, left=np.inf, right=np.inf)
            J_future = J_future.reshape(self.ns, self.nu)
            
            # Cost-to-Go
            Q_values = Total_Cost + J_future
            
            # Min over Control (dim 1)
            finite_mask = np.isfinite(Q_values)
            q_safe = np.where(finite_mask, Q_values, np.inf)
            min_vals = np.min(q_safe, axis=1)
            min_idxs = np.argmin(q_safe, axis=1)

            # Robust fallback: if a state has no finite control at this step,
            # keep DP numerically alive with a very large penalty and zero-torque control index.
            row_has_feasible = np.any(finite_mask, axis=1)
            if not np.all(row_has_feasible):
                bad_rows = ~row_has_feasible
                self._infeasible_rows_total += int(np.sum(bad_rows))
                fallback_idx = int(np.argmin(np.abs(self.u_control_grid)))
                stay_future = np.interp(self.soc_grid[bad_rows], self.soc_grid, self.J_next, left=np.inf, right=np.inf)
                stay_future = np.where(np.isfinite(stay_future), stay_future, 0.0)
                fallback_penalty = 1e6 + 10.0 * abs(t_req) * max(dt, 1e-3)
                min_vals[bad_rows] = stay_future + fallback_penalty
                min_idxs[bad_rows] = fallback_idx
            
            self.J_next = min_vals
            self.u_opt_idx[k, :] = min_idxs
            
            if k % 1000 == 0:
                 print(f"Step {k}: Min Cost = {np.nanmin(self.J_next):.2f}")
                 
        print(f"DP Solved in {time.time()-start_time:.1f}s")
        if self._infeasible_rows_total > 0:
            print(f"DP Warning: Applied infeasible-row fallback {self._infeasible_rows_total} times.")
        return self.J_next

    def _calc_current_standard(self, u_oc, r_bat, p_elec_kw):
        """
        Calculates I_bat using Standard Physics Convention.
        P_load (Watts) = P_elec_kw * 1000
        
        Discharge (P_load > 0):
            P_load = U * I - I^2 * R
            R*I^2 - U*I + P_load = 0
            I = (U - sqrt(U^2 - 4*R*P_load)) / 2R
            
        Charge (P_load < 0):
            P_source = -P_load
            P_source = U * I_chg + I_chg^2 * R  (I_chg defined as into battery)
            Or simply use the same quadratic eqn with signed P_load?
            Let I be discharge current (positive out).
            P_out = U*I - I^2*R.
            If P_out is negative (Charge), say -10kW.
            -10k = U*(-Ichg) - (-Ichg)^2 R
            -10k = -U*Ichg - Ichg^2 R
            10k = U*Ichg + Ichg^2 R.
            Matches.
            So one formula works for both:
            I = (U - sqrt(U^2 - 4*R*P_load)) / 2R.
        """
        p_bat_watts = p_elec_kw * 1000.0
        
        # Discriminant: U^2 - 4 R P
        discriminant = u_oc**2 - 4 * r_bat * p_bat_watts
        
        mask_feas = discriminant >= 0
        
        # Calculate I where valid
        sqrt_d = np.sqrt(np.maximum(0, discriminant))
        
        # Discharge Root (Smaller current preferred? No, I must match sign of P roughly)
        # I = (U - sqrt(D)) / 2R
        i_bat = (u_oc - sqrt_d) / (2 * r_bat)
        
        return i_bat, mask_feas

    def _calc_current_plant_match(self, u_oc, r_bat, p_elec_kw):
        # Match p2_hybrid.py sign convention
        p_bat_watts = p_elec_kw * 1000.0
        p_bat_eqn = -1.0 * p_bat_watts  
        # Discriminant
        discriminant = u_oc**2 - 4 * r_bat * p_bat_eqn
        
        mask_feas = discriminant >= 0
        
        # Calculate I
        sqrt_d = np.sqrt(np.maximum(0, discriminant))
        i_bat = (-u_oc + sqrt_d) / (2 * r_bat) 
        
        return i_bat, mask_feas

    def reconstruct_path(self, start_soc=0.70):
        print("Reconstructing Optimal Path...")
        
        soc_curr = start_soc
        target_end = getattr(self, 'dp_target_soc', None)
        
        time_hist = self.cycle_df['time'].values
        
        if 'dist_accum_m' in self.cycle_df.columns:
            dist_arr = self.cycle_df['dist_accum_m'].values
        else:
            dist_arr = np.linspace(0, 1, self.N)
        total_dist = dist_arr[-1] if dist_arr[-1] > 0 else 1.0

        if 'velocity_kmh' in self.cycle_df.columns:
            velocities = self.cycle_df['velocity_kmh'].values
        else:
            velocities = np.ones(self.N) * 10.0 # Dummy
        
        soc_hist = []
        target_soc_hist = [] if target_end is not None else None
        fuel_hist = []
        t_mot_hist = []
        t_eng_hist = []
        total_fuel = 0.0
        
        cap_kwh = self.bat_capacity_kwh
        v_nom = self.truck.get_ocv(0.7)
        cap_coulombs = (cap_kwh * 3.6e6) / v_nom
        
        for k in range(self.N):
            # Get optimal control for current SOC
            # Linear interp of T_mot
            opt_indices = self.u_opt_idx[k, :] # vector of Indices
            opt_tmots = self.u_control_grid[opt_indices] # vector of T_mot values
            
            # Interpolate T_mot for soc_curr
            t_mot = np.interp(soc_curr, self.soc_grid, opt_tmots)
            
            # Sim Forward
            dt = self.dts[k]
            t_req = self.t_reqs[k]
            w_rpm = self.rpms[k]
            v_kmh = velocities[k]

            t_sys_min, t_sys_max = self.truck.get_system_limits(w_rpm)
            t_req = max(t_sys_min, min(t_sys_max, t_req))
            
            t_mot_min_phys, t_mot_max_phys = self.truck.get_limits(w_rpm)
            if v_kmh < 0.1:
                t_mot = max(t_mot_min_phys, min(t_mot_max_phys, 0.0))
            
            # Physics
            val_eff = self.truck.em_eff_interp([[w_rpm, t_mot]])
            if hasattr(val_eff, "item"): val_eff = val_eff.item()
            p_el = float(val_eff)

            if np.isnan(p_el) and t_mot == 0: p_el = 0.0
            
            # Use Helper
            voc = self.truck.ocv_curve([soc_curr * 100])
            if hasattr(voc, "item"): voc = voc.item()
            voc = float(voc)

            if self.truck.r_int_curve:
                r = self.truck.r_int_curve([soc_curr * 100])
                if hasattr(r, "item"): r = r.item()
                r = float(r)
            else:
                r = self.truck.fallback_r_int
            
            i_bat, valid = self._calc_current_standard(voc, r, p_el)
            
            # Assuming valid path if DP converged
            if not valid:
                print(f"Warning: Reconstruct infeasible at k={k}")
                i_bat = 0.0
                
            dSOC = - (i_bat * dt) / cap_coulombs
            soc_next = soc_curr + dSOC
            
            t_eng = t_req - t_mot
            
            val_fuel = self.truck.fuel_interp([[w_rpm, t_eng]])
            if hasattr(val_fuel, "item"): val_fuel = val_fuel.item()
            fuel = float(val_fuel) * dt
            if np.isnan(fuel): fuel = 0.0
            
            total_fuel += fuel
            
            # Optional target profile for strategies that use one.
            if target_soc_hist is not None:
                curr_target = start_soc - (dist_arr[k] / total_dist) * (start_soc - target_end)
            
            soc_hist.append(soc_curr)
            if target_soc_hist is not None:
                target_soc_hist.append(curr_target)
            t_mot_hist.append(t_mot)
            t_eng_hist.append(t_eng)
            fuel_hist.append(fuel)
            
            soc_curr = soc_next
            
        print(f"DP Reconstruction Complete. Fuel: {total_fuel/1000.0:.3f} kg")
        print(f"DEBUG RECO STATS:")
        print(f"  Total Time: {np.sum(self.dts):.1f} s")
        print(f"  Mean T_req: {np.mean(self.t_reqs):.1f} Nm")
        print(f"  Mean T_mot: {np.mean(t_mot_hist):.1f} Nm")
        print(f"  Mean T_eng: {np.mean(t_eng_hist):.1f} Nm")
        print(f"  Mean RPM:   {np.mean(self.rpms):.1f}")
        dSOC_total = start_soc - soc_hist[-1]
        energy_batt_kwh = dSOC_total * float(self.bat_capacity_kwh)
        print(f"  SOC Drop: {dSOC_total*100:.2f}% -> {energy_batt_kwh:.2f} kWh used")
        
        out = {
            'time': time_hist,
            'soc': np.array(soc_hist),
            't_mot': np.array(t_mot_hist),
            't_eng': np.array(t_eng_hist),
            'total_fuel_kg': total_fuel/1000.0
        }
        if target_soc_hist is not None:
            out['target_soc'] = np.array(target_soc_hist)
        return out
