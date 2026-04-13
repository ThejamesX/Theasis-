import numpy as np
import time
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator

class DPOptimizer:
    def __init__(self, truck, cycle_df, soc_grid_size=150, bat_capacity_kwh =120.0):
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
        self.vels = cycle_df['velocity_kmh'].values if 'velocity_kmh' in cycle_df.columns else np.zeros_like(times)
        
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
        
        # Grid Setup
        # SOC Grid (States)
        self.J_next = np.full(self.ns, np.inf)
        
        # Terminal Cost: Soft Constraint
        penalty_weight = 1e6 
        self.J_next = penalty_weight * (self.soc_grid - target_soc)**2
        
        # Control Variable Grid (T_mot)
        # Moved below after getting dynamic limits
        
        # Storage for Optimal Control (We store the optimal T_mot index or value?)
        # Storing value is easier, or index into u_grid.
        # But u_grid is constant. so index.
        # u_opt[k, i] = index of best control
        self.u_opt_idx = np.zeros((self.N, self.ns), dtype=np.int32)
        
        # Determine global max limits for DP grid to ensure coverage
        max_rpm_test = np.linspace(400, 2500, 100)
        global_t_em_min = min([-self.truck.em_params.get('OverloadTorque', 1050)])
        global_t_em_max = max([self.truck.em_params.get('OverloadTorque', 1050)])
        if self.truck.fld_drive_interp is not None:
            global_t_em_min = min(float(self.truck.fld_drag_interp(r)) for r in max_rpm_test)
            global_t_em_max = max(float(self.truck.fld_drive_interp(r)) for r in max_rpm_test)
            
        self.u_control_grid = np.linspace(global_t_em_min, global_t_em_max, 101) # T_mot candidates
        self.nu = len(self.u_control_grid)
        
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

        for k in range(self.N - 1, -1, -1):
            dt = self.dts[k]
            t_req = self.t_reqs[k]
            w_rpm = self.rpms[k]
            v_kmh = self.vels[k]
            
            # Constraints Base Masks
            mask_feas = np.ones((self.ns, self.nu), dtype=bool)
            
            if v_kmh < 0.1:
                # Standstill constraint: Only T_mot = 0 is valid. Everything else becomes infeasible.
                valid_u_mask = (self.u_control_grid == 0.0)
                mask_feas[:, ~valid_u_mask] = False
            
            # --- 1. Calculate Next SOC using ECMS Logic ---
            t_sys_min, t_sys_max = self.truck.get_system_limits(w_rpm)
            t_req_hybrid = max(t_sys_min, min(t_sys_max, t_req))
            
            # P_elec from T_mot (Control)
            # Map (n, T) -> P_el (kW)
            # Optimize: Calc P_elec for the Control Grid once (1D array)
            pts_mot = np.column_stack((np.full(self.nu, w_rpm), self.u_control_grid))
            P_el_1d = self.truck.em_eff_interp(pts_mot) # kW. (Nu,)
            P_el_u = P_el_1d.reshape(1, -1) # Broadcastable
            
            # Use Helper (Standard Physics)
            I_bat, mask_batt_feas = self._calc_current_standard(voc_grid, r_grid, P_el_u)
            mask_feas &= mask_batt_feas
            
            # dSOC = - I * dt / Q 
            # I in Amps.
            dSOC = - (I_bat * dt) / q_max
            
            # Next SOC
            SOC_next = SOC_i + dSOC
            
            # --- 2. Calculate Fuel Cost ---
            # T_eng = T_req - T_mot
            T_eng_u = t_req_hybrid - T_mot_u # (1, Nu)
            
            # Fuel Map (n, T_eng) -> Fuel (1, Nu)
            # Again, depends only on Control, not SOC.
            pts_eng = np.column_stack((np.full(self.nu, w_rpm), T_eng_u.ravel()))
            Fuel_1d = self.truck.fuel_interp(pts_eng) # g/s
            # Calculate Idle Fuel (Fuel at 0 Torque) for this RPM
            pts_idle = np.column_stack((np.full(self.nu, w_rpm), np.zeros(self.nu)))
            idle_fuel_1d = self.truck.fuel_interp(pts_idle)

            # If RPM > 500 (Engine ON), Fuel cannot be less than Idle
            # (Assuming VECTO RPMs imply engine is spinning)
#if w_rpm > 500:
#Fuel_1d = np.maximum(Fuel_1d, idle_fuel_1d)
            
            Fuel_cost = (Fuel_1d * dt).reshape(1, -1) # (1, Nu)
            
            # Apply Constraints
            # 1. Motor Power Infeasible (Delta < 0)
            # 2. SOC Next out of bounds (0.3 to 0.7)
            mask_soc = (SOC_next >= self.soc_min) & (SOC_next <= self.soc_max)
            
            # Dynamic Torq Limits per step
            t_mot_min_phys, t_mot_max_phys = self.truck.get_limits(w_rpm)
            t_eng_min_phys, t_eng_max_phys = self.truck.get_eng_limits(w_rpm)
            
            # 3. Component Physical Limits
            # Add tolerance due to discrete control grid to prevent empty feasible sets
            tol = 40.0 # Nm
            mask_limits = (self.u_control_grid >= t_mot_min_phys - tol) & (self.u_control_grid <= t_mot_max_phys + tol)
            mask_limits &= (T_eng_u.ravel() >= t_eng_min_phys - tol) & (T_eng_u.ravel() <= t_eng_max_phys + tol)
            mask_limits = mask_limits.reshape(1, -1)

            mask_total = mask_feas & mask_soc & mask_limits
            
            # Additional Standstill Constraint (0 kmh)
            if v_kmh < 0.1:
                # Force T_mot = 0
                idx_zero = np.argmin(np.abs(self.u_control_grid))
                mask_total &= (np.arange(self.nu) == idx_zero).reshape(1, -1)
                mask_total[:, idx_zero] = True # Ensure at least 0 is feasible
            
            Total_Cost = Fuel_cost.copy() 
            
            # Handle Infeasibles
            Total_Cost = np.broadcast_to(Total_Cost, (self.ns, self.nu)).copy()
            # Use large penalty instead of inf to prevent backward cascade of infeasibility blocking the entire grid
            PENALTY = 1e9
            Total_Cost[~mask_total] = PENALTY
            Total_Cost[np.isnan(Total_Cost)] = PENALTY 
            
            # --- 3. Value Function Interpolation ---
            # Evaluate J_next at SOC_next
            # Use interp1d (fast linear interp)
            # Flatten SOC_next (Ns * Nu)
            SOC_next_flat = SOC_next.ravel()
            
            # extrapolation? usually inf, but we enforce bounds mask_soc. 
            # So just valid points.
            J_future = np.interp(SOC_next_flat, self.soc_grid, self.J_next, left=PENALTY, right=PENALTY)
            J_future = J_future.reshape(self.ns, self.nu)
            
            # Cost-to-Go
            Q_values = Total_Cost + J_future
            
            # Min over Control (dim 1)
            # Use nanmin to ignore NaNs
            min_vals = np.nanmin(Q_values, axis=1)
            min_idxs = np.nanargmin(Q_values, axis=1)
            
            # Safe Fallback: if all costs are > 1e11, force Motor Torque to closest equivalent of 0, not max regenerative limit.
            all_inf = min_vals >= 1e11
            if np.any(all_inf):
                idx_zero = np.argmin(np.abs(self.u_control_grid))
                min_idxs[all_inf] = idx_zero
            
            self.J_next = min_vals
            self.u_opt_idx[k, :] = min_idxs
            
            if k % 1000 == 0:
                 print(f"Step {k}: Min Cost = {np.nanmin(self.J_next):.2f}")
                 
        print(f"DP Solved in {time.time()-start_time:.1f}s")
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
        target_end = 0.30
        
        time_hist = self.cycle_df['time'].values
        
        if 'dist_accum_m' in self.cycle_df.columns:
            dist_arr = self.cycle_df['dist_accum_m'].values
        else:
            dist_arr = np.linspace(0, 1, self.N)
        total_dist = dist_arr[-1] if dist_arr[-1] > 0 else 1.0
        
        soc_hist = []
        target_soc_hist = []
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
            
            # Constraints override for reconstruct
            v_kmh = self.vels[k]
            if v_kmh < 0.1:
                opt_tmots[:] = 0.0
            
            # Use nearest instead of linear interpolation for control
            # Interpolating optimal controls can lead to completely sub-optimal intermediate controls.
            idx_soc = np.abs(self.soc_grid - soc_curr).argmin()
            t_mot = opt_tmots[idx_soc]
            
            # Sim Forward
            dt = self.dts[k]
            t_req = self.t_reqs[k]
            w_rpm = self.rpms[k]
            
            t_sys_min, t_sys_max = self.truck.get_system_limits(w_rpm)
            t_req_hybrid = max(t_sys_min, min(t_sys_max, t_req))
            
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
                i_bat = 0.0
                
            dSOC = - (i_bat * dt) / cap_coulombs
            soc_next = soc_curr + dSOC
            
            t_eng = t_req_hybrid - t_mot
            
            # Reconstruct Fuel with same idle logic as backward pass
            val_fuel = self.truck.fuel_interp([[w_rpm, t_eng]])
            if hasattr(val_fuel, "item"): val_fuel = val_fuel.item()
            
            val_idle_fuel = self.truck.fuel_interp([[w_rpm, 0.0]])
            if hasattr(val_idle_fuel, "item"): val_idle_fuel = val_idle_fuel.item()
            
#if w_rpm > 500:
#val_fuel = max(val_fuel, val_idle_fuel)
                
            fuel = float(val_fuel) * dt
            if np.isnan(fuel): fuel = 0.0
            
            total_fuel += fuel
            
            # Linear Reference Target
            curr_target = start_soc - (dist_arr[k] / total_dist) * (start_soc - target_end)
            
            soc_hist.append(soc_curr)
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
        
        return {
            'time': time_hist,
            'soc': np.array(soc_hist),
            'target_soc': np.array(target_soc_hist),
            't_mot': np.array(t_mot_hist),
            't_eng': np.array(t_eng_hist),
            'total_fuel_kg': total_fuel/1000.0
        }
