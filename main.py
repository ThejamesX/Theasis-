import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import PowerNorm
from scipy.interpolate import interp1d
from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller

def run_ecms_simulation(strategy='AECMS', cycle_file=None, bat_capacity_kwh=120.0, output_prefix='ecms'):
    # 1. Paths
    # Using absolute paths as requested or safer relative if running from root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cycle_file:
        vmod_path = cycle_file
    else:
        vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
        
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    # Note: "EM_Map - kopie.vemo" might be tricky with spaces if not handled well, but python open() handles it.
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    # New Resistance file check
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    # 2. Initialize
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    
    print(f"--- Running {strategy} | Cap: {bat_capacity_kwh} kWh | Cycle: {os.path.basename(vmod_path)} ---")
    # Pass resistance path (it handles if missing)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    # Override Capacity
    if bat_capacity_kwh is not None:
        truck.bat_params['Capacity'] = float(bat_capacity_kwh)

    # 3. Load Cycle
    cycle_df = loader.read_vmod(vmod_path)
    if cycle_df is None:
        print("Failed to load cycle.")
        return None

    # 4. Backward Physics
    # Add T_req column to cycle vector
    t_req = truck.calc_backward_physics(cycle_df)
    cycle_df['t_req_hybrid_in'] = t_req
    
    # 5. Simulation Loop
    target_soc = 0.70 # Start at 70%
    # Q_LHV = 42700 J/g
    controller = ECMS_Controller(truck, q_lhv=42700.0) 
    
    # Capacity handling strictly
    # VECTO "Capacity" is typically kWh.
    # Q_max [As] = (kWh * 3600 * 1000) / V_nom
    v_nom = truck.ocv_curve(70).item() # 70% SOC voltage
    cap_kwh = truck.bat_params.get('Capacity', 120.0)
    q_max_as = (cap_kwh * 3.6e6) / v_nom
    # print(f"DEBUG: V_nom={v_nom:.2f} V, Cap={cap_kwh:.2f} kWh, Q_max={q_max_as:.2f} As")

    # --- P-ECMS Strategy Selection ---
    # --- Strategy Selection ---
    # Options: 'ECMS', 'A-ECMS', 'LINEAR', 'GRAVITY', 'ENERGY', 'PECMS'
    STRATEGY = strategy
    
    # Imports
    sys.path.append(os.path.join(base_dir, 'P_ECMS')) 
    sys.path.append(os.path.join(base_dir, 'A_ECMS_Implementation'))
    
  
    from P_ECMS.new_horizon_predictor import NewHorizonPredictor
  
    from P_ECMS.pecms_supervisor import PECMS_Supervisor
    from P_ECMS.pecms_supervisor import PECMS_Supervisor
    from A_ECMS_Implementation.aecms_controller import AECMS_Controller

    # Default Predictor (can be overwritten)
    predictor = NewHorizonPredictor(cycle_df, spatial_step=50.0)
    
    supervisor = None
    
    # 1. Standard ECMS (Fixed)
    if STRATEGY == 'ECMS':
        # print("Strategy: Standard ECMS (Fixed Factors)")
        # Base Controller already initialized above
        pass 
        
    # 2. A-ECMS (Adaptive Proportional)
    elif STRATEGY in ['AECMS', 'A-ECMS']:
        # print("Strategy: A-ECMS (Proportional Feedback)")
        # Replace base controller with Adaptive one
        controller = AECMS_Controller(truck, kp_dis=30, kp_chg=0.01, target_soc=target_soc)  
     #3. P-ECMS Variants (Supervisor + Base Controller)
        # Use New PECMS Supervisor (Updated Init)
    elif STRATEGY == 'PECMS':
        # print("Strategy: P-ECMS (Constant Reference)") 
            
        controller.s_dis = 2.3395
        controller.s_chg = 1.7538
        predictor = NewHorizonPredictor(cycle_df, spatial_step=50.0)
        supervisor = PECMS_Supervisor(truck, controller, q_max_as, target_soc=target_soc)
        
    else:
        raise ValueError(f"Unknown Strategy: {STRATEGY}")
    
    # Storage
    results = {
        'time': cycle_df['time'].values,
        'velocity_kmh': cycle_df['velocity_kmh'].values,
        'altitude_m': cycle_df['altitude_m'].values if 'altitude_m' in cycle_df.columns else np.zeros_like(cycle_df['time'].values),
        'rpm_ice': cycle_df['rpm_ice'].values,
        't_req': t_req,
        'soc': [],
        'soc_target': [], 
        't_ice': [],
        't_em': [],
        'fuel_rate': [],
        'p_chem': [],
        'cost_inst': [],
        's_factor': []
    }
    
    soc = target_soc
    curr_target = target_soc # Initial
    
    # print(f"Starting Simulation... (Steps: {len(cycle_df)})")
    
    times = cycle_df['time'].values
    dts = cycle_df['dt'].values # New column from VECTO
    rpms = cycle_df['rpm_ice'].values
    t_reqs = cycle_df['t_req_hybrid_in'].values
    
    total_fuel_g = 0.0
    
    # Running Dist for logging
    curr_dist = 0
    
    for i in range(len(cycle_df)):
        t = times[i]
        rpm = rpms[i]
        tr = t_reqs[i]
        
        # dt from file (Exact VECTO Step)
        dt = dts[i]
        vel = cycle_df['velocity_kmh'].values[i] if 'velocity_kmh' in cycle_df.columns else 0.0
            
        # Linear Target Update
        curr_dist_val = predictor.dist_arr[i] if hasattr(predictor, 'dist_arr') else 0
        total_dist_val = predictor.dist_arr[-1] if hasattr(predictor, 'dist_arr') else 1
        curr_lin_target = 0.70 - (curr_dist_val / total_dist_val) * (0.70 - 0.30)
        
        if hasattr(controller, 'target_soc'):
            controller.target_soc = curr_lin_target

        # --- Update Logic ---
        if supervisor is not None:
             supervisor.target_soc = curr_lin_target
             if hasattr(supervisor, 'soc_nominal'):
                 supervisor.soc_nominal = curr_lin_target
             
             # P-ECMS Update (Every 3 steps? Or every step for accuracy? P-ECMS typically periodic)
             # Let's keep every 3 steps to save compute, matches legacy
             if i % 3 == 0:
                horizon = predictor.get_horizon(i)
                curr_dist = predictor.dist_arr[i]
                
                # Unpack target
                opt_s, curr_target, ratio = supervisor.get_optimal_s(curr_dist, soc, horizon)
                
                # Update Controller
                controller.s_dis = opt_s
                controller.s_chg = opt_s * ratio # maintain ratio
        else:
            # ECMS / A-ECMS
            curr_target = curr_lin_target # Update reference target for plotting
            
        # Logging print (reduced)
        # if i % 5000 == 0:
        #     print(f"Step {i}: s={controller.s_dis:.4f}, SOC={soc:.4f}, Target={curr_target:.4f}")
            
        # Call Controller (Uses updated or internal s)
        t_eng, t_mot, h_cost_watts, p_chem_watts, fuel_g_s = controller.decide_split(tr, rpm, soc, vel)
        
        # Store s and target
        results['s_factor'].append(controller.s_dis)
        results['soc_target'].append(curr_target)
        
        # Update SOC strictly (Eq 13)
        v_oc = truck.ocv_curve(soc * 100.0).item()
        
        # Protect div zero? Voc won't be 0.
        i_bat = p_chem_watts / v_oc
        
        # Eq 13: dot_soc = - i_bat / q_max_as
        dot_soc = - i_bat / q_max_as
        
        soc = soc + dot_soc * dt
        
        # Clamp
        soc = max(0.0, min(1.0, soc))
        
        # Store
        results['soc'].append(soc)
        results['t_ice'].append(t_eng)
        results['t_em'].append(t_mot)
        results['fuel_rate'].append(fuel_g_s)
        results['p_chem'].append(p_chem_watts)
        results['cost_inst'].append(h_cost_watts)
        
        total_fuel_g += fuel_g_s * dt
        
    # print("Simulation Complete.")
    # --- Plotting ---
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 11})
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    ax_soc = fig.add_subplot(gs[0, :])
    ax_ice = fig.add_subplot(gs[1, 0])
    ax_em = fig.add_subplot(gs[1, 1])

    # 1) SOC trajectory (reference line kept)
    ax_soc.plot(results['time'], np.array(results['soc']) * 100, label='Stav nabití [%]', color='black', linewidth=2.0)
    ax_soc.plot(results['time'], np.array(results['soc_target']) * 100, label='Cílový stav nabití [%]', color='tab:red', linestyle='--', linewidth=1.5)
    soc_time = np.asarray(results['time'])
    if soc_time.size > 1 and np.isfinite(soc_time).all():
        ax_soc.set_xlim(float(np.min(soc_time)), float(np.max(soc_time)))
    ax_soc.set_ylabel('Stav nabití [%]')
    ax_soc.set_xlabel('Čas [s]')
    ax_soc.set_title(f'Strategie: {STRATEGY} | Kapacita baterie: {bat_capacity_kwh} | Palivo: {total_fuel_g/1000:.2f} kg', fontweight='bold')
    ax_soc.legend(loc='upper right')
    ax_soc.grid(True, linestyle=':', alpha=0.7)

    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    # Keep percentage overlay code available but default to point-only runtime view.
    show_ice_runtime_percentage_overlay = False

    rpm_arr = np.array(results['rpm_ice'])
    t_ice_arr = np.array(results['t_ice'])
    t_em_arr = np.array(results['t_em'])

    # 2) ICE load point map
    max_rpm = np.array([])
    max_tq = np.array([])
    bsfc_triang = None
    bsfc_vals = None
    try:
        df_vmap = pd.read_csv(vmap_path, sep=',', header=0)
        rpm_raw = df_vmap.iloc[:, 0].to_numpy()
        torque_raw = df_vmap.iloc[:, 1].to_numpy()
        fuel_raw = df_vmap.iloc[:, 2].to_numpy() if df_vmap.shape[1] >= 3 else np.zeros_like(rpm_raw)

        # Build BSFC values for background contour.
        mask_bsfc = (torque_raw >= 0.0) & (fuel_raw >= 0.0)
        rpm_bsfc = rpm_raw[mask_bsfc]
        tq_bsfc = torque_raw[mask_bsfc]
        fuel_bsfc = fuel_raw[mask_bsfc]
        omega_bsfc = rpm_bsfc * 2.0 * np.pi / 60.0
        p_kw_bsfc = tq_bsfc * omega_bsfc / 1000.0
        bsfc = np.full_like(p_kw_bsfc, np.nan, dtype=float)
        valid_power = p_kw_bsfc > 1.0
        bsfc[valid_power] = fuel_bsfc[valid_power] / p_kw_bsfc[valid_power]
        bsfc[~valid_power] = 420.0

        drpm = np.diff(rpm_raw)
        block_ends = np.where(np.abs(drpm) > 5.0)[0]
        block_ends = np.append(block_ends, len(rpm_raw) - 1)

        max_tq_curve = []
        for k_idx, end_idx in enumerate(block_ends):
            start_idx = 0 if k_idx == 0 else block_ends[k_idx - 1] + 1
            rp_block = rpm_raw[start_idx:end_idx + 1]
            tq_block = torque_raw[start_idx:end_idx + 1]
            if tq_block.max() > 0:
                max_tq_curve.append((rp_block[np.argmax(tq_block)], tq_block.max()))

        if max_tq_curve:
            max_tq_curve = np.array(max_tq_curve)
            max_rpm = max_tq_curve[:, 0]
            max_tq = max_tq_curve[:, 1]

        if rpm_bsfc.size > 3:
            # Match plot_vecto_components.py: normalize, refine, convert back.
            rpm_scale = max(1e-9, float(rpm_bsfc.max() - rpm_bsfc.min()))
            tq_scale = max(1e-9, float(tq_bsfc.max() - tq_bsfc.min()))
            rpm_norm = (rpm_bsfc - rpm_bsfc.min()) / rpm_scale
            tq_norm = (tq_bsfc - tq_bsfc.min()) / tq_scale

            triang_norm = mtri.Triangulation(rpm_norm, tq_norm)
            refiner = mtri.UniformTriRefiner(triang_norm)
            triang_fine, bsfc_fine = refiner.refine_field(bsfc, subdiv=3)

            rpm_fine = triang_fine.x * rpm_scale + rpm_bsfc.min()
            tq_fine = triang_fine.y * tq_scale + tq_bsfc.min()
            bsfc_triang = mtri.Triangulation(rpm_fine, tq_fine, triang_fine.triangles)

            if max_rpm.size > 1 and bsfc_triang.triangles.size > 0:
                centroids_rpm = np.mean(rpm_fine[bsfc_triang.triangles], axis=1)
                centroids_tq = np.mean(tq_fine[bsfc_triang.triangles], axis=1)
                tq_env = np.interp(
                    np.clip(centroids_rpm, max_rpm.min(), max_rpm.max()),
                    max_rpm,
                    max_tq,
                )
                # Strictly enforce max torque envelope.
                bsfc_triang.set_mask((centroids_tq > tq_env) | (centroids_tq < 0.0))
            bsfc_vals = bsfc_fine
    except Exception as e:
        print(f"Warning: Could not load engine BSFC data for strategy plot: {e}")

    mask_ice = t_ice_arr > 0.0
    rpm_ice_map = rpm_arr[mask_ice]
    t_ice_map = t_ice_arr[mask_ice]

    # BSFC background first so runtime overlay is directly readable against efficiency islands.
    if bsfc_triang is not None and bsfc_vals is not None and len(bsfc_vals) > 0:
        levels_bsfc = np.concatenate([np.arange(182, 210, 2), np.arange(210, 340, 10)])
        cf_bsfc = ax_ice.tricontourf(
            bsfc_triang,
            bsfc_vals,
            levels=levels_bsfc,
            cmap='Spectral_r',
            extend='both',
            alpha=0.8,
            zorder=0,
        )
        ax_ice.tricontour(
            bsfc_triang,
            bsfc_vals,
            levels=np.arange(190, 320, 10),
            colors='black',
            linewidths=0.35,
            alpha=0.35,
            zorder=1,
        )
        fig.colorbar(cf_bsfc, ax=ax_ice, label='Měrná spotřeba paliva [g/kWh]')

    if rpm_ice_map.size > 0:
        y_max_ice = float(max_tq.max() * 1.05) if max_tq.size > 0 else float(max(200.0, np.max(t_ice_map) * 1.05))
        hist_ice, xedges_ice, yedges_ice = np.histogram2d(
            rpm_ice_map,
            t_ice_map,
            bins=50,
            range=[[400, 2500], [0, y_max_ice]],
        )
        if hist_ice.sum() > 0:
            hist_ice = (hist_ice / hist_ice.sum()) * 100.0

        hist_ice_map = gaussian_filter(hist_ice, sigma=1.0) if gaussian_filter is not None else hist_ice
        # Keep low-occupancy runtime visible while removing near-zero speckle.
        hist_ice_map[hist_ice_map < 0.005] = np.nan
        X_ice, Y_ice = np.meshgrid(xedges_ice[:-1], yedges_ice[:-1])
        finite_ice = np.isfinite(hist_ice_map)
        if show_ice_runtime_percentage_overlay and np.any(finite_ice):
            vmax_ice = float(np.nanmax(hist_ice_map))
            if vmax_ice > 0:
                ax_ice.contourf(
                    X_ice,
                    Y_ice,
                    hist_ice_map.T,
                    levels=np.linspace(0.0, vmax_ice, 18),
                    cmap='Blues',
                    norm=PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax_ice),
                    alpha=0.28,
                    zorder=2,
                )
                levels_occ = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3.0])
                levels_occ = levels_occ[levels_occ <= vmax_ice]
                if levels_occ.size > 0:
                    cs_ice = ax_ice.contour(
                        X_ice,
                        Y_ice,
                        hist_ice_map.T,
                        levels=levels_occ,
                        colors='#0b4fa2',
                        linewidths=1.0,
                        alpha=0.9,
                        zorder=3,
                    )
                    ax_ice.clabel(cs_ice, inline=True, fontsize=7, fmt='%g%%')

        # Show actual runtime trace points so operating region is unambiguous.
        stride_ice = max(1, rpm_ice_map.size // 1400)
        ax_ice.scatter(
            rpm_ice_map[::stride_ice],
            t_ice_map[::stride_ice],
            s=13,
            c='black',
            alpha=0.24,
            edgecolors='white',
            linewidths=0.15,
            zorder=4,
            label='Provozní body',
        )

    if max_rpm.size > 0:
        ax_ice.plot(max_rpm, max_tq, 'k-', linewidth=3.0, label='Křivka max. točivého momentu', zorder=5)

    # Use strict data min/max limits on x-axis (no extra padding).
    x_limit_data = []
    if max_rpm.size > 0:
        x_limit_data.append(max_rpm)
    if rpm_ice_map.size > 0:
        x_limit_data.append(rpm_ice_map)

    if x_limit_data:
        x_all = np.concatenate(x_limit_data)
        x_min = float(np.nanmin(x_all))
        x_max = float(np.nanmax(x_all))
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            x_min, x_max = 0.0, 2500.0
    else:
        x_min, x_max = 0.0, 2500.0

    y_top_candidates = []
    if max_tq.size > 0:
        y_top_candidates.append(float(np.nanmax(max_tq) * 1.05))
    if t_ice_map.size > 0:
        y_top_candidates.append(float(np.nanmax(t_ice_map) * 1.10))
    ice_ylim_top = max(400.0, *y_top_candidates) if y_top_candidates else 3000.0

    ax_ice.set_title('Mapa provozních bodů spalovacího motoru', fontweight='bold')
    ax_ice.set_xlabel('Otáčky motoru [1/min]')
    ax_ice.set_ylabel('Točivý moment motoru [Nm]')
    ax_ice.set_xlim(x_min, x_max)
    ax_ice.set_ylim(0.0, ice_ylim_top)
    ax_ice.grid(True, linestyle=':', alpha=0.7)
    if max_rpm.size > 0 or rpm_ice_map.size > 0:
        ax_ice.legend(
            loc='upper right',
            fontsize=8,
            framealpha=0.85,
            borderpad=0.3,
            labelspacing=0.25,
            handlelength=1.4,
            markerscale=0.85,
        )

    # 3) E-Motor load point map
    em_rpm_lim = np.array([])
    em_tq_drive = np.array([])
    em_tq_drag = np.array([])
    em_eff_triang = None
    em_eff_vals = None
    try:
        em_limit_path = os.path.join(base_dir, "Emotor", "EM_fld.vemp")
        df_vem = pd.read_csv(em_limit_path, skipinitialspace=True)
        em_rpm_lim = df_vem.iloc[:, 0].to_numpy()
        em_tq_drive = df_vem.iloc[:, 1].to_numpy()
        em_tq_drag = df_vem.iloc[:, 2].to_numpy()

        # E-Motor efficiency map background from vemo file.
        em_map_path = os.path.join(base_dir, "Emotor", "EM_Map - kopie.vemo")
        df_em_map = pd.read_csv(em_map_path, skipinitialspace=True)
        df_em_map.columns = [c.strip().lower() for c in df_em_map.columns]

        rpm_col = next((c for c in df_em_map.columns if ('rpm' in c) and ('n' in c or 'speed' in c)), None)
        tq_col = next((c for c in df_em_map.columns if ('nm' in c) and (c.startswith('t') or 'torque' in c)), None)
        pel_col = next((c for c in df_em_map.columns if 'p_el' in c), None)

        if rpm_col is not None and tq_col is not None and pel_col is not None:
            rpm_map = df_em_map[rpm_col].to_numpy()
            tq_map = df_em_map[tq_col].to_numpy()
            p_el_kw = df_em_map[pel_col].to_numpy()

            omega_map = rpm_map * 2.0 * np.pi / 60.0
            p_mech_kw = tq_map * omega_map / 1000.0

            eff = np.full_like(rpm_map, np.nan, dtype=float)
            mot_mask = (tq_map > 0.5) & (p_el_kw > 0.01) & (rpm_map > 10.0)
            gen_mask = (tq_map < -0.5) & (p_el_kw < -0.01) & (rpm_map > 10.0)
            eff[mot_mask] = (p_mech_kw[mot_mask] / p_el_kw[mot_mask]) * 100.0
            eff[gen_mask] = (np.abs(p_el_kw[gen_mask]) / np.abs(p_mech_kw[gen_mask])) * 100.0
            eff = np.clip(eff, 50.0, 99.0)

            valid_eff = (mot_mask | gen_mask) & np.isfinite(eff)
            if np.sum(valid_eff) > 3 and em_rpm_lim.size > 1:
                r_v = rpm_map[valid_eff]
                t_v = tq_map[valid_eff]
                e_v = eff[valid_eff]

                r_scale = max(1e-9, float(r_v.max() - r_v.min()))
                t_scale = max(1e-9, float(t_v.max() - t_v.min()))
                r_norm = (r_v - r_v.min()) / r_scale
                t_norm = (t_v - t_v.min()) / t_scale

                tri_norm = mtri.Triangulation(r_norm, t_norm)
                refiner = mtri.UniformTriRefiner(tri_norm)
                tri_fine, eff_fine = refiner.refine_field(e_v, subdiv=3)

                rpm_fine = tri_fine.x * r_scale + r_v.min()
                tq_fine = tri_fine.y * t_scale + t_v.min()
                tri_phys = mtri.Triangulation(rpm_fine, tq_fine, tri_fine.triangles)

                if tri_phys.triangles.size > 0:
                    drive_interp = interp1d(
                        em_rpm_lim,
                        em_tq_drive,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate',
                    )
                    drag_interp = interp1d(
                        em_rpm_lim,
                        em_tq_drag,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate',
                    )
                    c_rpm = np.mean(rpm_fine[tri_phys.triangles], axis=1)
                    c_tq = np.mean(tq_fine[tri_phys.triangles], axis=1)
                    drive_env = drive_interp(np.clip(c_rpm, em_rpm_lim.min(), em_rpm_lim.max()))
                    drag_env = drag_interp(np.clip(c_rpm, em_rpm_lim.min(), em_rpm_lim.max()))
                    tri_phys.set_mask((c_tq > drive_env) | (c_tq < drag_env))

                em_eff_triang = tri_phys
                em_eff_vals = eff_fine
    except Exception as e:
        print(f"Warning: Could not load E-Motor limits for strategy plot: {e}")

    mask_em = np.abs(t_em_arr) > 0.5
    rpm_em_map = rpm_arr[mask_em]
    t_em_map = t_em_arr[mask_em]

    # Efficiency contour map background.
    if em_eff_triang is not None and em_eff_vals is not None and len(em_eff_vals) > 0:
        levels_eff = np.arange(60, 100, 2)
        cf_eff = ax_em.tricontourf(
            em_eff_triang,
            em_eff_vals,
            levels=levels_eff,
            cmap='RdYlGn',
            extend='both',
            alpha=0.85,
            zorder=0,
        )
        ax_em.tricontour(
            em_eff_triang,
            em_eff_vals,
            levels=[65, 70, 75, 80, 85, 90, 92, 95],
            colors='black',
            linewidths=0.35,
            alpha=0.35,
            zorder=1,
        )
        fig.colorbar(cf_eff, ax=ax_em, label='Účinnost [%]')

    if rpm_em_map.size > 0:
        # Point-based runtime view for E-Motor map (same idea as ICE points).
        stride_em = max(1, rpm_em_map.size // 1400)
        ax_em.scatter(
            rpm_em_map[::stride_em],
            t_em_map[::stride_em],
            s=13,
            c='black',
            alpha=0.15,
            edgecolors='white',
            linewidths=0.2,
            zorder=3,
            label='Provozní body',
        )

        # Intentionally kept (commented) so runtime-percentage overlay can be re-enabled later.
        # hist_em, xedges_em, yedges_em = np.histogram2d(
        #     rpm_em_map,
        #     t_em_map,
        #     bins=50,
        #     range=[[0, 3000], [-1500, 1500]],
        # )
        # if hist_em.sum() > 0:
        #     hist_em = (hist_em / hist_em.sum()) * 100.0
        #
        # hist_em_map = gaussian_filter(hist_em, sigma=1.0) if gaussian_filter is not None else hist_em
        # hist_em_map[hist_em_map < 0.05] = np.nan
        # X_em, Y_em = np.meshgrid(xedges_em[:-1], yedges_em[:-1])
        # cf_em = ax_em.contourf(X_em, Y_em, hist_em_map.T, cmap='Greens', levels=15, alpha=0.8)
        # fig.colorbar(cf_em, ax=ax_em, label='Time share / %')

    if em_rpm_lim.size > 0:
        ax_em.plot(em_rpm_lim, em_tq_drive, 'k-', linewidth=2.5, label='Max. hnací moment')
        ax_em.plot(em_rpm_lim, em_tq_drag, 'k--', linewidth=2.5, label='Max. rekuperační moment')

    ax_em.set_title('Mapa provozních bodů elektromotoru', fontweight='bold')
    ax_em.set_xlabel('Otáčky elektromotoru [1/min]')
    ax_em.set_ylabel('Točivý moment elektromotoru [Nm]')
    ax_em.axhline(0, color='gray', linewidth=1)

    # Center the EM map viewport around active data while keeping practical bounds.
    x_em_data = []
    y_em_data = []
    if em_rpm_lim.size > 0:
        x_em_data.append(em_rpm_lim)
        y_em_data.append(em_tq_drive)
        y_em_data.append(em_tq_drag)
    if rpm_em_map.size > 0:
        x_em_data.append(rpm_em_map)
        y_em_data.append(t_em_map)

    if x_em_data:
        x_em_all = np.concatenate(x_em_data)
        x_raw_min = float(np.nanmin(x_em_all))
        x_raw_max = float(np.nanmax(x_em_all))

        x_center = 0.5 * (x_raw_min + x_raw_max)
        x_width = max((x_raw_max - x_raw_min) + 240.0, 900.0)
        x_width = min(x_width, 2800.0)  # hard window [200, 3000]

        x_em_min = x_center - 0.5 * x_width
        x_em_max = x_center + 0.5 * x_width

        if x_em_min < 200.0:
            shift = 200.0 - x_em_min
            x_em_min += shift
            x_em_max += shift
        if x_em_max > 3000.0:
            shift = x_em_max - 3000.0
            x_em_min -= shift
            x_em_max -= shift

        x_em_min = max(200.0, x_em_min)
        x_em_max = min(3000.0, x_em_max)
        if x_em_max <= x_em_min:
            x_em_min, x_em_max = 200.0, 3000.0
    else:
        x_em_min, x_em_max = 200.0, 3000.0

    if y_em_data:
        y_em_all = np.concatenate(y_em_data)
        y_span_abs = max(
            float(np.abs(np.nanmin(y_em_all)) * 1.10),
            float(np.abs(np.nanmax(y_em_all)) * 1.10),
            300.0,
        )
        y_span_abs = min(y_span_abs, 1600.0)
        y_em_min, y_em_max = -y_span_abs, y_span_abs
    else:
        y_em_min, y_em_max = -1600.0, 1600.0

    ax_em.set_xlim(x_em_min, x_em_max)
    ax_em.set_ylim(y_em_min, y_em_max)
    ax_em.grid(True, linestyle=':', alpha=0.7)
    if em_rpm_lim.size > 0 or rpm_em_map.size > 0:
        ax_em.legend(
            loc='upper right',
            fontsize=8,
            framealpha=0.85,
            borderpad=0.3,
            labelspacing=0.25,
            handlelength=1.4,
            markerscale=0.85,
        )

    plt.tight_layout()
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    plot_filename = os.path.join(out_dir, f"{output_prefix}.pdf")
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig) # Close to release memory
    
    fuel_kg = total_fuel_g / 1000.0
    print(f"Saved {plot_filename} | Fuel: {fuel_kg:.3f} kg")
    
    return fuel_kg, results

def main():
    # Backward Comp for manual run
    run_ecms_simulation()

if __name__ == "__main__":
    main()
