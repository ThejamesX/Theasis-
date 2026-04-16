import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.tri as mtri
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, zoom
from matplotlib.colors import PowerNorm

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from dp_optimizer import DPOptimizer

def run_dp_simulation(cycle_file=None, bat_capacity_kwh=120.0, output_prefix='dp'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if cycle_file:
        vmod_path = cycle_file
    else:
        vmod_path = os.path.join(base_dir, "Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod")
        
    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    # print("Loading Components...")
    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)
    
    if bat_capacity_kwh is not None:
        truck.bat_params['Capacity'] = float(bat_capacity_kwh)
    
    # print("Loading Cycle...")
    cycle_df = loader.read_vmod(vmod_path)
    
    print(f"--- Running DP | Cap: {bat_capacity_kwh} kWh | Cycle: {os.path.basename(vmod_path)} ---")
    
    # UPDATED SETTINGS from User Request
    optimizer = DPOptimizer(truck, cycle_df, soc_grid_size=400, bat_capacity_kwh=bat_capacity_kwh)
    
    # print("Solving DP (Backward Sweep)...")
    # UPDATED SETTINGS from User Request (Target 0.51)
    J = optimizer.solve(start_soc=0.7, target_soc=0.3)
    
    # print("Reconstructing Optimal Path...")
    res = optimizer.reconstruct_path(start_soc=0.7)
    
    fuel_kg = res['total_fuel_kg']
    print(f"DP Result: {fuel_kg:.3f} kg")
    
    # Plotting (same 3-panel layout as strategy plots)
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 11})
    # Enable runtime percentage overlays (heatmap view instead of sparse points)
    show_ice_runtime_percentage_overlay = True

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)
    ax_soc = fig.add_subplot(gs[0, :])
    ax_ice = fig.add_subplot(gs[1, 0])
    ax_em = fig.add_subplot(gs[1, 1])

    # 1) SOC trajectory
    ax_soc.plot(res['time'], res['soc'] * 100, label='Stav nabití [%]', color='black', linewidth=2.0)
    if 'target_soc' in res:
        ax_soc.plot(res['time'], res['target_soc'] * 100, color='tab:red', linestyle='--', linewidth=1.5, label='Cílový stav nabití [%]')
    else:
        ax_soc.axhline(30.0, color='tab:red', linestyle='--', linewidth=1.5, label='Cílový stav nabití [%]')

    soc_time = np.asarray(res['time'])
    if soc_time.size > 1 and np.isfinite(soc_time).all():
        ax_soc.set_xlim(float(np.min(soc_time)), float(np.max(soc_time)))

    ax_soc.set_ylabel('Stav nabití [%]')
    ax_soc.set_xlabel('Čas [s]')
    ax_soc.set_title(f'Strategie: DP | Kapacita baterie: {bat_capacity_kwh} kWh | Palivo: {fuel_kg:.2f} kg', fontweight='bold')
    ax_soc.legend(loc='upper right')
    ax_soc.grid(True, linestyle=':', alpha=0.7)

    # 2) ICE load point map with BSFC background
    max_rpm = np.array([])
    max_tq = np.array([])
    bsfc_triang = None
    bsfc_vals = None
    try:
        df_vmap = pd.read_csv(vmap_path, sep=',', header=0)
        df_vmap.columns = ['rpm', 'torque', 'fuel']

        rpm_raw = df_vmap['rpm'].to_numpy()
        torque_raw = df_vmap['torque'].to_numpy()
        fuel_raw = df_vmap['fuel'].to_numpy()

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
            rpm_scale = max(1e-9, float(rpm_bsfc.max() - rpm_bsfc.min()))
            tq_scale = max(1e-9, float(tq_bsfc.max() - tq_bsfc.min()))
            rpm_norm = (rpm_bsfc - rpm_bsfc.min()) / rpm_scale
            tq_norm = (tq_bsfc - tq_bsfc.min()) / tq_scale

            tri_norm = mtri.Triangulation(rpm_norm, tq_norm)
            refiner = mtri.UniformTriRefiner(tri_norm)
            tri_fine, bsfc_fine = refiner.refine_field(bsfc, subdiv=3)

            rpm_fine = tri_fine.x * rpm_scale + rpm_bsfc.min()
            tq_fine = tri_fine.y * tq_scale + tq_bsfc.min()
            tri_phys = mtri.Triangulation(rpm_fine, tq_fine, tri_fine.triangles)

            if max_rpm.size > 1 and tri_phys.triangles.size > 0:
                c_rpm = np.mean(rpm_fine[tri_phys.triangles], axis=1)
                c_tq = np.mean(tq_fine[tri_phys.triangles], axis=1)
                tq_env = np.interp(np.clip(c_rpm, max_rpm.min(), max_rpm.max()), max_rpm, max_tq)
                tri_phys.set_mask((c_tq > tq_env) | (c_tq < 0.0))

            bsfc_triang = tri_phys
            bsfc_vals = bsfc_fine
    except Exception as e:
        print(f"Warning: Could not build DP ICE BSFC map: {e}")

    rpm_arr = cycle_df['rpm_ice'].to_numpy() if 'rpm_ice' in cycle_df.columns else np.array([])
    t_eng_arr = np.asarray(res.get('t_eng', []))

    if bsfc_triang is not None and bsfc_vals is not None and len(bsfc_vals) > 0:
        # Draw monochrome BSFC contour lines for context
        cs_bsfc = ax_ice.tricontour(
            bsfc_triang,
            bsfc_vals,
            levels=np.concatenate([np.arange(182, 210, 2), np.arange(210, 340, 10)]),
            colors='0.25',
            linewidths=0.4,
            alpha=0.35,
            zorder=4,
        )
        ax_ice.clabel(cs_bsfc, inline=True, fontsize=7, fmt='%d', colors='0.25')

    bsfc_handle = None
    if rpm_arr.size > 0 and t_eng_arr.size > 0:
        mask_ice = t_eng_arr > 0.0
        rpm_ice = rpm_arr[mask_ice]
        t_ice = t_eng_arr[mask_ice]

        if rpm_ice.size > 0:
            x_min_ice = float(min(rpm_ice.min(), max_rpm.min() if max_rpm.size > 0 else 400.0))
            x_max_ice = float(max(rpm_ice.max(), max_rpm.max() if max_rpm.size > 0 else 2500.0))
            y_max_ice = float(max_tq.max() * 1.05) if max_tq.size > 0 else float(max(200.0, np.max(t_ice) * 1.05))

            hist_ice, xedges_ice, yedges_ice = np.histogram2d(
                rpm_ice,
                t_ice,
                bins=50,
                range=[[x_min_ice, x_max_ice], [0, y_max_ice]],
            )
            if hist_ice.sum() > 0:
                hist_ice = (hist_ice / hist_ice.sum()) * 100.0
                hist_smooth = gaussian_filter(hist_ice, sigma=0.85) if gaussian_filter is not None else hist_ice
                hist_visible = np.ma.masked_less_equal(hist_smooth, 0.0)

                if show_ice_runtime_percentage_overlay and hist_visible.count() > 0:
                    positive_values = hist_smooth[hist_smooth > 0.0]
                    vmin = max(float(np.percentile(positive_values, 1)), 0.005)
                    vmax = float(np.nanpercentile(positive_values, 99.5))
                    if vmax <= vmin:
                        vmax = vmin * 1.01
                    if vmax > 0:
                        upsample = 4
                        smooth_fine = zoom(hist_smooth.T, zoom=upsample, order=3) if zoom is not None else hist_smooth.T
                        smooth_fine = np.ma.masked_less_equal(smooth_fine, 0.0)

                        x_fine = np.linspace(x_min_ice, x_max_ice, smooth_fine.shape[1])
                        y_fine = np.linspace(0.0, y_max_ice, smooth_fine.shape[0])

                        if max_rpm.size > 1 and max_tq.size > 1:
                            max_tq_interp = np.interp(x_fine, max_rpm, max_tq, left=np.nan, right=np.nan)
                            envelope_mask = y_fine[:, None] > max_tq_interp[None, :]
                            envelope_mask |= ~np.isfinite(max_tq_interp)[None, :]
                            smooth_fine = np.ma.masked_array(smooth_fine, mask=np.ma.getmaskarray(smooth_fine) | envelope_mask)

                        bsfc_handle = ax_ice.imshow(
                            smooth_fine,
                            origin='lower',
                            extent=[x_min_ice, x_max_ice, 0, y_max_ice*1.04],
                            cmap='Blues',
                            norm=PowerNorm(gamma=0.27, vmin=vmin, vmax=vmax),
                            interpolation='bicubic',
                            alpha=1,
                            aspect='auto',
                            zorder=1,
                        )

            if not show_ice_runtime_percentage_overlay and rpm_ice.size > 0:
                stride_ice = max(1, rpm_ice.size // 1400)
                ax_ice.scatter(
                    rpm_ice[::stride_ice],
                    t_ice[::stride_ice],
                    s=10,
                    c='black',
                    alpha=0.24,
                    edgecolors='white',
                    linewidths=0.18,
                    zorder=4,
                    label='Provozní body',
                )

    if bsfc_handle is not None:
        fig.colorbar(bsfc_handle, ax=ax_ice, label='Časový podíl [%]')

    if max_rpm.size > 0:
        ax_ice.plot(max_rpm, max_tq, 'k-', linewidth=3.0, label='Křivka max. točivého momentu', zorder=5)

    x_limit_data = []
    if max_rpm.size > 0:
        x_limit_data.append(max_rpm)
    if rpm_arr.size > 0 and t_eng_arr.size > 0:
        mask_ice = t_eng_arr > 0.0
        if np.any(mask_ice):
            x_limit_data.append(rpm_arr[mask_ice])

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
    if t_eng_arr.size > 0:
        mask_ice = t_eng_arr > 0.0
        if np.any(mask_ice):
            y_top_candidates.append(float(np.nanmax(t_eng_arr[mask_ice]) * 1.10))
    ice_y_top = max(400.0, *y_top_candidates) if y_top_candidates else 3000.0

    ax_ice.set_title('Mapa provozních bodů spalovacího motoru', fontweight='bold')
    ax_ice.set_xlabel('Otáčky motoru [1/min]')
    ax_ice.set_ylabel('Točivý moment motoru [Nm]')
    ax_ice.set_xlim(x_min, x_max)
    ax_ice.set_ylim(0.0, ice_y_top)
    ax_ice.grid(False)
    if max_rpm.size > 0 or (rpm_arr.size > 0 and t_eng_arr.size > 0):
        ax_ice.legend(loc='upper right', fontsize=8, framealpha=0.85, borderpad=0.3, labelspacing=0.25, handlelength=1.4, markerscale=0.85)

    # 3) E-Motor load point map with efficiency background
    em_rpm_lim = np.array([])
    em_tq_drive = np.array([])
    em_tq_drag = np.array([])
    em_eff_triang = None
    em_eff_vals = None
    try:
        df_vem = pd.read_csv(os.path.join(base_dir, 'Emotor', 'EM_fld.vemp'), skipinitialspace=True)
        em_rpm_lim = df_vem.iloc[:, 0].to_numpy()
        em_tq_drive = df_vem.iloc[:, 1].to_numpy()
        em_tq_drag = df_vem.iloc[:, 2].to_numpy()

        df_em_map = pd.read_csv(vemo_path, skipinitialspace=True)
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
                    drive_interp = interp1d(em_rpm_lim, em_tq_drive, kind='linear', bounds_error=False, fill_value='extrapolate')
                    drag_interp = interp1d(em_rpm_lim, em_tq_drag, kind='linear', bounds_error=False, fill_value='extrapolate')
                    c_rpm = np.mean(rpm_fine[tri_phys.triangles], axis=1)
                    c_tq = np.mean(tq_fine[tri_phys.triangles], axis=1)
                    drive_env = drive_interp(np.clip(c_rpm, em_rpm_lim.min(), em_rpm_lim.max()))
                    drag_env = drag_interp(np.clip(c_rpm, em_rpm_lim.min(), em_rpm_lim.max()))
                    tri_phys.set_mask((c_tq > drive_env) | (c_tq < drag_env))

                em_eff_triang = tri_phys
                em_eff_vals = eff_fine
    except Exception as e:
        print(f"Warning: Could not build DP E-Motor efficiency map: {e}")

    t_mot_arr = np.asarray(res.get('t_mot', []))
    # Efficiency contour background: keep monochrome contour lines and overlay runtime heatmap in dark green
    if em_eff_triang is not None and em_eff_vals is not None and len(em_eff_vals) > 0:
        cs_eff = ax_em.tricontour(
            em_eff_triang,
            em_eff_vals,
            levels=np.arange(60, 100, 2),
            colors='0.25',
            linewidths=0.4,
            alpha=0.35,
            zorder=4,
        )
        ax_em.clabel(cs_eff, inline=True, fontsize=7, fmt='%d', colors='0.25')

    em_handle = None
    if rpm_arr.size > 0 and t_mot_arr.size > 0:
        mask_em = np.abs(t_mot_arr) > 0.5
        rpm_em = rpm_arr[mask_em]
        t_em = t_mot_arr[mask_em]

        if rpm_em.size > 0:
            x_min_em = float(min(rpm_em.min(), em_rpm_lim.min() if em_rpm_lim.size > 0 else 0.0))
            x_max_em = float(max(rpm_em.max(), em_rpm_lim.max() if em_rpm_lim.size > 0 else 3000.0))
            y_min_em = float(min(t_em.min(), em_tq_drag.min() if em_tq_drag.size > 0 else -1500.0))
            y_max_em = float(max(t_em.max(), em_tq_drive.max() if em_tq_drive.size > 0 else 1500.0))

            hist_em, xedges_em, yedges_em = np.histogram2d(
                rpm_em,
                t_em,
                bins=50,
                range=[[x_min_em, x_max_em], [y_min_em, y_max_em]],
            )
            if hist_em.sum() > 0:
                hist_em = (hist_em / hist_em.sum()) * 100.0
                hist_smooth_em = gaussian_filter(hist_em, sigma=0.85) if gaussian_filter is not None else hist_em
                hist_visible_em = np.ma.masked_less_equal(hist_smooth_em, 0.0)

                if show_ice_runtime_percentage_overlay and hist_visible_em.count() > 0:
                    positive_values = hist_smooth_em[hist_smooth_em > 0.0]
                    vmin = max(float(np.percentile(positive_values, 1)), 0.009)
                    vmax = float(np.nanpercentile(positive_values, 99.5))
                    if vmax <= vmin:
                        vmax = vmin * 1.01
                    if vmax > 0:
                        smooth_fine_em = zoom(hist_smooth_em.T, zoom=3, order=2) if zoom is not None else hist_smooth_em.T
                        smooth_fine_em = np.ma.masked_less_equal(smooth_fine_em, 0.0)

                        x_fine_em = np.linspace(x_min_em, x_max_em, smooth_fine_em.shape[1])
                        y_fine_em = np.linspace(y_min_em, y_max_em, smooth_fine_em.shape[0])

                        if em_rpm_lim.size > 1 and em_tq_drive.size > 1 and em_tq_drag.size > 1:
                            max_tq_interp_drive = np.interp(x_fine_em, em_rpm_lim, em_tq_drive, left=np.nan, right=np.nan)
                            max_tq_interp_drag = np.interp(x_fine_em, em_rpm_lim, em_tq_drag, left=np.nan, right=np.nan)
                            envelope_mask_em = (y_fine_em[:, None] > max_tq_interp_drive[None, :]) | (y_fine_em[:, None] < max_tq_interp_drag[None, :])
                            envelope_mask_em |= ~np.isfinite(max_tq_interp_drive)[None, :]
                            envelope_mask_em |= ~np.isfinite(max_tq_interp_drag)[None, :]
                            smooth_fine_em = np.ma.masked_array(smooth_fine_em, mask=np.ma.getmaskarray(smooth_fine_em) | envelope_mask_em)
                            em_handle = ax_em.imshow(
                            smooth_fine_em,
                            origin='lower',
                            extent=[x_min_em, x_max_em, y_min_em*1.04, y_max_em*1.04],
                            cmap='Greens',
                            norm=PowerNorm(gamma=0.27, vmin=vmin, vmax=vmax),
                            interpolation='bicubic',
                            alpha=1,
                            aspect='auto',
                            zorder=1,
                        )

            if not show_ice_runtime_percentage_overlay and rpm_em.size > 0:
                stride_em = max(1, rpm_em.size // 1400)
                ax_em.scatter(
                    rpm_em[::stride_em],
                    t_em[::stride_em],
                    s=10,
                    c='black',
                    alpha=0.22,
                    edgecolors='white',
                    linewidths=0.18,
                    zorder=3,
                    label='Provozní body',
                )

    if em_handle is not None:
        fig.colorbar(em_handle, ax=ax_em, label='Časový podíl [%]')

    if em_rpm_lim.size > 0:
        ax_em.plot(em_rpm_lim, em_tq_drive, 'k-', linewidth=2.5, label='Max. hnací moment')
        ax_em.plot(em_rpm_lim, em_tq_drag, 'k--', linewidth=2.5, label='Max. rekuperační moment')

    ax_em.set_title('Mapa provozních bodů elektromotoru', fontweight='bold')
    ax_em.set_xlabel('Otáčky elektromotoru [1/min]')
    ax_em.set_ylabel('Točivý moment elektromotoru [Nm]')
    ax_em.axhline(0, color='gray', linewidth=1)

    # Center EM map viewport around active data.
    x_em_data = []
    y_em_data = []
    if em_rpm_lim.size > 0:
        x_em_data.append(em_rpm_lim)
        y_em_data.append(em_tq_drive)
        y_em_data.append(em_tq_drag)
    if rpm_arr.size > 0 and t_mot_arr.size > 0:
        mask_em = np.abs(t_mot_arr) > 0.5
        if np.any(mask_em):
            x_em_data.append(rpm_arr[mask_em])
            y_em_data.append(t_mot_arr[mask_em])

    if x_em_data:
        x_em_all = np.concatenate(x_em_data)
        x_raw_min = float(np.nanmin(x_em_all))
        x_raw_max = float(np.nanmax(x_em_all))

        x_center = 0.5 * (x_raw_min + x_raw_max)
        x_width = max((x_raw_max - x_raw_min) + 240.0, 900.0)
        x_width = min(x_width, 2800.0)

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
    ax_em.grid(False)
    if em_rpm_lim.size > 0 or (rpm_arr.size > 0 and t_mot_arr.size > 0):
        ax_em.legend(loc='upper right', fontsize=8, framealpha=0.85, borderpad=0.3, labelspacing=0.25, handlelength=1.4, markerscale=0.85)

    plt.tight_layout()
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    plot_filename = os.path.join(out_dir, f"{output_prefix}.pdf")
    
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    
    return fuel_kg, res

def main():
    run_dp_simulation()
