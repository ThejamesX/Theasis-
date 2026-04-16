import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import PowerNorm
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from main import run_ecms_simulation
from run_dp import run_dp_simulation

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cycle_dir = os.path.join(base_dir, "Driving Cycle")
    
    # Identify Cycles
    # Expecting:
    # 1. Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod
    # 2. Class5_Tractor_DECL_RegionalDeliveryEMSReferenceLoad.vmod
    
    cycles = []
    if os.path.exists(cycle_dir):
        for f in os.listdir(cycle_dir):
            if f.endswith(".vmod"):
                cycles.append(os.path.join(cycle_dir, f))
    
    if not cycles:
        print("No cycles found in Driving Cycle/")
        return
        
    print(f"Found {len(cycles)} cycles.")
    
    params_cap = [120.0]
    plot_standalone_em_maps = False
    
    # Note: main.py expects 'A-ECMS' instead of 'AECMS'
    strategies_ecms = ['AECMS'] #'ECMS', 'AECMS', 'PECMS'
    
    results = []
    out_dir = os.path.join(base_dir, 'output')
    os.makedirs(out_dir, exist_ok=True)
    
    # Preconfigure matplotlib academic styling globally
    plt.rcParams.update({'font.size': 12, 'axes.labelsize': 12, 'legend.fontsize': 11})
    for cycle_path in cycles:
        cycle_name = os.path.basename(cycle_path).replace(".vmod", "")
        print(f"\n=== Processing Cycle: {cycle_name} ===\n")
        
        # Dictionary to store trajectory data for combined plotting
        # Plot drive cycle and torque request once per cycle
        cycle_plotted = False
        
        for cap in params_cap:
            
            # Dictionary to store trajectory data for combined plotting
            plot_data = {}
            
            # 1. DP
            prefix_dp = f"{cycle_name}_DP_{int(cap)}kWh"
            print(f"-> Strategy: DP, Cap: {cap}")
            fuel_dp, res_dp = run_dp_simulation(cycle_file=cycle_path, bat_capacity_kwh=cap, output_prefix=prefix_dp)
            results.append({
                'Cycle': cycle_name,
                'Capacity_kWh': cap,
                'Strategy': 'DP',
                'Fuel_kg': fuel_dp
            })
            
            plot_data['DP'] = {
                'time': res_dp['time'],
                'soc': res_dp['soc'] * 100,
                'fuel': fuel_dp
            }
            if 'target_soc' in res_dp:
                plot_data['Target'] = {
                    'time': res_dp['time'],
                    'soc': res_dp['target_soc'] * 100
                }
            
            # 2. ECMS Variants
            for strat in strategies_ecms:
                prefix = f"{cycle_name}_{strat}_{int(cap)}kWh"
                print(f"-> Strategy: {strat}, Cap: {cap}")
                fuel, res_ecms = run_ecms_simulation(strategy=strat, cycle_file=cycle_path, bat_capacity_kwh=cap, output_prefix=prefix)
                
                results.append({
                    'Cycle': cycle_name,
                    'Capacity_kWh': cap,
                    'Strategy': strat,
                    'Fuel_kg': fuel
                })
                
                plot_data[strat] = {
                    'time': res_ecms['time'],
                    'soc': np.array(res_ecms['soc']) * 100,
                    'fuel': fuel,
                    's_factor': np.array(res_ecms.get('s_factor', [])),
                    'rpm_ice': np.array(res_ecms.get('rpm_ice', [])),
                    't_ice': np.array(res_ecms.get('t_ice', [])),
                    't_em': np.array(res_ecms.get('t_em', [])),
                    'velocity_kmh': np.array(res_ecms.get('velocity_kmh', [])),
                    't_req': np.array(res_ecms.get('t_req', []))
                }
                
                # Capture reference target if not yet taken
                if 'Target' not in plot_data and 'soc_target' in res_ecms:
                    plot_data['Target'] = {
                        'time': res_ecms['time'],
                        'soc': np.array(res_ecms['soc_target']) * 100
                    }
                    
                # Plot Driving Cycle and Torque Request ONCE per cycle
                if not cycle_plotted and 'velocity_kmh' in res_ecms:
                    fig_dc, axes_dc = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                    
                    # 1. Drive Cycle Speed & Altitude
                    axes_dc[0].plot(res_ecms['time'], res_ecms['velocity_kmh'], label='Rychlost [km/h]', color='black', linewidth=1.5)
                    axes_dc[0].set_ylabel('Rychlost [km/h]', fontweight='bold')
                    axes_dc[0].set_title(f'Jízdní cyklus: {cycle_name.replace("_", " ")}', fontweight='bold')
                    axes_dc[0].grid(True, linestyle=':', alpha=0.7)
                    
                    if 'altitude_m' in res_ecms and np.any(res_ecms['altitude_m']):
                        ax0_alt = axes_dc[0].twinx()
                        ax0_alt.plot(res_ecms['time'], res_ecms['altitude_m'], label='Nadmořská výška [m]', color='gray', alpha=0.6, linestyle='--')
                        ax0_alt.set_ylabel('Nadmořská výška [m]', color='gray', fontweight='bold')
                        
                    # 2. Torque Request
                    if 't_req' in res_ecms:
                        axes_dc[1].plot(res_ecms['time'], res_ecms['t_req'], label='Požadovaný točivý moment vozidla [Nm]', color='tab:blue', linewidth=1)
                        axes_dc[1].set_ylabel('Požadovaný točivý moment [Nm]', fontweight='bold')
                        axes_dc[1].set_xlabel('Čas [s]', fontweight='bold')
                        axes_dc[1].grid(True, linestyle=':', alpha=0.7)
                        axes_dc[1].legend(loc='upper right')
                        
                    plt.tight_layout()
                    cycle_plot_name = os.path.join(out_dir, f"DriveCycle_Torque_{cycle_name}.pdf")
                    plt.savefig(cycle_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig_dc)
                    print(f"Saved Drive Cycle plot: {cycle_plot_name}")
                    cycle_plotted = True

            # --- Combined Plotting per Cycle and Capacity ---
            plt.figure(figsize=(12, 6))
            
            # Define specific colors
            colors = {'DP': 'black', 'ECMS': 'tab:blue', 'AECMS': 'tab:orange', 'PECMS': 'tab:green'}
            
            # Plot each strategy
            for p_strat, data in plot_data.items():
                if p_strat != 'Target':
                    plt.plot(data['time'], data['soc'], label=f"{p_strat} (Palivo: {data['fuel']:.2f} kg)", 
                             color=colors.get(p_strat, 'gray'), linewidth=2.5 if p_strat == 'DP' else 1.5, alpha=0.9 if p_strat == 'DP' else 0.8)
            
            # Plot reference Target line
            if 'Target' in plot_data:
                plt.plot(plot_data['Target']['time'], plot_data['Target']['soc'], color='black', linestyle='--', linewidth=2, alpha=0.8, label='Referenční cíl')
            
            plt.title(f'Porovnání trajektorií SOC | {cycle_name.replace("_", " ")} | Kapacita baterie: {cap} kWh', fontweight='bold')
            plt.xlabel('Čas [s]', fontweight='bold')
            plt.ylabel('Stav nabití (SOC) [%]', fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            combined_plot_name = os.path.join(out_dir, f"Combined_SOC_{cycle_name}_{int(cap)}kWh.pdf")
            plt.savefig(combined_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
            print(f"Saved combined plot: {combined_plot_name}")
            
            # --- 1) EF Comparison Plot (AECMS vs PECMS) ---
            if 'AECMS' in plot_data and 'PECMS' in plot_data:
                fig_ef, ax_ef = plt.subplots(figsize=(10, 5))
                ax_ef.plot(plot_data['AECMS']['time'], plot_data['AECMS']['s_factor'], label='Ekvivalenční faktor A-ECMS', color='tab:orange', linewidth=1.5)
                ax_ef.plot(plot_data['PECMS']['time'], plot_data['PECMS']['s_factor'], label='Ekvivalenční faktor P-ECMS', color='tab:green', linewidth=2.0)
                cycle_title = cycle_name.replace("_", " ").replace("EMSReferenceLoad", "").strip()
                ax_ef.set_title(f'Porovnání ekvivalenčního faktoru (EF) | {cycle_title} | {int(cap)} kWh', fontweight='bold')
                ax_ef.set_xlabel('Čas [s]', fontweight='bold')
                ax_ef.set_ylabel('Ekvivalenční faktor (s)', fontweight='bold')
                ax_ef.grid(True, linestyle=':', alpha=0.7)
                ax_ef.legend(loc='upper right')
                plt.tight_layout()
                ef_plot_name = os.path.join(out_dir, f"EF_Comparison_{cycle_name}_{int(cap)}kWh.pdf")
                plt.savefig(ef_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig_ef)
                print(f"Saved EF Comparison plot: {ef_plot_name}")

            # --- 2) Engine Map Plot (ICE-only vs AECMS/PECMS) ---
            engine_strat = 'AECMS' if 'AECMS' in plot_data else 'PECMS'
            if engine_strat in plot_data:
                fig_map, axes_map = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                # Show runtime percentage overlay instead of sparse runtime points for the ICE comparison.
                show_ice_runtime_percentage_overlay = True
                
                # Fetch BSFC data and maximum torque curve
                try:
                    df_vmap = pd.read_csv("Engine/325kW.vmap", sep=',', header=0)
                    df_vmap.columns = ['rpm', 'torque', 'fuel']
                    
                    rpm_raw = df_vmap['rpm'].values
                    torque_raw = df_vmap['torque'].values
                    fuel_raw = df_vmap['fuel'].values

                    # BSFC [g/kWh]
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
                    
                    # Same logic as plot_vecto_components.py for smooth max torque line
                    drpm = np.diff(rpm_raw)
                    block_ends = np.where(np.abs(drpm) > 5.0)[0]
                    block_ends = np.append(block_ends, len(rpm_raw)-1)

                    max_tq_curve = []
                    for k_idx, end_idx in enumerate(block_ends):
                        start_idx = 0 if k_idx == 0 else block_ends[k_idx-1] + 1
                        rp_block = rpm_raw[start_idx:end_idx+1]
                        tq_block = torque_raw[start_idx:end_idx+1]
                        if tq_block.max() > 0:
                            max_tq_curve.append((rp_block[np.argmax(tq_block)], tq_block.max()))

                    if max_tq_curve:
                        max_tq_curve = np.array(max_tq_curve)
                        max_rpm = max_tq_curve[:, 0]
                        max_tq = max_tq_curve[:, 1]
                    else:
                        max_rpm, max_tq = np.array([]), np.array([])

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
                    triang = mtri.Triangulation(rpm_fine, tq_fine, triang_fine.triangles)

                    if len(max_rpm) > 1 and triang.triangles.size > 0:
                        centroids_rpm = np.mean(rpm_fine[triang.triangles], axis=1)
                        centroids_tq = np.mean(tq_fine[triang.triangles], axis=1)
                        tq_env = np.interp(
                            np.clip(centroids_rpm, max_rpm.min(), max_rpm.max()),
                            max_rpm,
                            max_tq,
                        )
                        # Strictly enforce max torque envelope.
                        triang.set_mask((centroids_tq > tq_env) | (centroids_tq < 0.0))

                    bsfc = bsfc_fine
                except Exception as e:
                    print("Warning: Could not load engine BSFC map:", e)
                    triang, bsfc = None, None
                    max_rpm, max_tq = [], []

                strat_data = plot_data[engine_strat]
                rpm = strat_data['rpm_ice']
                t_req = strat_data['t_req']
                t_ice_strat = strat_data['t_ice']
                
                # ICE-only: Engine is driven exactly by T_req -> filter positive requests
                mask_ice = (t_req > 0)
                rpm_ice_only = rpm[mask_ice]
                t_ice_only = t_req[mask_ice]
                
                # Active strategy: moments where ICE was providing positive torque
                mask_strat = (t_ice_strat > 0)
                rpm_strat = rpm[mask_strat]
                t_ice_act = t_ice_strat[mask_strat]

                from scipy.ndimage import gaussian_filter
                
                # Setup grids for smooth density contour
                bins = 50

                x_limit_data = []
                if len(max_rpm) > 0:
                    x_limit_data.append(np.asarray(max_rpm))
                if len(rpm_ice_only) > 0:
                    x_limit_data.append(np.asarray(rpm_ice_only))
                if len(rpm_strat) > 0:
                    x_limit_data.append(np.asarray(rpm_strat))

                if x_limit_data:
                    x_all = np.concatenate(x_limit_data)
                    x_min = float(np.nanmin(x_all))
                    x_max = float(np.nanmax(x_all))
                    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                        x_min, x_max = 0.0, 2500.0
                else:
                    x_min, x_max = 0.0, 2500.0

                y_top_candidates = []
                if len(max_tq) > 0:
                    y_top_candidates.append(float(np.nanmax(max_tq) * 1.05))
                if len(t_ice_only) > 0:
                    y_top_candidates.append(float(np.nanmax(t_ice_only) * 1.10))
                if len(t_ice_act) > 0:
                    y_top_candidates.append(float(np.nanmax(t_ice_act) * 1.10))
                y_top = max(400.0, *y_top_candidates) if y_top_candidates else 3000.0

                def plot_runtime_on_bsfc(ax, rpm_points, tq_points, title):
                    bsfc_handle = None

                    # 1) BSFC contour lines (monochrome only)
                    if triang is not None and bsfc is not None and len(bsfc) > 0:
                        levels_bsfc = np.concatenate([np.arange(182, 210, 2), np.arange(210, 340, 10)])
                        cs_bsfc = ax.tricontour(
                            triang,
                            bsfc,
                            levels=levels_bsfc,
                            colors='0.25',
                            linewidths=0.4,
                            alpha=0.35,
                            zorder=4,
                        )
                        ax.clabel(cs_bsfc, inline=True, fontsize=7, fmt='%d', colors='0.25')

                    # 2) Runtime density + load points overlay
                    if len(rpm_points) > 0:
                        hist, xedges, yedges = np.histogram2d(
                            rpm_points,
                            tq_points,
                            bins=bins,
                            range=[[x_min, x_max], [0, y_top]],
                        )
                        if hist.sum() > 0:
                            hist = (hist / hist.sum()) * 100.0
                            hist_smooth = gaussian_filter(hist, sigma=0.95)
                            hist_visible = np.ma.masked_less_equal(hist_smooth, 0.0)

                            X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
                            if show_ice_runtime_percentage_overlay and hist_visible.count() > 0:
                                positive_values = hist_smooth[hist_smooth > 0.0]
                                vmin = max(float(np.percentile(positive_values, 1)), 0.001)
                                vmax = float(np.nanpercentile(positive_values, 99.5))
                                if vmax <= vmin:
                                    vmax = vmin * 1.01
                                if vmax > 0:
                                    upsample = 8
                                    smooth_fine = zoom(hist_smooth.T, zoom=upsample, order=3)
                                    smooth_fine = np.ma.masked_less_equal(smooth_fine, 0.0)

                                    x_fine = np.linspace(x_min, x_max, smooth_fine.shape[1])
                                    y_fine = np.linspace(0.0, y_top, smooth_fine.shape[0])

                                    if len(max_rpm) > 1 and len(max_tq) > 1:
                                        max_tq_interp = np.interp(x_fine, max_rpm, max_tq, left=np.nan, right=np.nan)
                                        envelope_mask = y_fine[:, None] > max_tq_interp[None, :]
                                        envelope_mask |= ~np.isfinite(max_tq_interp)[None, :]
                                        smooth_fine = np.ma.masked_array(smooth_fine, mask=np.ma.getmaskarray(smooth_fine) | envelope_mask)

                                    bsfc_handle = ax.imshow(
                                        smooth_fine,
                                        origin='lower',
                                        extent=[x_min, x_max, 0, y_top],
                                        cmap='Blues',
                                        norm=PowerNorm(gamma=0.23, vmin=vmin, vmax=vmax),
                                        interpolation='bicubic',
                                        alpha=1,
                                        aspect='auto',
                                        zorder=1,
                                    )
                                    

                        if not show_ice_runtime_percentage_overlay:
                            # Sparse points make true operating traces explicit.
                            stride = max(1, len(rpm_points) // 1400)
                            ax.scatter(
                                rpm_points[::stride],
                                tq_points[::stride],
                                s=15,
                                c='black',
                                alpha=0.15,
                                edgecolors='white',
                                linewidths=0.18,
                                zorder=4,
                                label='Provozní body',
                            )

                    if len(max_rpm) > 0:
                        ax.plot(max_rpm, max_tq, 'k-', linewidth=2.8, label='Křivka max. točivého momentu', zorder=5)

                    ax.set_title(title, fontweight='bold')
                    ax.set_xlabel('Otáčky motoru [1/min]')
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(0.0, y_top)
                    ax.grid(True, linestyle=':', alpha=0.7)
                    if not show_ice_runtime_percentage_overlay:
                        ax.legend(
                            loc='upper left',
                            fontsize=8,
                            framealpha=0.85,
                            borderpad=0.3,
                            labelspacing=0.25,
                            handlelength=1.4,
                            markerscale=0.85,
                        )

                    return bsfc_handle
                
                # (a) ICE-only Map
                ax1 = axes_map[0]
                bsfc_handle = plot_runtime_on_bsfc(ax1, rpm_ice_only, t_ice_only, '(a) Pouze ICE na mapě BSFC')
                ax1.set_ylabel('Točivý moment motoru [Nm]')
                
                # (b) Selected strategy map
                ax2 = axes_map[1]
                bsfc_handle = plot_runtime_on_bsfc(ax2, rpm_strat, t_ice_act, f'(b) {engine_strat} na mapě BSFC')

                if bsfc_handle is not None:
                    cbar_bsfc = fig_map.colorbar(bsfc_handle, ax=axes_map, pad=0.02)
                    if show_ice_runtime_percentage_overlay:
                        cbar_bsfc.set_label('Time share / %')
                    else:
                        cbar_bsfc.set_label('Měrná spotřeba paliva [g/kWh]')
                
                plt.tight_layout()
                map_plot_name = os.path.join(out_dir, f"EngineMap_Comparison_{engine_strat}_{cycle_name}_{int(cap)}kWh.pdf")
                plt.savefig(map_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig_map)
                print(f"Saved Engine Map Plot: {map_plot_name}")
                
            # --- 3) E-Motor Map Plot (standalone export) ---
            if plot_standalone_em_maps:
                for strat in plot_data.keys():
                    if strat not in ['AECMS', 'PECMS', 'ECMS']:
                        continue
                    fig_em, ax_em = plt.subplots(figsize=(8, 6))
                    em_eff_triang = None
                    em_eff_vals = None
                    
                    try:
                        df_vem = pd.read_csv("Emotor/EM_fld.vemp", skipinitialspace=True)
                        em_rpm_lim = df_vem.iloc[:, 0].values
                        em_tq_drive = df_vem.iloc[:, 1].values
                        em_tq_drag = df_vem.iloc[:, 2].values

                        # E-Motor efficiency map background from vemo file.
                        df_em_map = pd.read_csv("Emotor/EM_Map - kopie.vemo", skipinitialspace=True)
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
                            if np.sum(valid_eff) > 3 and len(em_rpm_lim) > 1:
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
                        print("Warning: Could not load EM limits:", e)
                        em_rpm_lim, em_tq_drive, em_tq_drag = [], [], []

                    em_data = plot_data[strat]
                    rpm_em = em_data['rpm_ice']
                    t_em_act = em_data['t_em']

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
                        fig_em.colorbar(cf_eff, ax=ax_em, label='Účinnost [%]')
                    
                    # Filter points where EM is active
                    mask_em = (np.abs(t_em_act) > 0.5)
                    if np.sum(mask_em) > 0:
                        rpm_em_active = rpm_em[mask_em]
                        t_em_active = t_em_act[mask_em]

                        # Point-based runtime view for E-Motor map.
                        stride_em = max(1, len(rpm_em_active) // 1400)
                        ax_em.scatter(
                            rpm_em_active[::stride_em],
                            t_em_active[::stride_em],
                            s=10,
                            c='black',
                            alpha=0.22,
                            edgecolors='white',
                            linewidths=0.18,
                            zorder=3,
                            label='Provozní body',
                        )

                        # Intentionally kept (commented) so runtime-percentage overlay can be re-enabled later.
                        # from scipy.ndimage import gaussian_filter
                        # bins = 50
                        # hist_em, xedges_em, yedges_em = np.histogram2d(
                        #     rpm_em_active, t_em_active, bins=bins,
                        #     range=[[0, 3000], [-1500, 1500]]
                        # )
                        # hist_em = (hist_em / hist_em.sum()) * 100.0
                        # hist_em_smooth = gaussian_filter(hist_em, sigma=1.0)
                        # hist_em_smooth[hist_em_smooth < 0.05] = np.nan
                        # X_em, Y_em = np.meshgrid(xedges_em[:-1], yedges_em[:-1])
                        # cf_em = ax_em.contourf(X_em, Y_em, hist_em_smooth.T, cmap='Greens', levels=15, alpha=0.8)
                        # cb_em = fig_em.colorbar(cf_em, ax=ax_em, label='Time share / %')
                        
                    if len(em_rpm_lim) > 0:
                        ax_em.plot(em_rpm_lim, em_tq_drive, 'k-', linewidth=2.5, label='Max. hnací moment')
                        ax_em.plot(em_rpm_lim, em_tq_drag, 'k--', linewidth=2.5, label='Max. rekuperační moment')
                        
                    ax_em.set_title(f'Provozní mapa elektromotoru ({strat})', fontweight='bold')
                    ax_em.set_xlabel('Otáčky elektromotoru [1/min]')
                    ax_em.set_ylabel('Točivý moment elektromotoru [Nm]')
                    ax_em.axhline(0, color='gray', linewidth=1)

                    # Zoom to effective EM limits (runtime + envelope).
                    x_em_data = []
                    y_em_data = []
                    if len(em_rpm_lim) > 0:
                        x_em_data.append(np.asarray(em_rpm_lim))
                        y_em_data.append(np.asarray(em_tq_drive))
                        y_em_data.append(np.asarray(em_tq_drag))
                    if np.sum(mask_em) > 0:
                        x_em_data.append(np.asarray(rpm_em_active))
                        y_em_data.append(np.asarray(t_em_active))

                    if x_em_data:
                        x_em_all = np.concatenate(x_em_data)
                        x_em_min = max(200.0, float(np.nanmin(x_em_all) - 120.0))
                        x_em_max = min(3000.0, float(np.nanmax(x_em_all) + 120.0))
                        if (x_em_max - x_em_min) < 700.0:
                            x_em_mid = 0.5 * (x_em_min + x_em_max)
                            x_em_min = max(200.0, x_em_mid - 350.0)
                            x_em_max = min(3000.0, x_em_mid + 350.0)
                    else:
                        x_em_min, x_em_max = 200.0, 3000.0

                    if y_em_data:
                        y_em_all = np.concatenate(y_em_data)
                        y_em_min = min(-1600.0, float(np.nanmin(y_em_all) * 1.10))
                        y_em_max = max(300.0, float(np.nanmax(y_em_all) * 1.10))
                    else:
                        y_em_min, y_em_max = -1600.0, 1600.0

                    ax_em.set_xlim(x_em_min, x_em_max)
                    ax_em.set_ylim(y_em_min, y_em_max)
                    ax_em.grid(True, linestyle=':', alpha=0.7)
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
                    em_plot_name = os.path.join(out_dir, f"EMotorMap_{strat}_{cycle_name}_{int(cap)}kWh.pdf")
                    plt.savefig(em_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig_em)
                    print(f"Saved E-Motor Map Plot: {em_plot_name}")
            
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "batch_results.csv"), index=False)
            
    print("\n\n=== BATCH SIMULATION COMPLETE ===")
    print(df_res)
    print("Results saved to batch_results.csv")

if __name__ == "__main__":
    main()
