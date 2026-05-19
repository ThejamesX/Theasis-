import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.cm import ScalarMappable
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, zoom
from main import run_ecms_simulation
from run_dp import run_dp_simulation


def get_cycle_display_name(cycle_name):
    if 'LongHaulEMSReferenceLoad' in cycle_name:
        return 'Dálková přeprava'
    if 'RegionalDeliveryEMSReferenceLoad' in cycle_name:
        return 'Regionální rozvoz'
    return cycle_name.replace('_', ' ')


def get_cycle_length_km(cycle_name):
    if 'LongHaulEMSReferenceLoad' in cycle_name:
        return 100.2
    if 'RegionalDeliveryEMSReferenceLoad' in cycle_name:
        return 100.0
    return None

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
    export_drive_cycle_plot = True
    
    # Generate both offline and online strategy comparisons.
    strategies_ecms = ['ECMS', 'AECMS', 'PECMS']
    
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
                'fuel': fuel_dp,
                'rpm_ice': np.array(res_dp.get('rpm_ice', [])),
                't_ice': np.array(res_dp.get('t_eng', [])),
                't_em': np.array(res_dp.get('t_mot', []))
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
                    'soc_target': np.array(res_ecms.get('soc_target', [])) * 100,
                    'fuel': fuel,
                    's_factor': np.array(res_ecms.get('s_factor', [])),
                    'rpm_ice': np.array(res_ecms.get('rpm_ice', [])),
                    't_ice': np.array(res_ecms.get('t_ice', [])),
                    't_em': np.array(res_ecms.get('t_em', [])),
                    'velocity_kmh': np.array(res_ecms.get('velocity_kmh', [])),
                    'altitude_m': np.array(res_ecms.get('altitude_m', [])),
                    't_req': np.array(res_ecms.get('t_req', []))
                }
                
                # Capture reference target only from adaptive/predictive strategies.
                if 'Target' not in plot_data and strat in ['AECMS', 'A-ECMS', 'PECMS'] and 'soc_target' in res_ecms:
                    plot_data['Target'] = {
                        'time': res_ecms['time'],
                        'soc': np.array(res_ecms['soc_target']) * 100
                    }
                    
                # Plot Driving Cycle and Torque Request ONCE per cycle
                if export_drive_cycle_plot and (not cycle_plotted) and 'velocity_kmh' in res_ecms:
                    fig_dc, axes_dc = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    cycle_length_km = get_cycle_length_km(cycle_name)
                    cycle_length_text = f' ({cycle_length_km:.1f} km)' if cycle_length_km is not None else ''
                    
                    # 1. Drive Cycle Speed & Altitude
                    axes_dc[0].plot(res_ecms['time'], res_ecms['velocity_kmh'], label='Rychlost', color='black', linewidth=1.5)
                    axes_dc[0].set_ylabel('Rychlost [km/h]', fontweight='bold')
                    axes_dc[0].set_title(f'Jízdní cyklus: {get_cycle_display_name(cycle_name)}{cycle_length_text}', fontweight='bold')
                    axes_dc[0].grid(True, linestyle=':', alpha=0.7)
                    
                    # Ensure altitude array exists and isn't entirely zeros/empty
                    if 'altitude_m' in res_ecms and len(res_ecms['altitude_m']) > 0 and np.max(np.abs(res_ecms['altitude_m'])) > 0:
                        ax0_alt = axes_dc[0].twinx()
                        ax0_alt.plot(res_ecms['time'], res_ecms['altitude_m'], color='gray', alpha=0.6, linestyle='--')
                        ax0_alt.set_ylabel('Výškový profil [m]', color='gray', fontweight='bold')
                        # Přidání neviditelné čáry na hlavní osu, aby se zobrazila v legendě bez problémů s twinx
                        axes_dc[0].plot([], [], color='gray', alpha=0.6, linestyle='--', label='Výškový profil')
                        
                    if 'RegionalDelivery' in cycle_name:
                        axes_dc[0].legend(loc='lower center', bbox_to_anchor=(0.72, 0.05), framealpha=0.95)
                    else:
                        axes_dc[0].legend(loc='lower center', bbox_to_anchor=(0.76, 0.05), framealpha=0.95)
                        
                    # 2. Torque Request
                    if 't_req' in res_ecms:
                        axes_dc[1].plot(res_ecms['time'], res_ecms['t_req'], label='Požadovaný točivý moment vozidla [Nm]', color='tab:blue', linewidth=1)
                        axes_dc[1].set_ylabel('Požadovaný točivý moment [Nm]', fontweight='bold')
                        axes_dc[1].set_xlabel('Čas [s]', fontweight='bold')
                        axes_dc[1].grid(True, linestyle=':', alpha=0.7)
                        axes_dc[1].legend(loc='upper right', framealpha=0.7)
                        
                    plt.tight_layout()
                    cycle_plot_name = os.path.join(out_dir, f"DriveCycle_Torque_{cycle_name}.pdf")
                    plt.savefig(cycle_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig_dc)
                    print(f"Saved Drive Cycle plot: {cycle_plot_name}")
                    cycle_plotted = True

            # --- 1) SOC comparison output: offline optimization (DP vs ECMS) ---
            if 'DP' in plot_data and 'ECMS' in plot_data:
                fig_soc, ax_soc = plt.subplots(figsize=(12, 4.8))
                lines_soc = []
                for strat, color, lw in [('DP', 'black', 2.5), ('ECMS', 'tab:blue', 2.0)]:
                    data = plot_data[strat]
                    fuel_label = np.trunc(float(data['fuel']) * 1000.0) / 1000.0
                    l, = ax_soc.plot(
                        data['time'],
                        data['soc'],
                        label=f"{strat} (Palivo: {fuel_label:.3f} kg)",
                        color=color,
                        linewidth=lw,
                    )
                    lines_soc.append(l)

                ax_soc.set_title(
                    f'{get_cycle_display_name(cycle_name)} | offline optimalizace | Kapacita baterie: {cap} kWh',
                    fontweight='bold',
                )
                ax_soc.set_xlabel('Čas [s]', fontweight='bold')
                ax_soc.set_ylabel('Stav nabití (SOC) [%]', fontweight='bold')
                ax_soc.grid(True, linestyle=':', alpha=0.7)
                
                has_alt = False
                for st in plot_data:
                    if 'altitude_m' in plot_data[st] and np.any(plot_data[st]['altitude_m']):
                        has_alt = True
                        ax_alt = ax_soc.twinx()
                        l_alt, = ax_alt.plot(plot_data[st]['time'], plot_data[st]['altitude_m'], label='Výškový profil', color='gray', alpha=0.6, linestyle='--')
                        ax_alt.set_ylabel('Výškový profil [m]', color='gray', fontweight='bold')
                        lines_soc.append(l_alt)
                        break
                        
                labels_soc = [l.get_label() for l in lines_soc]
                if has_alt:
                    ax_alt.legend(lines_soc, labels_soc, loc='upper right')
                else:
                    ax_soc.legend(lines_soc, labels_soc, loc='upper right')
                fig_soc.tight_layout()

                soc_plot_name = os.path.join(out_dir, f"SOC_Comparison_DP_ECMS_{cycle_name}_{int(cap)}kWh.pdf")
                fig_soc.savefig(soc_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig_soc)
                print(f"Saved SOC comparison plot: {soc_plot_name}")

            # --- 1b) SOC comparison output: online optimization (AECMS vs PECMS) ---
            if 'AECMS' in plot_data and 'PECMS' in plot_data:
                fig_soc_on, ax_soc_on = plt.subplots(figsize=(12, 4.8))
                lines_soc_on = []
                for strat, color, lw in [('AECMS', 'tab:orange', 2.0), ('PECMS', 'tab:green', 2.0)]:
                    data = plot_data[strat]
                    fuel_label = np.trunc(float(data['fuel']) * 1000.0) / 1000.0
                    l, = ax_soc_on.plot(
                        data['time'],
                        data['soc'],
                        label=f"{strat} (Palivo: {fuel_label:.3f} kg)",
                        color=color,
                        linewidth=lw,
                    )
                    lines_soc_on.append(l)

                # Shared target SOC: plot a single red reference line.
                target_time = None
                target_soc = None
                for strat in ['AECMS', 'PECMS']:
                    data = plot_data[strat]
                    soc_target = np.array(data.get('soc_target', []))
                    if soc_target.size == len(data['time']) and soc_target.size > 0:
                        target_time = data['time']
                        target_soc = soc_target
                        break

                if target_soc is not None:
                    l_target, = ax_soc_on.plot(
                        target_time,
                        target_soc,
                        linestyle='--',
                        linewidth=1.6,
                        color='tab:red',
                        alpha=0.95,
                        label='Cílový SOC',
                    )
                    lines_soc_on.append(l_target)

                ax_soc_on.set_title(
                    f'{get_cycle_display_name(cycle_name)} | online optimalizace | Kapacita baterie: {cap} kWh',
                    fontweight='bold',
                )
                ax_soc_on.set_xlabel('Čas [s]', fontweight='bold')
                ax_soc_on.set_ylabel('Stav nabití (SOC) [%]', fontweight='bold')
                ax_soc_on.grid(True, linestyle=':', alpha=0.7)
                
                has_alt_on = False
                for st in plot_data:
                    if 'altitude_m' in plot_data[st] and np.any(plot_data[st]['altitude_m']):
                        has_alt_on = True
                        ax_alt_on = ax_soc_on.twinx()
                        l_alt_on, = ax_alt_on.plot(plot_data[st]['time'], plot_data[st]['altitude_m'], label='Výškový profil', color='gray', alpha=0.6, linestyle='--')
                        ax_alt_on.set_ylabel('Výškový profil [m]', color='gray', fontweight='bold')
                        lines_soc_on.append(l_alt_on)
                        break
                        
                labels_soc_on = [l.get_label() for l in lines_soc_on]
                if has_alt_on:
                    ax_alt_on.legend(lines_soc_on, labels_soc_on, loc='upper right')
                else:
                    ax_soc_on.legend(lines_soc_on, labels_soc_on, loc='upper right')
                fig_soc_on.tight_layout()

                soc_plot_name_on = os.path.join(out_dir, f"SOC_Comparison_AECMS_PECMS_{cycle_name}_{int(cap)}kWh.pdf")
                fig_soc_on.savefig(soc_plot_name_on, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig_soc_on)
                print(f"Saved SOC comparison plot: {soc_plot_name_on}")

            # --- 1c) Legacy combined SOC output: all available strategies ---
            if plot_data:
                fig_soc_comb, ax_soc_comb = plt.subplots(figsize=(12, 4.8))
                lines_soc_comb = []
                combined_soc_strategies = [
                    ('DP', 'black', 2.5),
                    ('ECMS', 'tab:blue', 2.0),
                    ('AECMS', 'tab:orange', 2.0),
                    ('PECMS', 'tab:green', 2.0),
                ]
                if 'RegionalDeliveryEMSReferenceLoad' in cycle_name:
                    combined_soc_strategies = [s for s in combined_soc_strategies if s[0] != 'ECMS']

                for strat, color, lw in combined_soc_strategies:
                    if strat not in plot_data:
                        continue
                    data = plot_data[strat]
                    fuel_label = np.trunc(float(data['fuel']) * 1000.0) / 1000.0
                    l, = ax_soc_comb.plot(
                        data['time'],
                        data['soc'],
                        label=f"{strat} (Palivo: {fuel_label:.3f} kg)",
                        color=color,
                        linewidth=lw,
                    )
                    lines_soc_comb.append(l)

                target_time = None
                target_soc = None
                for strat in ['AECMS', 'PECMS']:
                    if strat not in plot_data:
                        continue
                    data = plot_data[strat]
                    soc_target = np.array(data.get('soc_target', []))
                    if soc_target.size == len(data['time']) and soc_target.size > 0:
                        target_time = data['time']
                        target_soc = soc_target
                        break

                if target_soc is not None:
                    l_target, = ax_soc_comb.plot(
                        target_time,
                        target_soc,
                        linestyle='--',
                        linewidth=1.6,
                        color='tab:red',
                        alpha=0.95,
                        label='Cílový SOC',
                    )
                    lines_soc_comb.append(l_target)

                ax_soc_comb.set_title(
                    f'{get_cycle_display_name(cycle_name)} | kombinované porovnání SOC | Kapacita baterie: {cap} kWh',
                    fontweight='bold',
                )
                ax_soc_comb.set_xlabel('Čas [s]', fontweight='bold')
                ax_soc_comb.set_ylabel('Stav nabití (SOC) [%]', fontweight='bold')
                ax_soc_comb.grid(True, linestyle=':', alpha=0.7)
                
                has_alt_comb = False
                for st in plot_data:
                    if 'altitude_m' in plot_data[st] and np.any(plot_data[st]['altitude_m']):
                        has_alt_comb = True
                        ax_alt_comb = ax_soc_comb.twinx()
                        l_alt_comb, = ax_alt_comb.plot(plot_data[st]['time'], plot_data[st]['altitude_m'], label='Výškový profil', color='gray', alpha=0.6, linestyle='--')
                        ax_alt_comb.set_ylabel('Výškový profil [m]', color='gray', fontweight='bold')
                        lines_soc_comb.append(l_alt_comb)
                        break
                        
                labels_soc_comb = [l.get_label() for l in lines_soc_comb]
                if has_alt_comb:
                    ax_alt_comb.legend(lines_soc_comb, labels_soc_comb, loc='upper right')
                else:
                    ax_soc_comb.legend(lines_soc_comb, labels_soc_comb, loc='upper right')
                fig_soc_comb.tight_layout()

                combined_soc_name = os.path.join(out_dir, f"Combined_SOC_{cycle_name}_{int(cap)}kWh.pdf")
                fig_soc_comb.savefig(combined_soc_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                plt.close(fig_soc_comb)
                print(f"Saved combined SOC plot: {combined_soc_name}")

            # --- 2) Map output: 4 equal panels (DP row, ECMS row) ---
            if ('DP' in plot_data and 'ECMS' in plot_data) or ('AECMS' in plot_data and 'PECMS' in plot_data):
                # Prepare ICE background map
                try:
                    df_vmap = pd.read_csv("Engine/325kW.vmap", sep=',', header=0)
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
                    else:
                        max_rpm, max_tq = np.array([]), np.array([])

                    if rpm_bsfc.size > 3:
                        rpm_scale = max(1e-9, float(rpm_bsfc.max() - rpm_bsfc.min()))
                        tq_scale = max(1e-9, float(tq_bsfc.max() - tq_bsfc.min()))
                        rpm_norm = (rpm_bsfc - rpm_bsfc.min()) / rpm_scale
                        tq_norm = (tq_bsfc - tq_bsfc.min()) / tq_scale

                        triang_norm = mtri.Triangulation(rpm_norm, tq_norm)
                        refiner = mtri.UniformTriRefiner(triang_norm)
                        triang_fine, bsfc_fine = refiner.refine_field(bsfc, subdiv=3)

                        rpm_fine = triang_fine.x * rpm_scale + rpm_bsfc.min()
                        tq_fine = triang_fine.y * tq_scale + tq_bsfc.min()
                        ice_triang = mtri.Triangulation(rpm_fine, tq_fine, triang_fine.triangles)

                        if max_rpm.size > 1 and ice_triang.triangles.size > 0:
                            centroids_rpm = np.mean(rpm_fine[ice_triang.triangles], axis=1)
                            centroids_tq = np.mean(tq_fine[ice_triang.triangles], axis=1)
                            tq_env = np.interp(np.clip(centroids_rpm, max_rpm.min(), max_rpm.max()), max_rpm, max_tq)
                            ice_triang.set_mask((centroids_tq > tq_env) | (centroids_tq < 0.0))

                        ice_bsfc_vals = bsfc_fine
                    else:
                        ice_triang = None
                        ice_bsfc_vals = None
                except Exception as e:
                    print(f"Warning: Could not load ICE map background: {e}")
                    max_rpm, max_tq = np.array([]), np.array([])
                    ice_triang = None
                    ice_bsfc_vals = None

                # Prepare EM background map
                em_rpm_lim = np.array([])
                em_tq_drive = np.array([])
                em_tq_drag = np.array([])
                em_eff_triang = None
                em_eff_vals = None
                try:
                    df_vem = pd.read_csv("Emotor/EM_fld.vemp", skipinitialspace=True)
                    em_rpm_lim = df_vem.iloc[:, 0].to_numpy()
                    em_tq_drive = df_vem.iloc[:, 1].to_numpy()
                    em_tq_drag = df_vem.iloc[:, 2].to_numpy()

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
                    print(f"Warning: Could not load E-Motor map background: {e}")

                def plot_ice_panel(ax, rpm_arr, t_eng_arr, title):
                    rpm_arr = np.asarray(rpm_arr)
                    t_eng_arr = np.asarray(t_eng_arr)
                    n = min(rpm_arr.size, t_eng_arr.size)
                    if n <= 0:
                        ax.set_title(title, fontweight='bold')
                        ax.set_xlabel('Otáčky motoru [1/min]')
                        ax.set_ylabel('Točivý moment motoru [Nm]')
                        ax.grid(False)
                        return None

                    rpm_arr = rpm_arr[:n]
                    t_eng_arr = t_eng_arr[:n]
                    mask_ice = t_eng_arr > 0.0
                    rpm_pts = rpm_arr[mask_ice]
                    t_pts = t_eng_arr[mask_ice]

                    x_candidates = [arr for arr in [max_rpm, rpm_pts] if np.size(arr) > 0]
                    if x_candidates:
                        x_all = np.concatenate(x_candidates)
                        x_min = float(np.nanmin(x_all))
                        x_max = float(np.nanmax(x_all))
                        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
                            x_min, x_max = 0.0, 2500.0
                    else:
                        x_min, x_max = 0.0, 2500.0

                    y_candidates = []
                    if max_tq.size > 0:
                        y_candidates.append(float(np.nanmax(max_tq) * 1.05))
                    if t_pts.size > 0:
                        y_candidates.append(float(np.nanmax(t_pts) * 1.10))
                    y_top = max(400.0, *y_candidates) if y_candidates else 3000.0

                    if ice_triang is not None and ice_bsfc_vals is not None and len(ice_bsfc_vals) > 0:
                        cs = ax.tricontour(
                            ice_triang,
                            ice_bsfc_vals,
                            levels=np.concatenate([np.arange(182, 210, 2), np.arange(210, 340, 10)]),
                            colors='0.25',
                            linewidths=0.35,
                            alpha=0.35,
                            zorder=4,
                        )
                        ax.clabel(cs, inline=True, fontsize=7, fmt='%d', colors='0.25')

                    handle = None
                    if rpm_pts.size > 0:
                        hist, _, _ = np.histogram2d(
                            rpm_pts,
                            t_pts,
                            bins=50,
                            range=[[x_min, x_max], [0.0, y_top]],
                        )
                        if hist.sum() > 0:
                            hist = (hist / hist.sum()) * 100.0
                            hist_smooth = gaussian_filter(hist, sigma=0.85)
                            positive_values = hist_smooth[hist_smooth > 0.0]
                            if positive_values.size > 0:
                                vmin = max(float(np.percentile(positive_values, 1)), 0.005)
                                vmax = float(np.nanpercentile(positive_values, 99.5))
                                if vmax <= vmin:
                                    vmax = vmin * 1.01
                                smooth_fine = zoom(hist_smooth.T, zoom=4, order=3)
                                smooth_fine = np.ma.masked_less_equal(smooth_fine, 0.0)
                                x_fine = np.linspace(x_min, x_max, smooth_fine.shape[1])
                                y_fine = np.linspace(0.0, y_top, smooth_fine.shape[0])

                                if max_rpm.size > 1 and max_tq.size > 1:
                                    max_tq_interp = np.interp(x_fine, max_rpm, max_tq, left=np.nan, right=np.nan)
                                    env_mask = y_fine[:, None] > max_tq_interp[None, :]
                                    env_mask |= ~np.isfinite(max_tq_interp)[None, :]
                                    smooth_fine = np.ma.masked_array(smooth_fine, mask=np.ma.getmaskarray(smooth_fine) | env_mask)

                                handle = ax.imshow(
                                    smooth_fine,
                                    origin='lower',
                                    extent=[x_min, x_max, 0.0, y_top * 1.01],
                                    cmap='Blues',
                                    norm=PowerNorm(gamma=0.27, vmin=vmin, vmax=vmax),
                                    interpolation='bicubic',
                                    aspect='auto',
                                    alpha=1.0,
                                    zorder=1,
                                )

                    if max_rpm.size > 0:
                        ax.plot(max_rpm, max_tq, 'k-', linewidth=2.6, zorder=5)

                    ax.set_title(title, fontweight='bold')
                    ax.set_xlabel('Otáčky motoru [1/min]')
                    ax.set_ylabel('Točivý moment motoru [Nm]')
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(0.0, y_top)
                    ax.grid(False)
                    return handle

                def plot_em_panel(ax, rpm_arr, t_em_arr, title):
                    rpm_arr = np.asarray(rpm_arr)
                    t_em_arr = np.asarray(t_em_arr)
                    n = min(rpm_arr.size, t_em_arr.size)
                    if n <= 0:
                        ax.set_title(title, fontweight='bold')
                        ax.set_xlabel('Otáčky elektromotoru [1/min]')
                        ax.set_ylabel('')
                        ax.axhline(0, color='gray', linewidth=1)
                        ax.grid(False)
                        return None

                    rpm_arr = rpm_arr[:n]
                    t_em_arr = t_em_arr[:n]
                    mask_em = np.abs(t_em_arr) > 0.5
                    rpm_pts = rpm_arr[mask_em]
                    t_pts = t_em_arr[mask_em]

                    if em_eff_triang is not None and em_eff_vals is not None and len(em_eff_vals) > 0:
                        cs = ax.tricontour(
                            em_eff_triang,
                            em_eff_vals,
                            levels=np.arange(60, 100, 2),
                            colors='0.25',
                            linewidths=0.35,
                            alpha=0.35,
                            zorder=4,
                        )
                        ax.clabel(cs, inline=True, fontsize=7, fmt='%d', colors='0.25')

                    x_candidates = [arr for arr in [em_rpm_lim, rpm_pts] if np.size(arr) > 0]
                    if x_candidates:
                        x_all = np.concatenate(x_candidates)
                        x_min = float(np.nanmin(x_all))
                        x_max = float(np.nanmax(x_all))
                    else:
                        x_min, x_max = 200.0, 3000.0

                    y_candidates = [arr for arr in [em_tq_drive, em_tq_drag, t_pts] if np.size(arr) > 0]
                    if y_candidates:
                        y_all = np.concatenate(y_candidates)
                        y_min = float(np.nanmin(y_all))
                        y_max = float(np.nanmax(y_all))
                    else:
                        y_min, y_max = -1600.0, 1600.0

                    x_min = max(200.0, x_min)
                    x_max = min(3000.0, x_max)
                    if x_max <= x_min:
                        x_min, x_max = 200.0, 3000.0

                    y_span_abs = max(abs(y_min) * 1.10, abs(y_max) * 1.10, 300.0)
                    y_span_abs = min(y_span_abs, 1600.0)
                    y_min, y_max = -y_span_abs, y_span_abs

                    handle = None
                    if rpm_pts.size > 0:
                        hist, _, _ = np.histogram2d(
                            rpm_pts,
                            t_pts,
                            bins=50,
                            range=[[x_min, x_max], [y_min, y_max]],
                        )
                        if hist.sum() > 0:
                            hist = (hist / hist.sum()) * 100.0
                            hist_smooth = gaussian_filter(hist, sigma=0.85)
                            positive_values = hist_smooth[hist_smooth > 0.0]
                            if positive_values.size > 0:
                                vmin = max(float(np.percentile(positive_values, 1)), 0.009)
                                vmax = float(np.nanpercentile(positive_values, 99.5))
                                if vmax <= vmin:
                                    vmax = vmin * 1.01
                                smooth_fine = zoom(hist_smooth.T, zoom=3, order=2)
                                smooth_fine = np.ma.masked_less_equal(smooth_fine, 0.0)
                                x_fine = np.linspace(x_min, x_max, smooth_fine.shape[1])
                                y_fine = np.linspace(y_min, y_max, smooth_fine.shape[0])

                                if em_rpm_lim.size > 1 and em_tq_drive.size > 1 and em_tq_drag.size > 1:
                                    drive_env = np.interp(x_fine, em_rpm_lim, em_tq_drive, left=np.nan, right=np.nan)
                                    drag_env = np.interp(x_fine, em_rpm_lim, em_tq_drag, left=np.nan, right=np.nan)
                                    env_mask = (y_fine[:, None] > drive_env[None, :]) | (y_fine[:, None] < drag_env[None, :])
                                    env_mask |= ~np.isfinite(drive_env)[None, :]
                                    env_mask |= ~np.isfinite(drag_env)[None, :]
                                    smooth_fine = np.ma.masked_array(smooth_fine, mask=np.ma.getmaskarray(smooth_fine) | env_mask)

                                handle = ax.imshow(
                                    smooth_fine,
                                    origin='lower',
                                    extent=[x_min, x_max, y_min * 1.04, y_max * 1.04],
                                    cmap='Greens',
                                    norm=PowerNorm(gamma=0.27, vmin=vmin, vmax=vmax),
                                    interpolation='bicubic',
                                    aspect='auto',
                                    alpha=1.0,
                                    zorder=1,
                                )

                    if em_rpm_lim.size > 0:
                        ax.plot(em_rpm_lim, em_tq_drive, 'k-', linewidth=2.3, zorder=5)
                        ax.plot(em_rpm_lim, em_tq_drag, 'k--', linewidth=2.3, zorder=5)

                    ax.set_title(title, fontweight='bold')
                    ax.set_xlabel('Otáčky elektromotoru [1/min]')
                    ax.set_ylabel('')
                    ax.axhline(0, color='gray', linewidth=1)
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(y_min, y_max)
                    ax.grid(False)
                    return handle

                # --- 2a) Engine-only BSFC comparison: 3 panels (DP | ICE-only | strategy) ---
                if 'DP' in plot_data and any(s in plot_data for s in ['ECMS', 'PECMS', 'AECMS']):
                    # Prefer PECMS as requested; fallback to AECMS, then ECMS.
                    ref_strat = 'PECMS' if 'PECMS' in plot_data else ('AECMS' if 'AECMS' in plot_data else 'ECMS')

                    # DP operating points (active engine torque only)
                    dp_data = plot_data['DP']
                    rpm_dp = np.asarray(dp_data.get('rpm_ice', []))
                    t_dp = np.asarray(dp_data.get('t_ice', []))
                    n_dp = min(rpm_dp.size, t_dp.size)
                    if n_dp > 0:
                        rpm_dp = rpm_dp[:n_dp]
                        t_dp = t_dp[:n_dp]
                        mask_dp = t_dp > 0.0
                        rpm_dp = rpm_dp[mask_dp]
                        t_dp = t_dp[mask_dp]

                    # ICE-only reference from cycle torque request (taken from selected strategy run)
                    ref_data = plot_data[ref_strat]
                    rpm_ref = np.asarray(ref_data.get('rpm_ice', []))
                    t_req_ref = np.asarray(ref_data.get('t_req', []))
                    n_ref_req = min(rpm_ref.size, t_req_ref.size)
                    rpm_ice_only = np.array([])
                    t_ice_only = np.array([])
                    if n_ref_req > 0:
                        rpm_tmp = rpm_ref[:n_ref_req]
                        t_req_tmp = t_req_ref[:n_ref_req]
                        mask_req = t_req_tmp > 0.0
                        rpm_ice_only = rpm_tmp[mask_req]
                        t_ice_only = t_req_tmp[mask_req]

                    # Selected strategy active engine points
                    t_ref = np.asarray(ref_data.get('t_ice', []))
                    n_ref = min(rpm_ref.size, t_ref.size)
                    rpm_ref_act = np.array([])
                    t_ref_act = np.array([])
                    if n_ref > 0:
                        rpm_tmp = rpm_ref[:n_ref]
                        t_tmp = t_ref[:n_ref]
                        mask_ref = t_tmp > 0.0
                        rpm_ref_act = rpm_tmp[mask_ref]
                        t_ref_act = t_tmp[mask_ref]

                    # Smaller height as requested.
                    fig_ice3, axes_ice3 = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

                    h_dp = plot_ice_panel(axes_ice3[0], rpm_dp, t_dp, '(a) DP na mapě BSFC')
                    h_base = plot_ice_panel(axes_ice3[1], rpm_ice_only, t_ice_only, '(b) Pouze ICE na mapě BSFC')
                    h_ref = plot_ice_panel(axes_ice3[2], rpm_ref_act, t_ref_act, f'(c) {ref_strat} na mapě BSFC')

                    # Use one normalized color scale across all three maps.
                    handles_ice3 = [h for h in [h_dp, h_base, h_ref] if h is not None and hasattr(h, 'norm')]
                    shared_norm_ice3 = None
                    # Locked scale so colors have identical meaning across all runs and panels.
                    fixed_vmin_ice3 = 0.01
                    fixed_vmax_ice3 = 1.25
                    # Distribute ticks evenly in color-space for better visual spacing on PowerNorm.
                    gamma_ice3 = 0.27
                    tick_pos = np.linspace(0.0, 1.0, 9)
                    shared_ticks_ice3 = fixed_vmin_ice3 + ((tick_pos ** (1.0 / gamma_ice3)) * (fixed_vmax_ice3 - fixed_vmin_ice3))
                    shared_ticks_ice3[0] = fixed_vmin_ice3
                    shared_ticks_ice3[-1] = fixed_vmax_ice3
                    if handles_ice3:
                        shared_norm_ice3 = PowerNorm(gamma=gamma_ice3, vmin=fixed_vmin_ice3, vmax=fixed_vmax_ice3)
                        for h in handles_ice3:
                            h.set_norm(shared_norm_ice3)
                            if hasattr(h, 'set_clim'):
                                h.set_clim(fixed_vmin_ice3, fixed_vmax_ice3)

                    # Single shared colorbar on the right.
                    h_cb = h_dp if h_dp is not None else (h_base if h_base is not None else h_ref)
                    if h_cb is not None:
                        fig_ice3.tight_layout(rect=[0.0, 0.0, 0.95, 1.0])
                        cax = fig_ice3.add_axes([0.955, 0.14, 0.012, 0.74])
                        if shared_norm_ice3 is not None:
                            sm_ice3 = ScalarMappable(norm=shared_norm_ice3, cmap='Blues')
                            sm_ice3.set_array([])
                            cb = fig_ice3.colorbar(sm_ice3, cax=cax)
                            if shared_ticks_ice3 is not None:
                                cb.set_ticks(shared_ticks_ice3)
                                cb.ax.yaxis.set_major_formatter(
                                    FuncFormatter(lambda x, _: f"{x:.3f}".rstrip('0').rstrip('.'))
                                )
                        else:
                            cb = fig_ice3.colorbar(h_cb, cax=cax)
                        cb.set_label('Časový podíl běhu [%]')
                    else:
                        fig_ice3.tight_layout()

                    ice3_name = os.path.join(
                        out_dir,
                        f"EngineMap_Comparison_DP_ICE_{ref_strat}_{cycle_name}_{int(cap)}kWh.pdf",
                    )
                    fig_ice3.savefig(ice3_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig_ice3)
                    print(f"Saved 3-panel engine map plot: {ice3_name}")

                def save_pair_maps(top_strat, bottom_strat, file_prefix):
                    fig_maps, axes_maps = plt.subplots(2, 2, figsize=(17, 12))

                    h_ice_top = plot_ice_panel(
                        axes_maps[0, 0],
                        plot_data[top_strat].get('rpm_ice', []),
                        plot_data[top_strat].get('t_ice', []),
                        f'{top_strat} - mapa provozních bodů spalovacího motoru',
                    )
                    h_em_top = plot_em_panel(
                        axes_maps[0, 1],
                        plot_data[top_strat].get('rpm_ice', []),
                        plot_data[top_strat].get('t_em', []),
                        f'{top_strat} - mapa provozních bodů elektromotoru',
                    )
                    h_ice_bottom = plot_ice_panel(
                        axes_maps[1, 0],
                        plot_data[bottom_strat].get('rpm_ice', []),
                        plot_data[bottom_strat].get('t_ice', []),
                        f'{bottom_strat} - mapa provozních bodů spalovacího motoru',
                    )
                    h_em_bottom = plot_em_panel(
                        axes_maps[1, 1],
                        plot_data[bottom_strat].get('rpm_ice', []),
                        plot_data[bottom_strat].get('t_em', []),
                        f'{bottom_strat} - mapa provozních bodů elektromotoru',
                    )

                    # Keep Y values visible on EM plots, but remove the Y-axis title text.
                    for ax_em_col in [axes_maps[0, 1], axes_maps[1, 1]]:
                        ax_em_col.set_ylabel('')
                        ax_em_col.yaxis.tick_left()
                        ax_em_col.tick_params(labelleft=True, left=True, labelright=False, right=False, pad=1)

                    # Harmonize color scales within one engine type across both rows.
                    def _global_limits(handles):
                        limits = []
                        for h in handles:
                            if h is None or not hasattr(h, 'norm'):
                                continue
                            vmin = getattr(h.norm, 'vmin', None)
                            vmax = getattr(h.norm, 'vmax', None)
                            if vmin is None or vmax is None:
                                continue
                            if np.isfinite(vmin) and np.isfinite(vmax):
                                limits.append((float(vmin), float(vmax)))
                        if not limits:
                            return 0.0, 1.0
                        gmin = min(v[0] for v in limits)
                        gmax = max(v[1] for v in limits)
                        if gmax <= gmin:
                            gmax = gmin + 1.0
                        return gmin, gmax

                    def _legend_ticks(vmin, vmax, n_ticks=9):
                        # Use only real data limits: include exact min/max and no out-of-range labels.
                        legend_vmin = float(vmin)
                        legend_vmax = float(vmax)
                        if not np.isfinite(legend_vmin) or not np.isfinite(legend_vmax):
                            legend_vmin, legend_vmax = 0.0, 1.0
                        if legend_vmax <= legend_vmin:
                            legend_vmax = legend_vmin + 1.0

                        ticks = np.linspace(legend_vmin, legend_vmax, n_ticks)
                        fmt = FuncFormatter(lambda x, _: f"{x:.2f}".rstrip('0').rstrip('.'))
                        return ticks, fmt, legend_vmin, legend_vmax

                    ice_vmin_global, ice_vmax_global = _global_limits([h_ice_top, h_ice_bottom])
                    em_vmin_global, em_vmax_global = _global_limits([h_em_top, h_em_bottom])

                    for h in [h_ice_top, h_ice_bottom]:
                        if h is not None and hasattr(h, 'set_clim'):
                            h.set_clim(ice_vmin_global, ice_vmax_global)
                    for h in [h_em_top, h_em_bottom]:
                        if h is not None and hasattr(h, 'set_clim'):
                            h.set_clim(em_vmin_global, em_vmax_global)

                    ice_ticks, ice_fmt, ice_leg_vmin, ice_leg_vmax = _legend_ticks(ice_vmin_global, ice_vmax_global, n_ticks=9)
                    em_ticks, em_fmt, em_leg_vmin, em_leg_vmax = _legend_ticks(em_vmin_global, em_vmax_global, n_ticks=9)

                    fig_maps.tight_layout(rect=[0.0, 0.0, 0.93, 1.0])
                    fig_maps.subplots_adjust(wspace=0.20)

                    # Per-row adjacent legends (ICE blue + EM green) next to EM subplots.
                    def add_row_legends(row_idx, h_ice_row, h_em_row):
                        if h_ice_row is None and h_em_row is None:
                            return

                        row_pos_ice = axes_maps[row_idx, 0].get_position()
                        row_pos_em = axes_maps[row_idx, 1].get_position()
                        cbar_w = 0.010
                        cbar_pad = 0.002
                        # ICE legend in the inter-column gap, biased left to avoid EM Y-axis overlap.
                        cbar_x_ice = row_pos_ice.x1 + cbar_pad
                        cbar_x_ice = min(cbar_x_ice, row_pos_em.x0 - cbar_w - 0.018)
                        # EM legend to the right of EM map.
                        cbar_x_em = row_pos_em.x1 + cbar_pad
                        cbar_y = row_pos_em.y0
                        cbar_h = row_pos_em.height

                        if h_ice_row is not None and hasattr(h_ice_row, 'norm'):
                            cax_ice = fig_maps.add_axes([cbar_x_ice, cbar_y, cbar_w, cbar_h])
                            ice_proxy = ScalarMappable(norm=Normalize(vmin=ice_leg_vmin, vmax=ice_leg_vmax), cmap=h_ice_row.get_cmap())
                            ice_proxy.set_array([])
                            cb_ice = fig_maps.colorbar(
                                ice_proxy,
                                cax=cax_ice,
                                ticks=ice_ticks,
                            )
                            cb_ice.set_label('Časový podíl běhu spalovacího motoru [%]')
                            cb_ice.ax.yaxis.set_major_formatter(ice_fmt)
                            cb_ice.ax.yaxis.set_ticks_position('right')
                            cb_ice.ax.yaxis.set_label_position('right')
                            cb_ice.ax.yaxis.label.set_fontsize(10)
                            cb_ice.ax.tick_params(labelleft=False, labelright=True, left=False, right=True, pad=1)

                        if h_em_row is not None and hasattr(h_em_row, 'norm'):
                            cax_em = fig_maps.add_axes([cbar_x_em, cbar_y, cbar_w, cbar_h])
                            em_proxy = ScalarMappable(norm=Normalize(vmin=em_leg_vmin, vmax=em_leg_vmax), cmap=h_em_row.get_cmap())
                            em_proxy.set_array([])
                            cb_em = fig_maps.colorbar(
                                em_proxy,
                                cax=cax_em,
                                ticks=em_ticks,
                            )
                            cb_em.set_label('Časový podíl běhu elektromotoru [%]')
                            cb_em.ax.yaxis.set_major_formatter(em_fmt)
                            cb_em.ax.yaxis.set_label_position('right')
                            cb_em.ax.yaxis.label.set_fontsize(10)
                            cb_em.ax.tick_params(labelleft=False, labelright=True, left=False, right=True, pad=1)

                    add_row_legends(0, h_ice_top, h_em_top)
                    add_row_legends(1, h_ice_bottom, h_em_bottom)

                    maps_plot_name = os.path.join(out_dir, f"{file_prefix}_{cycle_name}_{int(cap)}kWh.pdf")
                    fig_maps.savefig(maps_plot_name, dpi=300, bbox_inches='tight', pad_inches=0.05)
                    plt.close(fig_maps)
                    print(f"Saved strategy maps plot: {maps_plot_name}")

                if 'DP' in plot_data and 'ECMS' in plot_data:
                    save_pair_maps('DP', 'ECMS', 'Strategy_Maps_DP_ECMS')

                if 'AECMS' in plot_data and 'PECMS' in plot_data:
                    save_pair_maps('AECMS', 'PECMS', 'Strategy_Maps_AECMS_PECMS')
            
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, "batch_results.csv"), index=False)
            
    print("\n\n=== BATCH SIMULATION COMPLETE ===")
    print(df_res)
    print("Results saved to batch_results.csv")

if __name__ == "__main__":
    main()
