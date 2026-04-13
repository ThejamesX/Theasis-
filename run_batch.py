import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    
    # Note: main.py expects 'A-ECMS' instead of 'AECMS'
    strategies_ecms = ['ECMS', 'AECMS', 'PECMS'] #'ECMS', 'AECMS', 'PECMS'
    
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
                    axes_dc[0].plot(res_ecms['time'], res_ecms['velocity_kmh'], label='Speed [km/h]', color='black', linewidth=1.5)
                    axes_dc[0].set_ylabel('Speed [km/h]', fontweight='bold')
                    axes_dc[0].set_title(f'Drive Cycle: {cycle_name.replace("_", " ")}', fontweight='bold')
                    axes_dc[0].grid(True, linestyle=':', alpha=0.7)
                    
                    if 'altitude_m' in res_ecms and np.any(res_ecms['altitude_m']):
                        ax0_alt = axes_dc[0].twinx()
                        ax0_alt.plot(res_ecms['time'], res_ecms['altitude_m'], label='Altitude [m]', color='gray', alpha=0.6, linestyle='--')
                        ax0_alt.set_ylabel('Altitude [m]', color='gray', fontweight='bold')
                        
                    # 2. Torque Request
                    if 't_req' in res_ecms:
                        axes_dc[1].plot(res_ecms['time'], res_ecms['t_req'], label='Vehicle Required Torque [Nm]', color='tab:blue', linewidth=1)
                        axes_dc[1].set_ylabel('Required Torque [Nm]', fontweight='bold')
                        axes_dc[1].set_xlabel('Time [s]', fontweight='bold')
                        axes_dc[1].grid(True, linestyle=':', alpha=0.7)
                        axes_dc[1].legend(loc='upper right')
                        
                    plt.tight_layout()
                    cycle_plot_name = os.path.join(out_dir, f"DriveCycle_Torque_{cycle_name}.png")
                    plt.savefig(cycle_plot_name, dpi=300)
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
                    plt.plot(data['time'], data['soc'], label=f"{p_strat} (Fuel: {data['fuel']:.2f} kg)", 
                             color=colors.get(p_strat, 'gray'), linewidth=2.5 if p_strat == 'DP' else 1.5, alpha=0.9 if p_strat == 'DP' else 0.8)
            
            # Plot reference Target line
            if 'Target' in plot_data:
                plt.plot(plot_data['Target']['time'], plot_data['Target']['soc'], color='black', linestyle='--', linewidth=2, alpha=0.8, label='Target Reference')
            
            plt.title(f'SOC Trajectories Comparison | {cycle_name.replace("_", " ")} | Capacity: {cap} kWh', fontweight='bold')
            plt.xlabel('Time [s]', fontweight='bold')
            plt.ylabel('State of Charge (SOC) [%]', fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            combined_plot_name = os.path.join(out_dir, f"Combined_SOC_{cycle_name}_{int(cap)}kWh.png")
            plt.savefig(combined_plot_name, dpi=300)
            plt.close()
            print(f"Saved combined plot: {combined_plot_name}")
            
            # --- 1) EF Comparison Plot (AECMS vs PECMS) ---
            if 'AECMS' in plot_data and 'PECMS' in plot_data:
                fig_ef, ax_ef = plt.subplots(figsize=(10, 5))
                ax_ef.plot(plot_data['AECMS']['time'], plot_data['AECMS']['s_factor'], label='A-ECMS EF', color='tab:orange', linewidth=1.5)
                ax_ef.plot(plot_data['PECMS']['time'], plot_data['PECMS']['s_factor'], label='P-ECMS EF', color='tab:green', linewidth=2.0)
                ax_ef.set_title(f'Equivalence Factor (EF) Comparison | {cycle_name.replace("_", " ")} | {int(cap)} kWh', fontweight='bold')
                ax_ef.set_xlabel('Time [s]', fontweight='bold')
                ax_ef.set_ylabel('Equivalence Factor (s)', fontweight='bold')
                ax_ef.grid(True, linestyle=':', alpha=0.7)
                ax_ef.legend(loc='upper right')
                plt.tight_layout()
                ef_plot_name = os.path.join(out_dir, f"EF_Comparison_{cycle_name}_{int(cap)}kWh.png")
                plt.savefig(ef_plot_name, dpi=300)
                plt.close(fig_ef)
                print(f"Saved EF Comparison plot: {ef_plot_name}")

            # --- 2) Engine Map Plot (ICE-only vs PECMS) ---
            if 'PECMS' in plot_data:
                fig_map, axes_map = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                
                # Fetch Maximum Torque Curve
                try:
                    df_vmap = pd.read_csv("Engine/325kW.vmap", sep=',', header=0) 
                    df_vmap.columns = ['rpm', 'torque', 'fuel']
                    
                    rpm_raw = df_vmap['rpm'].values
                    torque_raw = df_vmap['torque'].values
                    
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

                    max_tq_curve = np.array(max_tq_curve)
                    max_rpm = max_tq_curve[:,0]
                    max_tq = max_tq_curve[:,1]
                except Exception as e:
                    print("Warning: Could not load max torque curve:", e)
                    max_rpm, max_tq = [], []

                pecms_data = plot_data['PECMS']
                rpm = pecms_data['rpm_ice']
                t_req = pecms_data['t_req']
                t_ice_pecms = pecms_data['t_ice']
                
                # ICE-only: Engine is driven exactly by T_req -> filter positive requests
                mask_ice = (t_req > 0)
                rpm_ice_only = rpm[mask_ice]
                t_ice_only = t_req[mask_ice]
                
                # PECMS: Plot moments where ICE was providing positive torque
                mask_pecms = (t_ice_pecms > 0)
                rpm_pecms = rpm[mask_pecms]
                t_ice_act = t_ice_pecms[mask_pecms]

                from scipy.ndimage import gaussian_filter
                
                # Setup grids for smooth density contour
                bins = 50
                
                # (a) ICE-only Map
                ax1 = axes_map[0]
                if len(rpm_ice_only) > 0:
                    hist1, xedges1, yedges1 = np.histogram2d(rpm_ice_only, t_ice_only, bins=bins, range=[[400, 2500], [0, max_tq.max()*1.05]])
                    # Time share percentage
                    hist1 = (hist1 / hist1.sum()) * 100.0
                    hist1_smooth = gaussian_filter(hist1, sigma=1.0)
                    # Mask out near-zero values for cleaner look
                    hist1_smooth[hist1_smooth < 0.05] = np.nan
                    
                    X1, Y1 = np.meshgrid(xedges1[:-1], yedges1[:-1])
                    cf1 = ax1.contourf(X1, Y1, hist1_smooth.T, cmap='Blues', levels=15, alpha=0.8)
                    cb1 = fig_map.colorbar(cf1, ax=ax1, label='Time share / %')
                    
                if len(max_rpm) > 0:
                    ax1.plot(max_rpm, max_tq, 'k-', linewidth=3.5)
                ax1.set_title('(a) ICE-only', fontweight='bold')
                ax1.set_xlabel('Engine speed [1/min]')
                ax1.set_ylabel('Engine torque [Nm]')
                ax1.set_xlim(0, 2500)
                ax1.set_ylim(-100, 3000)
                ax1.grid(True, linestyle=':', alpha=0.7)
                
                # (b) PECMS Map
                ax2 = axes_map[1]
                if len(rpm_pecms) > 0:
                    hist2, xedges2, yedges2 = np.histogram2d(rpm_pecms, t_ice_act, bins=bins, range=[[400, 2500], [0, max_tq.max()*1.05]])
                    # Time share percentage
                    hist2 = (hist2 / hist2.sum()) * 100.0
                    hist2_smooth = gaussian_filter(hist2, sigma=1.0)
                    hist2_smooth[hist2_smooth < 0.05] = np.nan
                    
                    X2, Y2 = np.meshgrid(xedges2[:-1], yedges2[:-1])
                    cf2 = ax2.contourf(X2, Y2, hist2_smooth.T, cmap='Blues', levels=15, alpha=0.8)
                    cb2 = fig_map.colorbar(cf2, ax=ax2, label='Time share / %')
                    
                if len(max_rpm) > 0:
                    ax2.plot(max_rpm, max_tq, 'k-', linewidth=3.5)
                ax2.set_title('(b) P-ECMS', fontweight='bold')
                ax2.set_xlabel('Engine speed [1/min]')
                ax2.set_xlim(0, 2500)
                ax2.set_ylim(-100, 3000)
                ax2.grid(True, linestyle=':', alpha=0.7)
                
                plt.tight_layout()
                map_plot_name = os.path.join(out_dir, f"EngineMap_Comparison_{cycle_name}_{int(cap)}kWh.png")
                plt.savefig(map_plot_name, dpi=300)
                plt.close(fig_map)
                print(f"Saved Engine Map Plot: {map_plot_name}")
                
            # --- 3) E-Motor Map Plot ---
            for strat in plot_data.keys():
                if strat in ['AECMS', 'PECMS', 'ECMS']:
                    fig_em, ax_em = plt.subplots(figsize=(8, 6))
                    
                    try:
                        df_vem = pd.read_csv("Emotor/EM_fld.vemp", skipinitialspace=True)
                        em_rpm_lim = df_vem.iloc[:, 0].values
                        em_tq_drive = df_vem.iloc[:, 1].values
                        em_tq_drag = df_vem.iloc[:, 2].values
                    except Exception as e:
                        print("Warning: Could not load EM limits:", e)
                        em_rpm_lim, em_tq_drive, em_tq_drag = [], [], []

                    em_data = plot_data[strat]
                    rpm_em = em_data['rpm_ice']
                    t_em_act = em_data['t_em']
                    
                    # Filter points where EM is active
                    mask_em = (np.abs(t_em_act) > 0.5)
                    if np.sum(mask_em) > 0:
                        rpm_em_active = rpm_em[mask_em]
                        t_em_active = t_em_act[mask_em]
                        
                        from scipy.ndimage import gaussian_filter
                        bins = 50
                        hist_em, xedges_em, yedges_em = np.histogram2d(
                            rpm_em_active, t_em_active, bins=bins, 
                            range=[[0, 3000], [-1500, 1500]]
                        )
                        # Time share percentage
                        hist_em = (hist_em / hist_em.sum()) * 100.0
                        hist_em_smooth = gaussian_filter(hist_em, sigma=1.0)
                        hist_em_smooth[hist_em_smooth < 0.05] = np.nan
                        
                        X_em, Y_em = np.meshgrid(xedges_em[:-1], yedges_em[:-1])
                        cf_em = ax_em.contourf(X_em, Y_em, hist_em_smooth.T, cmap='Greens', levels=15, alpha=0.8)
                        cb_em = fig_em.colorbar(cf_em, ax=ax_em, label='Time share / %')
                        
                    if len(em_rpm_lim) > 0:
                        ax_em.plot(em_rpm_lim, em_tq_drive, 'k-', linewidth=2.5, label='Max Drive Torque')
                        ax_em.plot(em_rpm_lim, em_tq_drag, 'k--', linewidth=2.5, label='Max Regen Torque')
                        
                    ax_em.set_title(f'E-Motor Operating Map ({strat})', fontweight='bold')
                    ax_em.set_xlabel('Motor speed [1/min]')
                    ax_em.set_ylabel('Motor torque [Nm]')
                    ax_em.axhline(0, color='gray', linewidth=1)
                    ax_em.set_xlim(0, 3000)
                    ax_em.set_ylim(-1600, 1600)
                    ax_em.grid(True, linestyle=':', alpha=0.7)
                    ax_em.legend(loc='upper right')
                    
                    plt.tight_layout()
                    em_plot_name = os.path.join(out_dir, f"EMotorMap_{strat}_{cycle_name}_{int(cap)}kWh.png")
                    plt.savefig(em_plot_name, dpi=300)
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
