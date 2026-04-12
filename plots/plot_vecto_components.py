"""
VECTO Component Plots for Academic Article
Generates three publication-quality figures:
  1. Battery OCV vs SOC curve
  2. Engine BSFC contour map
  3. Electric Motor efficiency contour map
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter

# --- Style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = BASE_DIR


# ============================================================
# PLOT 1: Battery OCV vs SOC
# ============================================================
def plot_ocv_soc():
    filepath = os.path.join(ROOT_DIR, "Emotor", "REESS_SOC_curve.vbatv")
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]

    soc = df.iloc[:, 0].values
    ocv = df.iloc[:, 1].values

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(soc, ocv, 'o-', color='#1f77b4', linewidth=2.0, markersize=7,
            markerfacecolor='white', markeredgewidth=1.8, markeredgecolor='#1f77b4')

    ax.set_xlabel('Stav nabití [%]')
    ax.set_ylabel('Napětí naprázdno [V]')
    ax.set_xlim(-2, 102)
    ax.set_ylim(620, 710)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.annotate(f'{ocv[0]:.1f} V', xy=(soc[0], ocv[0]),
                xytext=(soc[0]+8, ocv[0]-5), fontsize=10, color='#333333',
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))
    ax.annotate(f'{ocv[-1]:.1f} V', xy=(soc[-1], ocv[-1]),
                xytext=(soc[-1]-22, ocv[-1]+3), fontsize=10, color='#333333',
                arrowprops=dict(arrowstyle='->', color='#666666', lw=1.2))

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "battery_ocv_soc.pdf")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ============================================================
# PLOT 2: Engine BSFC Map (Brake Specific Fuel Consumption)
# ============================================================
def plot_bsfc_map():
    filepath = os.path.join(ROOT_DIR, "Engine", "325kW.vmap")
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    rpm_col = next(c for c in df.columns if 'speed' in c or 'rpm' in c)
    tq_col = next(c for c in df.columns if 'torque' in c)
    fc_col = next(c for c in df.columns if 'fuel' in c)

    rpm_raw = df[rpm_col].values
    torque_raw = df[tq_col].values
    fuel_gh_raw = df[fc_col].values

    # --- Compute BSFC for positive-torque points ---
    mask = (torque_raw >= 0) & (fuel_gh_raw >= 0)
    rpm_f = rpm_raw[mask]
    tq_f = torque_raw[mask]
    fc_f = fuel_gh_raw[mask]

    omega_f = rpm_f * 2.0 * np.pi / 60.0
    power_kw_f = tq_f * omega_f / 1000.0
    
    bsfc = np.zeros_like(power_kw_f)
    valid = power_kw_f > 1.0
    bsfc[valid] = fc_f[valid] / power_kw_f[valid]  # g/kWh
    bsfc[~valid] = 400.0  # Cap high value to fill contour gap down to 0 Nm

    # --- Use Delaunay triangulation on normalised coordinates ---
    rpm_scale = rpm_f.max() - rpm_f.min()
    tq_scale = tq_f.max() - tq_f.min()
    rpm_norm = (rpm_f - rpm_f.min()) / rpm_scale
    tq_norm = (tq_f - tq_f.min()) / tq_scale

    triang = tri.Triangulation(rpm_norm, tq_norm)

    # Refine the triangulation for smooth contours
    refiner = tri.UniformTriRefiner(triang)
    triang_fine, bsfc_fine = refiner.refine_field(bsfc, subdiv=3)

    # Convert refined triangulation back to physical coordinates
    rpm_fine = triang_fine.x * rpm_scale + rpm_f.min()
    tq_fine = triang_fine.y * tq_scale + tq_f.min()
    triang_phys = tri.Triangulation(rpm_fine, tq_fine, triang_fine.triangles)

    # Calculate exact full-load point for each mapping RPM block
    drpm = np.diff(rpm_raw)
    block_ends = np.where(np.abs(drpm) > 5.0)[0]
    block_ends = np.append(block_ends, len(rpm_raw)-1)

    max_tq_curve = []
    for i, end_idx in enumerate(block_ends):
        start_idx = 0 if i == 0 else block_ends[i-1] + 1
        rp_block = rpm_raw[start_idx:end_idx+1]
        tq_block = torque_raw[start_idx:end_idx+1]
        if tq_block.max() > 0:
            max_tq_curve.append((rp_block[np.argmax(tq_block)], tq_block.max()))

    max_tq_curve = np.array(max_tq_curve)
    rpm_unique = max_tq_curve[:,0]
    tq_max = max_tq_curve[:,1]
    
    # --- Mask triangles outside the torque envelope ---
    fld_drive_interp = interp1d(rpm_unique, tq_max, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
    centroids_rpm = np.mean(rpm_fine[triang_phys.triangles], axis=1)
    centroids_tq = np.mean(tq_fine[triang_phys.triangles], axis=1)
    fl_drive_at = fld_drive_interp(np.clip(centroids_rpm, rpm_unique.min(), rpm_unique.max()))
    mask_tri = (centroids_tq > fl_drive_at)
    triang_phys.set_mask(mask_tri)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # Non-uniform levels: fine 2 g/kWh steps in optimal zone, coarser above
    levels_fine = np.arange(182, 210, 2)     # 14 levels in sweet spot
    levels_coarse = np.arange(210, 320, 10)  # coarser above
    levels = np.concatenate([levels_fine, levels_coarse])

    cf = ax.tricontourf(triang_phys, bsfc_fine, levels=levels,
                        cmap='Spectral_r', extend='both')
    cs = ax.tricontour(triang_phys, bsfc_fine,
                       levels=np.arange(185, 310, 5),
                       colors='black', linewidths=0.4, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=7, fmt='%d')

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Měrná spotřeba paliva [g/kWh]')

    ax.set_xlabel('Otáčky motoru [1/min]')
    ax.set_ylabel('Točivý moment motoru [Nm]')
    
    ax.plot(rpm_unique, tq_max, 'k-', linewidth=2.0, label='Max. točivý moment')
    ax.legend(loc='upper right')
    
    ax.set_xlim(rpm_raw.min(), rpm_raw.max())
    ax.set_ylim(0, tq_f.max() * 1.05)
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "engine_bsfc_map.pdf")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ============================================================
# PLOT 3: Electric Motor Efficiency Map
# ============================================================
def plot_em_efficiency():
    filepath = os.path.join(ROOT_DIR, "Emotor", "EM_Map - kopie.vemo")
    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = [c.strip().lower() for c in df.columns]

    rpm_col = next(c for c in df.columns if 'n' in c and 'rpm' in c)
    tq_col = next(c for c in df.columns if 't' in c and 'nm' in c)
    pel_col = next(c for c in df.columns if 'p_el' in c)

    rpm = df[rpm_col].values
    torque = df[tq_col].values
    p_el_kw = df[pel_col].values

    # Mechanical power [kW]
    omega = rpm * 2.0 * np.pi / 60.0
    p_mech_kw = torque * omega / 1000.0

    # --- Compute efficiency for ALL points ---
    efficiency = np.full_like(rpm, np.nan, dtype=float)

    # Motoring (T > 0, P_el > 0): eta = P_mech / P_el
    mot_mask = (torque > 0.5) & (p_el_kw > 0.01) & (rpm > 10)
    efficiency[mot_mask] = (p_mech_kw[mot_mask] / p_el_kw[mot_mask]) * 100.0

    # Generating (T < 0, P_el < 0): eta = |P_el| / |P_mech|
    gen_mask = (torque < -0.5) & (p_el_kw < -0.01) & (rpm > 10)
    efficiency[gen_mask] = (np.abs(p_el_kw[gen_mask]) / np.abs(p_mech_kw[gen_mask])) * 100.0

    # Clamp to physical range
    efficiency = np.clip(efficiency, 50, 99)

    # --- Full-load curve for overlay ---
    fld_path = os.path.join(ROOT_DIR, "Emotor", "EM_fld.vemp")
    fld_df = pd.read_csv(fld_path, skipinitialspace=True)
    fld_df.columns = [c.strip().lower() for c in fld_df.columns]
    fld_rpm = fld_df.iloc[:, 0].values
    fld_tq_drive = fld_df.iloc[:, 1].values
    fld_tq_drag = fld_df.iloc[:, 2].values

    # --- Single combined 4-quadrant plot using tricontourf ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Interpolators for torque envelope masking
    fld_drive_interp = interp1d(fld_rpm, fld_tq_drive, kind='linear',
                                 bounds_error=False, fill_value='extrapolate')
    fld_drag_interp = interp1d(fld_rpm, fld_tq_drag, kind='linear',
                                bounds_error=False, fill_value='extrapolate')
    # Combine ALL valid data for a single triangulation (no gap between quadrants)
    valid = (mot_mask | gen_mask) & ~np.isnan(efficiency)
    r_v = rpm[valid]
    t_v = torque[valid]
    e_v = efficiency[valid]

    # Normalise coordinates for balanced triangulation
    rpm_scale = r_v.max() - r_v.min()
    tq_range = t_v.max() - t_v.min()
    rpm_norm = (r_v - r_v.min()) / rpm_scale
    tq_norm = (t_v - t_v.min()) / tq_range

    triang_norm = tri.Triangulation(rpm_norm, tq_norm)
    refiner = tri.UniformTriRefiner(triang_norm)
    triang_fine, eff_fine = refiner.refine_field(e_v, subdiv=3)

    # Convert back to physical coordinates
    rpm_fine = triang_fine.x * rpm_scale + r_v.min()
    tq_fine = triang_fine.y * tq_range + t_v.min()
    triang_phys = tri.Triangulation(rpm_fine, tq_fine, triang_fine.triangles)

    # Mask triangles outside the torque envelope
    centroids_rpm = np.mean(rpm_fine[triang_phys.triangles], axis=1)
    centroids_tq = np.mean(tq_fine[triang_phys.triangles], axis=1)
    fl_drive_at = fld_drive_interp(np.clip(centroids_rpm, fld_rpm.min(), fld_rpm.max()))
    fl_drag_at = fld_drag_interp(np.clip(centroids_rpm, fld_rpm.min(), fld_rpm.max()))
    mask_tri = (centroids_tq > fl_drive_at) | (centroids_tq < fl_drag_at)
    triang_phys.set_mask(mask_tri)

    # Contour fill
    levels = np.arange(60, 100, 2)
    cf = ax.tricontourf(triang_phys, eff_fine, levels=levels,
                        cmap='RdYlGn', extend='both')
    cs = ax.tricontour(triang_phys, eff_fine, levels=[65, 70, 75, 80, 85, 90, 92, 95],
                       colors='black', linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f%%')

    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label('Účinnost [%]')

    # Draw full-load envelope
    ax.plot(fld_rpm, fld_tq_drive, 'k-', linewidth=2.0, label='Max. hnací moment')
    ax.plot(fld_rpm, fld_tq_drag, 'k--', linewidth=2.0, label='Max. rekuperační moment')

    # Zero-torque line
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='-')

    # Labels
    ax.text(3500, 600, 'MOTOROVÝ REŽIM', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#333333', alpha=0.5)
    ax.text(3500, -600, 'GENERÁTOROVÝ REŽIM', fontsize=13, fontweight='bold',
            ha='center', va='center', color='#333333', alpha=0.5)

    ax.set_xlabel('Otáčky motoru [1/min]')
    ax.set_ylabel('Točivý moment motoru [Nm]')
    # Set X limit as requested
    ax.set_xlim(200, 5000)
    ax.set_ylim(fld_tq_drag.min() * 1.05, fld_tq_drive.max() * 1.05)
    ax.legend(loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.3)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "emotor_efficiency_map.pdf")
    plt.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")


# ============================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Generating VECTO component plots...\n")

    plot_ocv_soc()
    plot_bsfc_map()
    plot_em_efficiency()

    print("\nAll plots generated successfully.")
