import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
import os


def _fmt3_no_round(value):
    """Format to x.xxx using truncation (no rounding up)."""
    return f"{np.trunc(float(value) * 1000.0) / 1000.0:.3f}"

def load_real_data(csv_path):
    """
    Loads real simulation results from CSV and pivots to meshgrid.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    
    # Pivot to Matrix
    # Assumes regular grid
    pivot_fuel = df.pivot(index='s_chg', columns='s_dis', values='fuel_kg')
    pivot_soc = df.pivot(index='s_chg', columns='s_dis', values='final_soc')
    
    # Extract Axes
    s_dis_ax = pivot_fuel.columns.values.astype(float)
    s_chg_ax = pivot_fuel.index.values.astype(float)
    
    X, Y = np.meshgrid(s_dis_ax, s_chg_ax)
    Fuel = pivot_fuel.values
    SOC = pivot_soc.values
    
    # Fuel is in kg. Convert to L/100km? 
    # Or keep as kg? User concept used L/100km.
    # Truck Cycle ~ 39kg. Distance? 
    # Let's stick to Total Fuel [kg] for accuracy, or estimate Distance.
    # VECTO cycle distance is constant.
    # For now, plotting 'Fuel [kg]'.
    
    return s_dis_ax, s_chg_ax, X, Y, Fuel, SOC

def find_optimal_point(X, Y, Fuel, SOC, target_soc=0.30):
    """
    Finds optimal point in grid where SOC >= target_soc minimizing Fuel.
    Also extracts target_soc contour for plotting.
    """
    from scipy.interpolate import RegularGridInterpolator

    fuel_interp = RegularGridInterpolator(
        (Y[:, 0], X[0, :]),
        Fuel,
        bounds_error=False,
        fill_value=None,
    )

    # 1. Find min fuel in grid where SOC >= target_soc
    valid_mask = SOC >= target_soc
    if not np.any(valid_mask):
        print(f"Warning: No points found with SOC >= {target_soc}")
        # Fallback to absolute max SOC point
        idx_opt = np.unravel_index(np.argmax(SOC), SOC.shape)
    else:
        valid_fuel = np.where(valid_mask, Fuel, np.inf)
        idx_opt = np.unravel_index(np.argmin(valid_fuel), Fuel.shape)
        
    opt_s_dis = X[idx_opt]
    opt_s_chg = Y[idx_opt]
    min_fuel = float(fuel_interp((opt_s_chg, opt_s_dis)))
    
    # 2. Extract contour for plotting
    fig_temp = plt.figure()
    ax_temp = fig_temp.add_subplot(111)
    contour = ax_temp.contour(X, Y, SOC, levels=[target_soc], alpha=0)
    
    if not contour.allsegs or not contour.allsegs[0]:
        x_line = np.array([])
        y_line = np.array([])
        fuel_line = np.array([])
    else:
        verts = contour.allsegs[0][0]
        x_line = verts[:,0]
        y_line = verts[:,1]
        pts = np.column_stack((y_line, x_line))
        fuel_line = fuel_interp(pts)
    plt.close(fig_temp)
    
    return opt_s_dis, opt_s_chg, min_fuel, x_line, y_line, fuel_line

def plot_calibration(s_dis_ax, s_chg_ax, X, Y, Fuel, SOC, opt_point, constr_line, target_soc=0.30):
    opt_s_dis, opt_s_chg, opt_fuel = opt_point
    line_x, line_y, line_fuel = constr_line

    # Build a dense interpolated surface for smoother 3D rendering.
    try:
        from scipy.interpolate import RegularGridInterpolator

        s_dis_fine = np.linspace(float(np.nanmin(s_dis_ax)), float(np.nanmax(s_dis_ax)), max(220, len(s_dis_ax) * 8))
        s_chg_fine = np.linspace(float(np.nanmin(s_chg_ax)), float(np.nanmax(s_chg_ax)), max(220, len(s_chg_ax) * 8))
        X3d, Y3d = np.meshgrid(s_dis_fine, s_chg_fine)

        fuel_interp = RegularGridInterpolator(
            (s_chg_ax, s_dis_ax),
            Fuel,
            bounds_error=False,
            fill_value=np.nan,
        )
        interp_pts = np.column_stack((Y3d.ravel(), X3d.ravel()))
        Fuel3d = fuel_interp(interp_pts).reshape(X3d.shape)
    except Exception:
        X3d, Y3d, Fuel3d = X, Y, Fuel
    
    fig = plt.figure(figsize=(16, 6))
    
    # --- GRAPH 1: 3D Surface Plot ---
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(
        X3d,
        Y3d,
        Fuel3d,
        cmap='viridis',
        alpha=1.0,
        rcount=Fuel3d.shape[0],
        ccount=Fuel3d.shape[1],
        linewidth=0,
        edgecolor='none',
        antialiased=False,
        shade=True,
        zorder=1,
    )
    # Rasterize surface in PDF export to prevent vector hairline artifacts between polygons.
    surf.set_rasterized(True)
    surf.set_edgecolor('none')
    surf.set_linewidth(0.0)

    # Add a subtle sparse mesh overlay so structure is visible without heavy faceting.
    mesh_stride_r = max(1, Fuel3d.shape[0] // 26)
    mesh_stride_c = max(1, Fuel3d.shape[1] // 26)
    ax1.plot_wireframe(
        X3d,
        Y3d,
        Fuel3d,
        rstride=mesh_stride_r,
        cstride=mesh_stride_c,
        color='white',
        linewidth=0.20,
        alpha=0.22,
    )

    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Spotřeba paliva [kg]')
    
    # Use ax.plot instead of scatter for better visibility over surfaces, and elevate it clearly above the surface
    ax1.plot([opt_s_dis], [opt_s_chg], [opt_fuel ], color='red', marker='*', markersize=10, label='Optimální bod', zorder=100, clip_on=False)
    
    if len(line_x) > 0:
        # Also give the black constraint line a small Z boost
        ax1.plot(line_x, line_y, line_fuel + 0.2, color='black', linewidth=4, label=f'Cíl SOC {int(target_soc * 100)}%', zorder=10)

    # Shift the camera slightly to the right so the calibration surface is clearer.
    ax1.view_init(elev=25, azim=-125)
        
    ax1.set_xlabel('Vybíjecí faktor ($s_{dis}$)')
    ax1.set_ylabel('Rekuperační faktor ($s_{chg}$)')
    ax1.set_zlabel('Spotřeba paliva [kg]')
    ax1.set_xlim(0.0, float(np.nanmax(X)))
    ax1.set_title('3D optimalizační plocha')
    ax1.legend()

    # --- GRAPH 2: Optimization Map ---
    ax2 = fig.add_subplot(122)
    
    # Fuel Contour
    cf = ax2.contourf(X, Y, Fuel, levels=20, cmap='viridis')
    cbar = plt.colorbar(cf, ax=ax2, label='Spotřeba paliva [kg]')
    
    # Constraint Line
    if len(line_x) > 1:
        ax2.plot(line_x, line_y, 'k-', linewidth=3, label=f'Cíl SOC {int(target_soc * 100)}%')
        
    # Additional Isolines (0.25, 0.35)
    for soc_iso in [0.25, 0.35]:
        contour = ax2.contour(X, Y, SOC, levels=[soc_iso], colors='k', linestyles=':', linewidths=1.5)
        # Optionally add inline labels to the isolines 
        # ax2.clabel(contour, inline=True, fontsize=8, fmt=f'{int(soc_iso * 100)}%%')

    # Optimal Marker
    ax2.plot(opt_s_dis, opt_s_chg, 'r*', markersize=15, markeredgecolor='white', label='Optimální bod')
    
    ax2.set_xlabel('Vybíjecí faktor ($s_{dis}$)')
    ax2.set_ylabel('Rekuperační faktor ($s_{chg}$)')
    ax2.set_title(
        f'Optimalizační mapa'
    )
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('calibration_real_plots.pdf', dpi=150, bbox_inches='tight', pad_inches=0.05)
    print("Saved calibration_real_plots.pdf")

def main():
    csv_path = 'Calibration/calibration_results.csv' # in current dir (Calibration)
    target_soc = 0.30
    
    print("Loading Real Data...")
    s_dis, s_chg, X, Y, Fuel, SOC = load_real_data(csv_path)
    
    print("Finding Optimal Point...")
    opt_s_dis, opt_s_chg, min_fuel, lx, ly, lf = find_optimal_point(X, Y, Fuel, SOC, target_soc)
    print(f"Optimal Factors: DIS: ({opt_s_dis:.3f}, CH: {opt_s_chg:.3f}) -> Fuel: {_fmt3_no_round(min_fuel)} kg")
    
    print("Plotting...")
    plot_calibration(s_dis, s_chg, X, Y, Fuel, SOC, (opt_s_dis, opt_s_chg, min_fuel), (lx, ly, lf), target_soc)

if __name__ == "__main__":
    main()
