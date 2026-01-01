import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys
import os

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

def find_optimal_point(X, Y, Fuel, SOC, target_soc=0.50):
    """
    Finds optimal point on SOC isoline.
    """
    contour = plt.contour(X, Y, SOC, levels=[target_soc], alpha=0)
    
    if not contour.allsegs or not contour.allsegs[0]:
        print(f"Warning: No solution found for SOC={target_soc}")
        # Return min fuel point overall as fallback
        idx = np.unravel_index(np.argmin(Fuel), Fuel.shape)
        return X[idx], Y[idx], Fuel[idx], [X[idx]], [Y[idx]]
        
    # Get vertices
    verts = contour.allsegs[0][0]
    x_line = verts[:,0]
    y_line = verts[:,1]
    
    # Interpolate Fuel on this line
    # Using Nearest for simplicity or LinearND
    from scipy.interpolate import RegularGridInterpolator
    rgi = RegularGridInterpolator((Y[:,0], X[0,:]), Fuel, bounds_error=False, fill_value=None)
    
    # Points to query (y, x)
    pts = np.column_stack((y_line, x_line))
    fuel_line = rgi(pts)
    
    min_idx = np.argmin(fuel_line)
    opt_s_dis = x_line[min_idx]
    opt_s_chg = y_line[min_idx]
    min_fuel = fuel_line[min_idx]
    
    return opt_s_dis, opt_s_chg, min_fuel, x_line, y_line

def plot_calibration(s_dis_ax, s_chg_ax, X, Y, Fuel, SOC, opt_point, constr_line, target_soc=0.50):
    opt_s_dis, opt_s_chg, opt_fuel = opt_point
    line_x, line_y = constr_line
    
    fig = plt.figure(figsize=(18, 6))
    
    # --- GRAPH 1: Optimization Map ---
    ax1 = fig.add_subplot(131)
    
    # Fuel Contour
    cf = ax1.contourf(X, Y, Fuel, levels=20, cmap='viridis')
    cbar = plt.colorbar(cf, ax=ax1, label='Fuel Consumed [kg]')
    
    # Constraint Line
    if len(line_x) > 1:
        ax1.plot(line_x, line_y, 'k-', linewidth=3, label=f'Target (SOC={target_soc})')
        
    # Additional Isolines (0.45, 0.55)
    for soc_iso in [0.45, 0.55]:
        contour = ax1.contour(X, Y, SOC, levels=[soc_iso], colors='k', linestyles=':', linewidths=1.5)
        if contour.allsegs and contour.allsegs[0]:
             # Just plot directly to avoid label issues if wanted, or let contour handle it
             pass

    # Optimal Marker
    ax1.plot(opt_s_dis, opt_s_chg, 'r*', markersize=15, markeredgecolor='white', label='Optimal Point')
    
    ax1.set_xlabel('Discharge Factor ($s_{dis}$)')
    ax1.set_ylabel('Charge Factor ($s_{chg}$)')
    ax1.set_title(f'Real Optimization Map\nopt: ({opt_s_dis:.2f}, {opt_s_chg:.2f}) -> {opt_fuel:.2f}kg')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.grid(True, alpha=0.3)
    
    # --- GRAPH 2: Discharge Sensitivity ---
    ax2 = fig.add_subplot(132)
    # Fixed s_chg
    idx_y = np.abs(s_chg_ax - opt_s_chg).argmin()
    fixed_val = s_chg_ax[idx_y]
    
    x_slice = X[idx_y, :]
    fuel_slice = Fuel[idx_y, :]
    soc_slice = SOC[idx_y, :]
    
    ax2.set_xlabel('Discharge Factor ($s_{dis}$)')
    ax2.set_ylabel('Fuel [kg]', color='tab:blue', fontweight='bold')
    ax2.plot(x_slice, fuel_slice, 'b-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax2.grid(True, alpha=0.3)
    
    ax2r = ax2.twinx()
    ax2r.set_ylabel('Final SOC [-]', color='tab:red', fontweight='bold')
    ax2r.plot(x_slice, soc_slice, 'r-', linewidth=2)
    ax2r.tick_params(axis='y', labelcolor='tab:red')
    
    ax2r.axhline(target_soc, color='k', linestyle='--', label='Target')
    ax2r.axhline(0.45, color='gray', linestyle=':', alpha=0.7)
    ax2r.axhline(0.55, color='gray', linestyle=':', alpha=0.7)
    
    ax2.set_title(f'Discharge Sensitivity\n(Fixed $s_{{chg}}$={fixed_val:.2f})')
    
    # --- GRAPH 3: Charge Sensitivity ---
    ax3 = fig.add_subplot(133)
    # Fixed s_dis
    idx_x = np.abs(s_dis_ax - opt_s_dis).argmin()
    fixed_val_x = s_dis_ax[idx_x]
    
    y_slice = Y[:, idx_x]
    fuel_slice_3 = Fuel[:, idx_x]
    soc_slice_3 = SOC[:, idx_x]
    
    ax3.set_xlabel('Charge Factor ($s_{chg}$)')
    ax3.set_ylabel('Fuel [kg]', color='tab:blue', fontweight='bold')
    ax3.plot(y_slice, fuel_slice_3, 'b-', linewidth=2)
    ax3.tick_params(axis='y', labelcolor='tab:blue')
    ax3.grid(True, alpha=0.3)
    
    ax3r = ax3.twinx()
    ax3r.set_ylabel('Final SOC [-]', color='tab:red', fontweight='bold')
    ax3r.plot(y_slice, soc_slice_3, 'r-', linewidth=2)
    ax3r.tick_params(axis='y', labelcolor='tab:red')
    ax3r.axhline(target_soc, color='k', linestyle='--')
    ax3r.axhline(0.45, color='gray', linestyle=':', alpha=0.7)
    ax3r.axhline(0.55, color='gray', linestyle=':', alpha=0.7)
    
    ax3.set_title(f'Charge Sensitivity\n(Fixed $s_{{dis}}$={fixed_val_x:.2f})')
    
    plt.tight_layout()
    plt.savefig('calibration_real_plots.png', dpi=150)
    print("Saved calibration_real_plots.png")

def main():
    csv_path = 'calibration_results.csv' # in current dir (Calibration)
    target_soc = 0.50
    
    print("Loading Real Data...")
    s_dis, s_chg, X, Y, Fuel, SOC = load_real_data(csv_path)
    
    print("Finding Optimal Point...")
    opt_s_dis, opt_s_chg, min_fuel, lx, ly = find_optimal_point(X, Y, Fuel, SOC, target_soc)
    print(f"Optimal Factors: ({opt_s_dis:.3f}, {opt_s_chg:.3f}) -> Fuel: {min_fuel:.3f} kg")
    
    print("Plotting...")
    plot_calibration(s_dis, s_chg, X, Y, Fuel, SOC, (opt_s_dis, opt_s_chg, min_fuel), (lx, ly), target_soc)

if __name__ == "__main__":
    main()
