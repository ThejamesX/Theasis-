import sys
import os
import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from A_ECMS_Implementation.aecms_controller import AECMS_Controller


# Worker globals for multiprocessing
worker_truck = None
worker_t_req_arr = None
worker_dts = None
worker_rpms = None
worker_dist_arr = None
worker_total_dist = None
worker_steps = None
worker_q_max_as = None
worker_target_start = None
worker_target_end = None


def init_worker(vmod_path, vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path):
    """Initialize heavy model data once per worker process."""
    global worker_truck, worker_t_req_arr, worker_dts, worker_rpms
    global worker_dist_arr, worker_total_dist, worker_steps, worker_q_max_as
    global worker_target_start, worker_target_end

    loader = VectoLoader()
    truck = P2HybridTruck(loader)
    truck.load_components(vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path)

    cycle_df = loader.read_vmod(vmod_path)
    if cycle_df is None or len(cycle_df) == 0:
        raise ValueError(
            f"Failed to load valid cycle data from {vmod_path}. "
            "Check the .vmod file format and column mapping."
        )

    t_req_arr = truck.calc_backward_physics(cycle_df)
    dts = np.diff(cycle_df['time'].values, prepend=0)
    dts[0] = 0.5
    rpms = cycle_df['rpm_ice'].values
    steps = len(cycle_df)
    q_max_as = (120.0 * 3.6e6) / truck.get_ocv(0.7)

    target_start = 0.70
    target_end = 0.30

    if 'dist_accum_m' in cycle_df.columns:
        dist_arr = cycle_df['dist_accum_m'].values
    else:
        dist_arr = np.linspace(0, 1, steps)
    total_dist = dist_arr[-1] if dist_arr[-1] > 0 else 1.0

    worker_truck = truck
    worker_t_req_arr = t_req_arr
    worker_dts = dts
    worker_rpms = rpms
    worker_dist_arr = dist_arr
    worker_total_dist = total_dist
    worker_steps = steps
    worker_q_max_as = q_max_as
    worker_target_start = target_start
    worker_target_end = target_end


def run_simulation_task(params):
    """Run one A-ECMS simulation for a single (kp_dis, kp_chg) pair."""
    kd, kc = params

    ctrl = AECMS_Controller(worker_truck, kp_dis=kd, kp_chg=kc, target_soc=worker_target_start)
    soc = worker_target_start
    total_fuel = 0.0

    for i in range(worker_steps):
        curr_target = worker_target_start - (worker_dist_arr[i] / worker_total_dist) * (
            worker_target_start - worker_target_end
        )
        ctrl.target_soc = curr_target

        res = ctrl.decide_split(worker_t_req_arr[i], worker_rpms[i], soc)
        fuel_rate = res[4]
        p_chem = res[3]

        total_fuel += fuel_rate * worker_dts[i]

        u_oc = worker_truck.get_ocv(soc)
        dsoc = -((p_chem / u_oc) * worker_dts[i]) / worker_q_max_as
        soc += dsoc

    final_soc = soc
    soc_dev = abs(final_soc - worker_target_end) * 100
    fuel_kg = total_fuel / 1000.0

    return kd, kc, fuel_kg, soc_dev, final_soc

def grid_search_aecms(num_workers=12):
    # 1. Resolve paths
    
    # Paths
    base_dir = parent_dir
    cycle_candidates = [
        os.path.join(base_dir, "Driving Cycle", "LongHaulEMSReferenceLoad.vmod"),
        os.path.join(base_dir, "Driving Cycle", "Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod"),
    ]
    vmod_path = next((p for p in cycle_candidates if os.path.exists(p)), None)
    if vmod_path is None:
        tried = "\n".join(cycle_candidates)
        raise FileNotFoundError(f"No driving cycle file found. Tried:\n{tried}")

    vmap_path = os.path.join(base_dir, "Engine/325kW.vmap")
    vem_path = os.path.join(base_dir, "Emotor/P2_Group5_EM.vem")
    vemo_path = os.path.join(base_dir, "Emotor/EM_Map - kopie.vemo") 
    vreess_path = os.path.join(base_dir, "Emotor/P2_Group5_REESS.vreess")
    vbatv_path = os.path.join(base_dir, "Emotor/REESS_SOC_curve.vbatv")
    vbatr_path = os.path.join(base_dir, "Emotor/REESS_Internal_Resistance.vbatr")

    # 2. Grid Setup
    kp_dis_vals = np.linspace(5, 80, 60)
    kp_chg_vals = np.linspace(0.01, 10, 25)

    tasks = []
    for kd in kp_dis_vals:
        for kc in kp_chg_vals:
            tasks.append((float(kd), float(kc)))

    workers = max(1, min(int(num_workers), multiprocessing.cpu_count()))
    print(f"Starting Parallel Grid Search (Total {len(tasks)} runs, workers={workers})...")

    start_time = time.time()
    results = []

    with multiprocessing.Pool(
        processes=workers,
        initializer=init_worker,
        initargs=(vmod_path, vmap_path, vemo_path, vem_path, vreess_path, vbatv_path, vbatr_path),
    ) as pool:
        total_tasks = len(tasks)
        print(f"Progress: 0.0% (0/{total_tasks})", end='\r')

        for i, res in enumerate(pool.imap_unordered(run_simulation_task, tasks), 1):
            results.append(res)
            if i % 10 == 0 or i == total_tasks:
                elapsed = time.time() - start_time
                pct = (i / total_tasks) * 100
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total_tasks - i) / rate if rate > 0 else 0
                print(f"Progress: {pct:.1f}% ({i}/{total_tasks}) - ETA: {remaining:.0f}s   ", end='\r')

    print(f"\nGrid Search Complete in {time.time() - start_time:.1f}s")

    # 3. Analyze
    df = pd.DataFrame(results, columns=['kp_dis', 'kp_chg', 'fuel', 'dev', 'final_soc'])
    df['soc_err_pct'] = (df['final_soc'] - 0.30) * 100
    df.to_csv('aecms_grid_results.csv', index=False)
    
    # Best Fuel with Dev < 1%
    valid = df[df['dev'] < 0.8]
    if not valid.empty:
        best = valid.loc[valid['fuel'].idxmin()]
        print("\n--- BEST VALID RESULT (Dev < 0.8%) ---")
        print(best)
    else:
        print("\nNo run satisfied deviation < 0.8%. Best overall fuel:")
        best = df.loc[df['fuel'].idxmin()]
        print(best)
        
    # 4. Plot Heatmap
    # Reshape
    pivot = df.pivot(index='kp_dis', columns='kp_chg', values='fuel')
    
    plt.figure(figsize=(10, 8))
    plt.imshow(pivot, cmap='viridis_r', interpolation='nearest', origin='lower',
               extent=[kp_chg_vals.min(), kp_chg_vals.max(), kp_dis_vals.min(), kp_dis_vals.max()])
    plt.colorbar(label='Spotřeba paliva [kg]')
    plt.xlabel('Kp nabíjení')
    plt.ylabel('Kp vybíjení')
    plt.title('Mřížkové vyhledávání spotřeby paliva A-ECMS')
    plt.savefig('aecms_grid_heatmap.pdf', bbox_inches='tight', pad_inches=0.05)
    print("Saved heatmap to aecms_grid_heatmap.pdf")
    
    # 5. 3D Surface Plot (Mesh Grid)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(kp_chg_vals, kp_dis_vals)
    Z = pivot.values
    
    # Plot Surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis_r', edgecolor='none', alpha=0.9)
    # View from the other side
    ax.view_init(elev=30, azim=135)
    
    # Labels
    ax.set_xlabel('Kp nabíjení')
    ax.set_ylabel('Kp vybíjení')
    ax.set_zlabel('Spotřeba paliva [kg]')
    ax.set_title(f'Prostor optimalizace spotřeby paliva A-ECMS\nNejlepší: {best["fuel"]:.4f} kg')
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Spotřeba paliva [kg]')
    
    plt.savefig('aecms_3d_surface.pdf', bbox_inches='tight', pad_inches=0.05)
    print("Saved 3D surface plot to aecms_3d_surface.pdf")

if __name__ == "__main__":
    grid_search_aecms(num_workers=12)
