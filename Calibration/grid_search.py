import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vecto_loader import VectoLoader
from p2_hybrid import P2HybridTruck
from ecms_controller import ECMS_Controller
import time
import multiprocessing

# Global variables for worker processes
worker_truck = None
worker_cycle_data = None
worker_t_reqs = None
worker_q_max = None
worker_dt_arr = None
worker_rpms = None
worker_times = None

def init_worker(engine_map, motor_map, motor_param, bat_param, bat_ocv, bat_res, cycle_path):
    """
    Initializer for worker processes. Loads the Heavy model once.
    """
    global worker_truck, worker_cycle_data, worker_t_reqs, worker_q_max
    global worker_dt_arr, worker_rpms, worker_times
    
    loader = VectoLoader()
    worker_truck = P2HybridTruck(loader)
    worker_truck.load_components(
        engine_map, motor_map, motor_param, bat_param, bat_ocv, bat_res
    )
    
    # Load Cycle
    worker_cycle_data = loader.read_vmod(cycle_path)
    worker_t_reqs = worker_truck.calc_backward_physics(worker_cycle_data)
    
    # Pre-calc constants
    v_nom = worker_truck.get_ocv(0.5)
    cap_kwh = worker_truck.bat_params.get('Capacity', 100.0) 
    worker_q_max = (cap_kwh * 3.6e6) / v_nom
    print(f"DEBUG WORKER: V_nom={v_nom:.2f} V, Cap={cap_kwh:.2f} kWh, Q_max={worker_q_max:.2f} As")
    
    worker_times = worker_cycle_data['time'].values
    worker_rpms = worker_cycle_data['rpm_ice'].values
    
    # Dt
    worker_dt_arr = np.diff(worker_times, prepend=worker_times[0])
    worker_dt_arr[0] = 0.5

def run_simulation_task(params):
    """
    Worker function.
    params: (s_dis, s_chg)
    """
    s_dis, s_chg = params
    
    # Use global worker objects
    controller = ECMS_Controller(worker_truck, s_dis=s_dis, s_chg=s_chg, q_lhv=42700.0)
    
    soc = 0.50
    total_fuel_g = 0.0
    
    # Fast Loop
    # Access globals
    times = worker_times
    rpms = worker_rpms
    t_reqs = worker_t_reqs
    dts = worker_dt_arr
    q_max = worker_q_max
    truck = worker_truck
    
    n_steps = len(times)
    
    for i in range(n_steps):
        t_req = t_reqs[i]
        rpm = rpms[i]
        dt = dts[i]
        
        # Optimize
        _, _, _, p_chem, fuel_gs = controller.decide_split(t_req, rpm, soc)
        
        # Accumulate Fuel
        total_fuel_g += fuel_gs * dt
        
        # if i == 0 or i == 1000:
        #     print(f"DEBUG GS Step {i}: p_chem={p_chem:.4f}, Fuel={fuel_gs:.4f}, SOC_start={soc:.6f}")

        # Update SOC
        v_oc = truck.get_ocv(soc)
        if v_oc > 0:
            i_bat = p_chem / v_oc
            d_soc = - i_bat / q_max
            soc += d_soc * dt
            
        # Clamping
        soc = max(0.0, min(1.0, soc))
        
    return (s_dis, s_chg, total_fuel_g / 1000.0, soc)

def main():
    # 1. Paths
    engine_map = '/root/ECMS_Python/Engine/325kW.vmap'
    motor_map = '/root/ECMS_Python/Emotor/EM_Map - kopie.vemo'
    motor_param = '/root/ECMS_Python/Emotor/P2_Group5_EM.vem'
    bat_param = '/root/ECMS_Python/Emotor/P2_Group5_REESS.vreess'
    bat_ocv = '/root/ECMS_Python/Emotor/REESS_SOC_curve.vbatv'
    bat_res = '/root/ECMS_Python/Emotor/REESS_Internal_Resistance.vbatr'
    cycle_path = '/root/ECMS_Python/Driving Cycle/Class5_Tractor_DECL_LongHaulEMSReferenceLoad.vmod'
    
    # 2. Grid Definition
    # High Resolution as requested
    s_min = 1.8
    s_max = 4.0
    steps = 10 
    
    print(f"Defining Grid: {steps}x{steps} (Total {steps*steps} simulations)")
    
    s_vals = np.linspace(s_min, s_max, steps)
    s_dis_grid, s_chg_grid = np.meshgrid(s_vals, s_vals)
    
    # Prepare tasks
    tasks = []
    for s_d, s_c in zip(s_dis_grid.flatten(), s_chg_grid.flatten()):
        tasks.append((float(s_d), float(s_c)))
        
    print(f"Starting Parallel Grid Search with {multiprocessing.cpu_count()} cores...")
    start_time = time.time()
    
    # Pool
    with multiprocessing.Pool(processes=multiprocessing.cpu_count(), 
                              initializer=init_worker, 
                              initargs=(engine_map, motor_map, motor_param, bat_param, bat_ocv, bat_res, cycle_path)) as pool:
        
        # Use imap_unordered to track progress
        total_tasks = len(tasks)
        results = []
        start_time = time.time()
        
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
    
    # 3. Process Results
    df = pd.DataFrame(results, columns=['s_dis', 's_chg', 'fuel_kg', 'final_soc'])
    df.to_csv('calibration_results.csv', index=False)
    print("Saved calibration_results.csv")
    
    # 4. Plotting
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pivot for surface plot
    pivot_fuel = df.pivot(index='s_chg', columns='s_dis', values='fuel_kg')
    pivot_soc = df.pivot(index='s_chg', columns='s_dis', values='final_soc')
    
    X = pivot_fuel.columns.values
    Y = pivot_fuel.index.values
    X, Y = np.meshgrid(X, Y)
    Z = pivot_fuel.values
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('EF Discharge (Penalty)')
    ax.set_ylabel('EF Charge (Reward/Cost)')
    ax.set_zlabel('Fuel Consumption [kg]')
    ax.set_title('ECMS Calibration (High Res 25x25)')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Fuel [kg]')
    plt.savefig('calibration_map_3d.png')
    print("Saved calibration_map_3d.png")

if __name__ == "__main__":
    main()
