import os
import pandas as pd
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
    
    params_cap = [30.0]
    
    strategies_ecms = ['ECMS', 'A-ECMS', 'PECMS']
    # DP is run separately
    
    results = []
    
    for cycle_path in cycles:
        cycle_name = os.path.basename(cycle_path).replace(".vmod", "")
        print(f"\n=== Processing Cycle: {cycle_name} ===\n")
        
        for cap in params_cap:
            # 1. ECMS Variants
            for strat in strategies_ecms:
                prefix = f"{cycle_name}_{strat}_{int(cap)}kWh"
                print(f"-> Strategy: {strat}, Cap: {cap}")
                fuel = run_ecms_simulation(strategy=strat, cycle_file=cycle_path, bat_capacity_kwh=cap, output_prefix=prefix)
                
                results.append({
                    'Cycle': cycle_name,
                    'Capacity_kWh': cap,
                    'Strategy': strat,
                    'Fuel_kg': fuel
                })
                
            # 2. DP
            prefix_dp = f"{cycle_name}_DP_{int(cap)}kWh"
            print(f"-> Strategy: DP, Cap: {cap}")
            fuel_dp = run_dp_simulation(cycle_file=cycle_path, bat_capacity_kwh=cap, output_prefix=prefix_dp)
            results.append({
                'Cycle': cycle_name,
                'Capacity_kWh': cap,
                'Strategy': 'DP',
                'Fuel_kg': fuel_dp
            })
            
    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv("batch_results.csv", index=False)
    
    print("\n\n=== BATCH SIMULATION COMPLETE ===")
    print(df_res)
    print("Results saved to batch_results.csv")

if __name__ == "__main__":
    main()
