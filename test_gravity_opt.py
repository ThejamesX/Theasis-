import numpy as np
import sys
import os

# Adjust path to find modules
sys.path.append("/root/ECMS_Python")
sys.path.append("/root/ECMS_Python/P_ECMS")

from P_ECMS.gravity_supervisor import GravitySupervisor

# --- Mocks ---
class MockVehicle:
    def get_ocv(self, soc):
        return 300.0  # Constant OCV 300V

class MockController:
    def __init__(self):
        self.s_dis = 2.4
        self.s_chg = 2.4 * 0.95
    
    def decide_split(self, t_req, rpm, soc):
        # Mock behavior: 
        # If s is high, use Engine (battery power = 0 or negative)
        # If s is low, use Battery (battery power = positive)
        
        # Simple Logic:
        # P_batt_mock = Load - s * Factor
        # So higher s -> Lower P_batt (Charge/Less Discharge)
        
        load = 10000 # 10kW load
        factor = 4000
        p_batt = load - (self.s_dis * factor)
        
        # Return Dummy tuple, we only care about index 3 (p_chem/p_batt)
        return (0, 0, 0, p_batt)

# --- Test ---
def test_gravity_optimization():
    print("--- Starting Gravity Supervisor Verification ---")
    
    veh = MockVehicle()
    ctrl = MockController()
    
    # Init Supervisor
    # Start SOC 0.5, End SOC 0.5
    sup = GravitySupervisor(veh, ctrl, total_dist_m=10000, q_max_as=100*3600)
    
    print(f"Initial State: last_opt_s={sup.last_opt_s}, ratio={sup.ratio:.4f}")
    
    # 1. Create Dummy Horizon
    # 10 steps, 1 second each
    horizon = {
        'dist_covered': 0,
        'alts': [100]*10, # Flat
        'rpms': [2000]*10,
        't_reqs': [100]*10,
        'dts': [1.0]*10
    }
    
    # 2. Run Optimization Step 1
    # Current SOC is target (0.5). Mock Controller logic:
    # s=2.4 -> p_batt = 10000 - 2.4*4000 = 400. (Discharge slightly)
    # So SOC will drop below 0.5.
    # To fix, we need LESS discharge (or Charge). 
    # To use LESS battery, we need HIGHER s (cost of battery is higher).
    # Expected: s should INCREASE from 2.4.
    
    curr_soc = 0.5
    print("\n--- Step 1 Call (SOC=0.5, Target=0.5, Pred=Discharge) ---")
    best_s, target = sup.get_optimal_s(0, curr_soc, horizon)
    
    print(f"Result 1: best_s={best_s:.4f}, target={target:.2f}")
    print(f"New Internal State: last_opt_s={sup.last_opt_s}")
    
    if best_s > 2.4:
        print("SUCCESS: s increased to conserve battery.")
    elif best_s == 2.4:
        print("NEUTRAL: s stayed same.")
    else:
        print("FAILURE: s decreased (would discharge more).")

    # 3. Verify Memory Persistence
    # If we call again, candidates should center around the NEW best_s
    # Let's say best_s became 2.45 (2.4 + 0.05).
    # Next call candidates should be [2.35, 2.4, 2.45, 2.5, 2.55]
    
    print("\n--- Step 2 Call (Persistence Check) ---")
    # Make scenario dire: SOC dropped a lot!
    curr_soc = 0.48
    best_s_2, _ = sup.get_optimal_s(10, curr_soc, horizon)
    
    print(f"Result 2: best_s={best_s_2:.4f}")
    
    # Verify the search candidates logic implicitly by checking result is logical
    # But essentially if last_opt_s was updated, the search started from there.
    
    if abs(sup.last_opt_s - best_s_2) < 0.0001:
         print("State updated to new optimal.")
    
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    test_gravity_optimization()
