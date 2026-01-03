import numpy as np
import pandas as pd

class PECMS_Supervisor:
    """
    Implements P-ECMS Supervisory with Physics-based Look-Ahead.
    Optimized for Real-Time execution using Local Neighborhood Search.
    
    Logic:
    1. Retrieve Speed Vector from Horizon (Spatial Nodes).
    2. Calculate Forces (Aero, Roll, Grade, Accel) -> P_wheel -> T_req internally.
    3. Simulate multiple s-candidates locally centered on previous optimum.
    """
    def __init__(self, vehicle, controller, q_max_as, target_soc=0.50, k_slope=1):
        self.veh = vehicle
        self.controller = controller # Instance of ECMS_Controller
        self.target_soc = target_soc
        self.q_max_as = q_max_as
        self.k_slope = k_slope
        self.soc_nominal = target_soc

        
        # --- Physics Parameters (Extracted from Vehicle) ---
        self.mass = getattr(self.veh, 'M_veh', 40000.0)
        self.cd = getattr(self.veh, 'Cd', 0.6)
        self.area = getattr(self.veh, 'A_front', 10.0)
        self.cr = getattr(self.veh, 'C_roll', 0.006)
        self.rho = getattr(self.veh, 'Rho_air', 1.2)
        # Assuming Trans efficiency is needed for Load -> Engine Torque
        self.eta_trans = getattr(self.veh, 'eta_trans', 0.96)
        
        # --- Hardcoded EF Values & Ratio (User Request) ---
        self.s_dis_0 = 2.3395
        self.s_chg_0 = 1.7538
        self.ratio = self.s_chg_0 / self.s_dis_0
        
        # --- Optimization State Memory ---
        # Initialize with the hardcoded base value
        self.last_opt_s = self.s_dis_0
        
        # Params (Bounds)
        self.delta_s = 0.05
        
        # "min 1.6 for charge and max 2.6 for discharge based on the ratio"
        # s_chg >= 1.6  =>  s_dis * ratio >= 1.6  =>  s_dis >= 1.6 / ratio
        # s_dis <= 2.6
        self.s_max = 2.6
        self.s_min = 1.4 / self.ratio

    def calculate_horizon_energy_delta(self, horizon_data):
        """
        Vypočítá čistou změnu SoC, kterou získáme/ztratíme překonáním
        topografie horizontu, čistě na základě potenciální energie.
        """
        # 1. Získání dat
        grades = horizon_data.get('grades', []) # v radiánech nebo tan (slope)
        dists = horizon_data.get('dists', [])   # vzdálenosti segmentů [m]
        vels = horizon_data.get('vel_kmh', []) / 3.6
        
        if len(grades) == 0:
            return 0.0

        # 2. Výpočet převýšení (Delta H)
        # dh = sin(arctan(grade)) * distance  ~= grade * distance (pro malé úhly)
        # Lepší je integrovat přes segmenty
        delta_h_total = 0.0
        # Pokud máš v horizon_data přímo nadmořskou výšku 'alts', použij: alts[-1] - alts[0]
        # Pokud ne, integrujeme sklon:
        for i in range(len(grades)):
            dist_step = vels[i] * horizon_data['dts'][i] # nebo dists[i] pokud je k dispozici
            # grade je tan(alpha), pro malé úhly je sin ~ tan
            dh = grades[i] * dist_step 
            delta_h_total += dh

        # 3. Fyzikální konstanty
        # Gravitační zrychlení
        g = 9.81
        # Hmotnost (kg)
        m = self.mass 
        
        # Potenciální energie [J]
        # Pokud delta_h < 0 (jedeme dolů), E_pot je záporná (energie se uvolňuje)
        e_pot_joules = m * g * delta_h_total
        
        # 4. Konverze na SoC
        # Musíme odhadnout účinnost. 
        # Když brzdíme (delta_h < 0), získáváme energii * účinnost.
        # Když táhneme (delta_h > 0), ztrácíme energii / účinnost.
        
        eta_recup = 0.80 # Odhad účinnosti rekuperace (zahrnuje aero odpor, pneu, motor)
        eta_trac = 0.90  # Odhad účinnosti trakce
        
        nominal_voltage = 650.0 # ! DŮLEŽITÉ: Změň podle tvého modelu (např. self.veh.v_nom)
        total_capacity_joules = self.q_max_as * nominal_voltage
        
        delta_soc = 0.0
        
        if delta_h_total < 0:
            # Sjezd -> Získáváme energii (Rekuperace)
            # e_pot je záporné, chceme kladné delta_soc
            energy_recovered = -e_pot_joules * eta_recup 
            # Ale pozor: Část energie sežere odpor vzduchu a valivý odpor!
            # Pro P-ECMS je bezpečnější brát jen "Potenciální" složku,
            # protože odpor vzduchu se počítá v cost function.
            # Zde chceme jen vědět "jak moc se změní terén".
            
            delta_soc = energy_recovered / total_capacity_joules
            
        else:
            # Výjezd -> Spotřebováváme energii
            energy_needed = e_pot_joules / eta_trac
            delta_soc = - (energy_needed / total_capacity_joules)

        return delta_soc

    def get_optimal_target_soc(self, current_soc, horizon_data):
        """
        Generuje Target SOC na základě "prostoru pro rekuperaci" v horizontu.
        Vrací hladkou křivku inverzní k profilu tratě.
        """
        # 1. Extrakce dat z horizontu
        alts = horizon_data.get('alts', []) # Nadmořské výšky v [m]
        

        if len(alts) == 0:
            return self.soc_nominal

        # 2. Fyzikální parametry
        current_alt = alts[0]
        min_alt_in_horizon = np.min(alts) # Nejnižší bod, který vidíme
        
        # 3. Kolik metrů budeme klesat? (Maximalizujeme rekuperaci)
        # Hledáme "max drop" - tedy rozdíl mezi námi a dnem údolí před námi.
        drop_height = current_alt - min_alt_in_horizon
        
        # Pokud je drop_height záporné (jen stoupáme), je to 0.
        drop_height = max(0.0, drop_height)

        # 4. Výpočet Delta SOC (Kolik % baterie dobije tento sjezd?)
        # E = m * g * h
        m_veh = self.mass
        g = 9.81
        e_potential_joules = m_veh * g * drop_height
        
        # Účinnost: Ne všechna potenciální energie skončí v baterii.
        # Odpor vzduchu, valivý odpor, účinnost měniče...
        # Pro Target Generator buď konzervativní (např. 0.6 - 0.7)
        eta_total_recup = 0.65 
        nominal_voltage = 681.29
        e_batt_cap_joules = self.q_max_as * nominal_voltage # Celková kapacita J
        
        delta_soc_needed = (e_potential_joules * eta_total_recup) / e_batt_cap_joules

        # 5. Stanovení Targetu
        # Logika: Chci skončit sjezd na 'high_soc_limit' (např. 65% nebo 60%).
        # Proto musím začít sjezd s (high_limit - delta).
        
        high_soc_limit = 0.60 # Kam až chceme nechat baterii dobít rekuperací
        
        raw_target = high_soc_limit - delta_soc_needed
        
        # 6. Saturace (Bezpečnostní limity)
        # Target nesmí být nižší než min (např. 30%) a vyšší než max.
        soc_target = max(0.30, min(0.65, raw_target))
        
        # 7. (VOLITELNÉ) Low-Pass Filter pro absolutní hladkost
        # Pokud to voláš v cyklu, můžeš si pamatovat 'self.last_target'
        # alpha = 0.1
        # soc_target = (1 - alpha) * self.last_target + alpha * soc_target
        # self.last_target = soc_target

        return soc_target
        
    def get_optimal_s(self, current_dist, current_soc, horizon_data):
        # 1. Target SOC Calculation 
        # Base target is constant (Charge Sustaining)
        
        dist_covered = horizon_data['dist_covered']

        calculate_horizon_energy_delta = self.calculate_horizon_energy_delta(horizon_data)
        
        # 1. Update Target SOC with Slope Adjustment
        soc_adj = self.k_slope * calculate_horizon_energy_delta
        soc_target = self.soc_nominal + soc_adj
    
        soc_target = max(0.35, min(0.75, soc_target))
        
        # 2. Internal Physics Calculation
        # Force Calculation using Speed Vector + Grade + Spatial Accel
        vels_kmh = horizon_data.get('vel_kmh', np.zeros_like(horizon_data['rpms']))
        vels = vels_kmh / 3.6 # m/s
        grades = horizon_data.get('grades', np.zeros_like(vels))
        
        dts = horizon_data.get('dts', np.ones_like(vels))
        
        # Gradient accel (Spatial/Forward Diff)
        accels = np.zeros_like(vels)
        dv = np.diff(vels) 
        dt_steps = dts[:-1]
        dt_steps[dt_steps < 0.01] = 0.01 
        accels[:-1] = dv / dt_steps
        accels[-1] = accels[-2] 
        
        # Forces
        f_aero = 0.5 * self.rho * self.cd * self.area * (vels**2)
        theta = np.arctan(grades)
        f_roll = self.cr * self.mass * 9.81 * np.cos(theta)
        f_grade = self.mass * 9.81 * np.sin(theta)
        f_acc = self.mass * accels
        
        f_total = f_aero + f_roll + f_grade + f_acc
        
        # Power at Wheel [kW]
        p_load_kw = (f_total * vels) / 1000.0
        
        # P_trans_in
        p_trans_in_arr = np.where(p_load_kw >= 0, p_load_kw / self.eta_trans, p_load_kw * self.eta_trans)
        
        # T_req from VECTO RPM (Baseline Strategy)
        rpms = horizon_data['rpms']
        omega = rpms * 2 * np.pi / 60.0
        omega[omega < 1.0] = 1.0 
        
        t_reqs_calc = (p_trans_in_arr * 1000.0) / omega
        
        # 3. Optimization (Local Search)
        steps = len(rpms)
        
        candidates = [
            self.last_opt_s - 2 * self.delta_s,
            self.last_opt_s - 1 * self.delta_s,
            self.last_opt_s,
            self.last_opt_s + 1 * self.delta_s,
            self.last_opt_s + 2 * self.delta_s,
        ]
        
        best_s = self.last_opt_s
        min_error = float('inf')
        
        orig_s_dis = self.controller.s_dis
        orig_s_chg = self.controller.s_chg
        
        for s in candidates:
            if s < self.s_min: s = self.s_min
            if s > self.s_max: s = self.s_max
            
            sim_soc = current_soc
            self.controller.s_dis = s
            self.controller.s_chg = s * self.ratio
            
            sim_valid = True
            for k in range(steps):
                try:
                    res = self.controller.decide_split(t_reqs_calc[k], rpms[k], sim_soc)
                    p_chem = res[3] 
                    
                    u_oc = self.veh.get_ocv(sim_soc)
                    i_bat = p_chem / u_oc
                    d_soc = - (i_bat * dts[k]) / self.q_max_as
                    sim_soc += d_soc
                    
                except:
                    sim_valid = False
                    break
            
            if sim_valid:
                error = abs(sim_soc - soc_target)
                if error < min_error:
                    min_error = error
                    best_s = s
        
        # Restore
        self.controller.s_dis = orig_s_dis
        self.controller.s_chg = orig_s_chg
        
        # Memory
        self.last_opt_s = best_s
        
        return best_s, soc_target, self.ratio
