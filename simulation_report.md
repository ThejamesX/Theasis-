# Batch Simulation Report

## 1. Long Haul Cycle (100.2 km)
**Baseline ICE Fuel:** 36.56 kg (36.49 kg/100km | 364.9 g/km)

| Capacity | Strategy | Fuel (kg) | Cons. (kg/100km) | Avg (g/km) | vs ICE (%) | vs DP (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **120 kWh** | **DP (Optimal)** | 31.19 | 31.13 | 311.3 | -14.7% | 0.0% |
| | ECMS | 34.49 | 34.42 | 344.2 | -5.7% | +10.6% |
| | P-ECMS | 34.98 | 34.92 | 349.2 | -4.3% | +12.1% |
| | A-ECMS | 35.14 | 35.08 | 350.8 | -3.9% | +12.7% |
| | | | | | | |
| **60 kWh** | **DP (Optimal)** | 31.39 | 31.33 | 313.3 | -14.1% | 0.0% |
| | ECMS | 34.50 | 34.44 | 344.4 | -5.6% | +9.9% |
| | P-ECMS | 34.98 | 34.92 | 349.2 | -4.3% | +11.4% |
| | A-ECMS | 35.19 | 35.13 | 351.3 | -3.7% | +12.1% |


## 2. Regional Delivery Cycle (100.0 km)
**Baseline ICE Fuel:** 36.83 kg (36.84 kg/100km | 368.4 g/km)

| Capacity | Strategy | Fuel (kg) | Cons. (kg/100km) | Avg (g/km) | vs ICE (%) | vs DP (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **120 kWh** | **DP (Optimal)** | 31.79 | 31.79 | 317.9 | -13.7% | 0.0% |
| | P-ECMS | 35.23 | 35.23 | 352.3 | -4.3% | +10.8% |
| | A-ECMS | 35.65 | 35.65 | 356.5 | -3.2% | +12.1% |
| | ECMS | 36.45 | 36.45 | 364.5 | -1.0% | +14.7% |
| | | | | | | |
| **60 kWh** | **DP (Optimal)** | 31.97 | 31.97 | 319.7 | -13.2% | 0.0% |
| | P-ECMS | 35.23 | 35.23 | 352.3 | -4.3% | +10.2% |
| | A-ECMS | 35.55 | 35.55 | 355.5 | -3.5% | +11.2% |
| | ECMS | 36.46 | 36.46 | 364.6 | -1.0% | +14.0% |

## 3. 30 kWh Analysis (User Requested)
**Grid Size: 400 | DP Target: 0.51**

### Long Haul Cycle (100.2 km)
**Baseline ICE Fuel:** 36.56 kg (36.49 kg/100km | 364.9 g/km)

| Capacity | Strategy | Fuel (kg) | Cons. (kg/100km) | Avg (g/km) | vs ICE (%) | vs DP (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **30 kWh** | **DP (Optimal)** | 31.59 | 31.53 | 315.3 | -13.6% | 0.0% |
| | ECMS | 34.50 | 34.44 | 344.4 | -5.6% | +9.2% |
| | P-ECMS | 34.98 | 34.92 | 349.2 | -4.3% | +10.7% |
| | A-ECMS | 35.27 | 35.21 | 352.1 | -3.5% | +11.6% |


### Regional Delivery Cycle (100.0 km)
**Baseline ICE Fuel:** 36.83 kg (36.84 kg/100km | 368.4 g/km)

| Capacity | Strategy | Fuel (kg) | Cons. (kg/100km) | Avg (g/km) | vs ICE (%) | vs DP (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **30 kWh** | **DP (Optimal)** | 32.18 | 32.18 | 321.8 | -12.6% | 0.0% |
| | P-ECMS | 35.23 | 35.23 | 352.3 | -4.3% | +9.5% |
| | ECMS | 35.38 | 35.38 | 353.8 | -3.9% | +9.9% |
| | A-ECMS | 35.53 | 35.53 | 355.3 | -3.5% | +10.4% |

## 4. P-ECMS vs A-ECMS Analysis
Sorted by efficiency (Avg Fuel per km).

### Long Haul Cycle (100.2 km)
| Rank | Strategy | Capacity | Avg Fuel (g/km) | vs A-ECMS Baseline |
|:---:|:---|:---:|:---:|:---:|
| 1 | **P-ECMS** | 120 kWh | **349.1** | -0.46% |
| 2 | **P-ECMS** | 60 kWh | **349.1** | -0.46% |
| 3 | **P-ECMS** | 30 kWh | **349.1** | -0.46% |
| 4 | A-ECMS | 120 kWh | 350.7 | - |
| 5 | A-ECMS | 60 kWh | 351.2 | +0.14% |
| 6 | A-ECMS | 30 kWh | 352.0 | +0.37% |

*Observation: P-ECMS maintains consistent optimal efficiency regardless of battery size, whereas A-ECMS efficiency degrades slightly as battery capacity decreases.*

### Regional Delivery Cycle (100.0 km)
| Rank | Strategy | Capacity | Avg Fuel (g/km) | vs A-ECMS Baseline |
|:---:|:---|:---:|:---:|:---:|
| 1 | **P-ECMS** | 120 kWh | **352.3** | -1.18% |
| 2 | **P-ECMS** | 60 kWh | **352.3** | -1.18% |
| 3 | **P-ECMS** | 30 kWh | **352.3** | -1.18% |
| 4 | A-ECMS | 60 kWh | 355.5 | -0.28% |
| 5 | A-ECMS | 30 kWh | 355.3 | -0.34% |
| 6 | A-ECMS | 120 kWh | 356.5 | - |

*Observation: P-ECMS consistently outperforms A-ECMS by ~1.2% in the Regional Cycle.*
