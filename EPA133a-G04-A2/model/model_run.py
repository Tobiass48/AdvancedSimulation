from model import BangladeshModel
import numpy as np
import pandas as pd
import os

"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 7200

scenario = {
    0: {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.0},
    1: {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.05},
    2: {'A': 0.0, 'B': 0.0, 'C': 0.0, 'D': 0.10},
    3: {'A': 0.0, 'B': 0.0, 'C': 0.05, 'D': 0.10},
    4: {'A': 0.0, 'B': 0.0, 'C': 0.10, 'D': 0.20},
    5: {'A': 0.0, 'B': 0.05, 'C': 0.10, 'D': 0.20},
    6: {'A': 0.0, 'B': 0.10, 'C': 0.20, 'D': 0.40},
    7: {'A': 0.05, 'B': 0.10, 'C': 0.20, 'D': 0.40},
    8: {'A': 0.10, 'B': 0.20, 'C': 0.40, 'D': 0.80}
}

scenario_choice = 8

seed = 1234567
#np.random.randint(100000, 999999)
output_folder = "../data/result"
os.makedirs(output_folder, exist_ok=True)

#  Initialize the simulation model
sim_model = BangladeshModel(seed=int(seed), breakdown_probabilities=scenario, scenario=scenario_choice)

#  Run simulation
for _ in range(run_length):
    sim_model.step()

#  Save data in the `../data/` folder
output_file = os.path.join(output_folder, f"scenario{scenario_choice}.csv")
sim_model.save_data(output_file)

print(f"Simulation completed for scenario {scenario_choice}. Data saved in {output_file}.")