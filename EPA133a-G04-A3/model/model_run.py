from model import BangladeshModel
import os
import pandas as pd
import seaborn as sns
import random

"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
run_length = 5 * 24 * 60

# run time 1000 ticks
# run_length = 1000

base_seed = 1234567

# Define experiment output directory (replace xx with your group number)
experiment_folder = "../experiment"
os.makedirs(experiment_folder, exist_ok=True)

# Run experiments for scenarios 0-9, with 10 replications each
data_columns = ["Scenario", "Replication", "Average_Driving_Time"]
all_results = []

for scenario in range(9):  # Scenarios 0-8
    scenario_results = []
    for replication in range(10):  # 10 replications per scenario
        seed = base_seed + replication  # Vary the seed for each replication
        sim_model = BangladeshModel(seed=seed, scenario_id=scenario)

        # Run simulation
        for i in range(run_length):
            sim_model.step()

        # Gather average driving time
        avg_driving_time = sim_model.get_average_driving_time()
        scenario_results.append([scenario, replication, avg_driving_time])

    # Save results for this scenario
    df = pd.DataFrame(scenario_results, columns=data_columns)
    output_file = os.path.join(experiment_folder, f"scenario{scenario}.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

print("All experiments completed!")