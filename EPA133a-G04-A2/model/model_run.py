from model import BangladeshModel

"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------
# Run time: 5 days x 24 hours x 60 minutes = 5 * 24 * 60
# Adjusted to match the assignment requirement
run_length = 5 * 24 * 60  # Full 5-day simulation

seed = 1234567  # Set the seed for reproducibility

# Initialize the simulation model
sim_model = BangladeshModel(seed=seed, scenario=0, csv_output="scenario0.csv")

# Check if the seed is set correctly
print("SEED " + str(sim_model._seed))

# Run the simulation for the specified number of steps
for i in range(run_length):
    sim_model.step()

# Collect the recorded data from DataCollector
travel_data = sim_model.datacollector.get_model_vars_dataframe()

# Save the output to CSV as required
travel_data.to_csv(sim_model.csv_output)

# Print final output summary
print("Simulation complete. Results saved to:", sim_model.csv_output)
print(travel_data.describe())  # Print summary statistics