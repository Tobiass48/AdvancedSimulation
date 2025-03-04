from model import BangladeshModel

"""
    Run simulation
    Print output at terminal
"""

# ---------------------------------------------------------------

# run time 5 x 24 hours; 1 tick 1 minute
# run_length = 5 * 24 * 60

# run time 1000 ticks
run_length = 1000

seed = 1234567


#for scenario in range(1, 11):  # Run scenarios 1-5
#    for replication in range(10):  # 10 replications per scenario
#        sim_model = BangladeshModel(seed=seed, scenario_id=scenario)

sim_model = BangladeshModel(seed = seed, scenario_id=9)
sim_model.run()

# Check if the seed is set
print("SEED " + str(sim_model._seed))

# One run with given steps
for i in range(run_length):
    sim_model.step()
