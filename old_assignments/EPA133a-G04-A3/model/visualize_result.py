import matplotlib
matplotlib.use("TkAgg")  # Use a compatible backend

import pandas as pd
import matplotlib.pyplot as plt

# List of CSV files
csv_files = ['../experiment/scenario0.csv', '../experiment/scenario1.csv']

average_times = []

# Load data and calculate mean average driving time
for file in csv_files:
    df = pd.read_csv(file)
    mean_time = (df['Average_Driving_Time'] / 60).mean()  # Convert to hours and calculate mean
    average_times.append(mean_time)

# Plot bar graph
plt.figure(figsize=(12, 6))
plt.bar(range(2), average_times, color='skyblue')

# Labels and titles
plt.xticks(range(2), [f"Scenario {i}" for i in range(2)], rotation=45)
plt.ylabel("Average Driving Time (hours)")
plt.title("Bar Graph of Average Driving Time Across Scenarios")

# Save and show plot
plt.savefig('../img/output.png', dpi=300, bbox_inches='tight')
plt.show()

