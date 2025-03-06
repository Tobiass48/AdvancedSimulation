import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot, savefig
#import seaborn as sns

#%%

csv_files = ['..data/result/scenario0.csv', '..data/result/scenario1.csv', '..data/result/scenario2.csv',
             '..data/result/scenario3.csv','..data/result/scenario4.csv', '..data/result/scenario5.csv',
             '..data/result/scenario6.csv','..data/result/scenario7.csv', '..data/result/scenario8.csv']


data_list = []


for file in csv_files:
    df = pd.read_csv(file)


    data_list.append((df['Average_driving_time']/60).tolist())


plt.figure(figsize=(12, 6))
plt.boxplot(data_list, patch_artist=True)


plt.xticks(range(1, 10), [f"Scenario {i}" for i in range(9)], rotation=45)
plt.title("Boxplot of All Scenarios")
plt.ylabel("Average driving time (hours)")

savefig('../img/output.png')

# Show the plot
plt.show()


