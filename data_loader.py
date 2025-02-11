import pandas as pd
import os

# Define the file path
file_path = os.path.join(os.path.dirname(__file__),'BMMS','BMMS_overview.xls')
# file_path_1 = "/BMMS/BMMS_overview_SortedRoad.xls"
# Read the Excel file into a DataFrame
df1 = pd.read_excel(file_path)
# df2 = pd.read_excel(file_path_1)

# Display the first few rows
print(df1.head())


