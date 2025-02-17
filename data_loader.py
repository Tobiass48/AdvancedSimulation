import nbformat

def extract_dataframe_from_notebook(notebook_path, var_name):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook_content = nbformat.read(f, as_version=4)

    # Look for the cell that contains the DataFrame assignment
    for cell in notebook_content.cells:
        if cell.cell_type == 'code' and var_name in cell.source:
            try:
                # Execute the code to get the DataFrame in the notebook's context
                exec(cell.source, globals())
                if var_name in globals():
                    return globals().get(var_name)
                else:
                    print(f"Variable {var_name} was not created in the notebook")
            except Exception as e:
                print(f"Error executing cell with {var_name}: {e}")
    return None

# Example of loading two DataFrames
df1 = extract_dataframe_from_notebook('A1_data_cleaning_bridges.ipynb', 'gdf_bridges1')
df2 = extract_dataframe_from_notebook('A1_data_cleaning_roads.ipynb', 'df_rd_transformed')

if df1 is not None and df2 is not None:
    # Merge df1 and df2 based on the 'Road_ID' column
    merged_df = pd.merge(df1, df2, on='Road', how='inner')  # Adjust 'how' based on the type of join

    # Show the result
    print(merged_df.head())
else:
    print("One or both DataFrames could not be loaded.")

