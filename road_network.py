import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('EPA133a-G04-A4/model/gdf_roads.csv')

# Create a directed graph
G = nx.DiGraph()

# Function to extract coordinates from POINT string
def extract_coords(point_str):
    try:
        if isinstance(point_str, float):
            return None, None
        # Remove 'POINT (' and ')' and split into lon, lat
        coords = point_str.replace('POINT (', '').replace(')', '').split()
        return float(coords[0]), float(coords[1])
    except:
        return None, None

# Add nodes and edges to the graph
for _, row in df.iterrows():
    try:
        # Extract start and end coordinates
        start_lon, start_lat = extract_coords(row['start_geometry'])
        end_lon, end_lat = extract_coords(row['end_geometry'])
        
        if start_lon is None or end_lon is None:
            continue
            
        # Add nodes if they don't exist
        # Using string representation of coordinates as node identifiers
        start_node = f"{start_lon},{start_lat}"
        end_node = f"{end_lon},{end_lat}"
        
        # Add nodes with separate lat/lon attributes
        G.add_node(start_node, longitude=start_lon, latitude=start_lat)
        G.add_node(end_node, longitude=end_lon, latitude=end_lat)
        
        # Add edge with attributes
        G.add_edge(start_node, end_node,
                   link_no=row['Link no'],
                   name=row['Name'],
                   length=row['Length-(Km)'],
                   total_traffic=row['Total-Total AADT'])
    except Exception as e:
        print(f"Error processing row {row['Link no']}: {e}")

# Create a position dictionary for drawing
pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes()}

# Plot the network
plt.figure(figsize=(12, 8))
nx.draw(G, pos, 
        node_size=20,
        node_color='red',
        with_labels=False,
        arrows=True,
        arrowsize=10)

plt.title('Road Network Graph')
plt.savefig('road_network.png')
plt.close()

# Print some basic network statistics
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Save the graph
nx.write_gexf(G, "road_network.gexf") 