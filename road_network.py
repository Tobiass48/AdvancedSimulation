import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
        
        # Convert traffic to float and handle missing values
        try:
            total_traffic = float(row['Total-Total AADT'])
        except (ValueError, TypeError):
            total_traffic = 0.0
        
        # Add edge with attributes
        G.add_edge(start_node, end_node,
                   link_no=row['Link no'],
                   name=row['Name'],
                   length=float(row['Length-(Km)']),
                   total_traffic=total_traffic)
    except Exception as e:
        print(f"Error processing row {row['Link no']}: {e}")

def calculate_criticality_metrics():
    # 1. Calculate betweenness centrality for edges
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    
    # 2. Calculate alternative paths (redundancy)
    redundancy = {}
    for edge in list(G.edges()):  # Create a list to avoid modification during iteration
        # Store edge attributes
        edge_attrs = G.edges[edge].copy()
        
        # Remove the edge temporarily
        G.remove_edge(*edge)
        
        # Check if there's still a path between the nodes
        try:
            path = nx.shortest_path(G, edge[0], edge[1])
            # If there's an alternative path, calculate its length
            alt_path_length = sum(G[path[i]][path[i+1]]['length'] for i in range(len(path)-1))
            original_length = edge_attrs['length']
            redundancy[edge] = original_length / alt_path_length  # Ratio of original to alternative path length
        except nx.NetworkXNoPath:
            # No alternative path exists
            redundancy[edge] = 0
        except Exception as e:
            print(f"Error calculating redundancy for edge {edge}: {e}")
            redundancy[edge] = 0
        
        # Restore the edge with its original attributes
        G.add_edge(*edge, **edge_attrs)
    
    # 3. Normalize traffic values
    traffic_values = [float(G.edges[edge]['total_traffic']) for edge in G.edges()]
    max_traffic = max(traffic_values) if traffic_values else 1.0
    normalized_traffic = {edge: float(G.edges[edge]['total_traffic']) / max_traffic for edge in G.edges()}
    
    # 4. Calculate combined criticality score
    criticality_scores = {}
    for edge in G.edges():
        # Weighted combination of metrics
        criticality_scores[edge] = (
            0.4 * normalized_traffic[edge] +  # Traffic importance
            0.4 * edge_betweenness[edge] +   # Network centrality
            0.2 * (1 - redundancy[edge])     # Lack of alternatives (inverse of redundancy)
        )
        
        # Store the criticality score as an edge attribute
        G.edges[edge]['criticality'] = criticality_scores[edge]
    
    return criticality_scores

# Calculate criticality metrics
criticality_scores = calculate_criticality_metrics()

# Get top 10 most critical segments
top_10_critical = sorted(criticality_scores.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 Most Critical Road Segments:")
print("===================================")
for edge, score in top_10_critical:
    link_no = G.edges[edge]['link_no']
    name = G.edges[edge]['name']
    traffic = G.edges[edge]['total_traffic']
    print(f"Link: {link_no}")
    print(f"Name: {name}")
    print(f"Traffic: {traffic:,.0f}")
    print(f"Criticality Score: {score:.4f}")
    print("-----------------------------------")

# Create a position dictionary for drawing
pos = {node: (G.nodes[node]['longitude'], G.nodes[node]['latitude']) for node in G.nodes()}

# Plot the network with criticality visualization
plt.figure(figsize=(15, 10))

# Create subplot with space for colorbar
ax = plt.gca()
edges = G.edges()
edge_colors = [G.edges[edge]['criticality'] for edge in edges]

# Draw the network
nx.draw(G, pos,
        node_size=20,
        node_color='gray',
        edge_color=edge_colors,
        edge_cmap=plt.cm.YlOrRd,
        with_labels=False,
        arrows=True,
        arrowsize=10,
        width=2,
        ax=ax)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=min(edge_colors), vmax=max(edge_colors)))
sm.set_array([])
plt.colorbar(sm, ax=ax, label='Criticality Score')

plt.title('Road Network Criticality Analysis')
plt.savefig('road_network_criticality.png', dpi=300, bbox_inches='tight')
plt.close()

# Save the graph with criticality scores
nx.write_gexf(G, "road_network_with_criticality.gexf")

# Print some basic network statistics
print(f"\nNetwork Statistics:")
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Print criticality statistics
criticality_values = list(criticality_scores.values())
print(f"\nCriticality Statistics:")
print(f"Mean criticality: {np.mean(criticality_values):.4f}")
print(f"Median criticality: {np.median(criticality_values):.4f}")
print(f"Standard deviation: {np.std(criticality_values):.4f}")
print(f"Min criticality: {min(criticality_values):.4f}")
print(f"Max criticality: {max(criticality_values):.4f}") 