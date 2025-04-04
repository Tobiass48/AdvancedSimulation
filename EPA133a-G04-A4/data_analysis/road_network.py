import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('../data/gdf_roads.csv')

traffic_columns = ["Traffic Data-Heavy Truck", "Traffic Data-Small Truck", "Traffic Data-Medium Truck", "Total-Total AADT"]
for column in traffic_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

df["weight"] = (df["Traffic Data-Heavy Truck"] + df["Traffic Data-Small Truck"] + df["Traffic Data-Medium Truck"]) / df[
    "Total-Total AADT"]

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
                   total_traffic=total_traffic,
                   weight_1 = row["weight"])
    except Exception as e:
        print(f"Error processing row {row['Link no']}: {e}")

def calculate_criticality_metrics():
    # 1. Calculate betweenness centrality for edges
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight', normalized=False)

    # 2. Normalize traffic values
    traffic_values = [G.edges[edge]['total_traffic'] for edge in G.edges()]
    max_traffic = max(traffic_values) if traffic_values else 1.0
    normalized_traffic = {edge: G.edges[edge]['total_traffic'] / max_traffic for edge in G.edges()}

    # 2.5 Economic importance

    economic_values = [G.edges[edge]['weight_1'] for edge in G.edges()]
    max_economic = max(economic_values) if economic_values else 1.0
    normalized_economic = {edge: G.edges[edge]['weight_1'] / max_economic for edge in G.edges()}

    # 3. Normalize betweenness values
    betweenness_values = list(edge_betweenness.values())
    min_betweenness, max_betweenness = min(betweenness_values), max(betweenness_values)
    range_betweenness = max_betweenness - min_betweenness if max_betweenness != min_betweenness else 1.0

    scaled_betweenness = {edge: (value - min_betweenness) / range_betweenness for edge, value in edge_betweenness.items()}

    # 4. Compute criticality score
    criticality_scores = {}
    for edge in G.edges():
        # weight_1 = G.edges[edge].get('weight_1', 0)  # Ensure weight_1 exists
        criticality_scores[edge] = (
            0.2 * normalized_traffic.get(edge, 0) +  # Traffic importance
            0.4 * scaled_betweenness.get(edge, 0) +  # Betweenness
            0.4 * normalized_economic.get(edge, 0)   # Direct weight_1 contribution
        )

        # Store criticality score as an edge attribute
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
plt.show()
# plt.savefig('road_network_criticality.png', dpi=300, bbox_inches='tight')
# plt.close()

# # Save the graph with criticality scores
# nx.write_gexf(G, "road_network_with_criticality.gexf")

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

# Prepare criticality data for export
criticality_data = []

# Normalize edge betweenness and redundancy for output
edge_betweenness = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)
betweenness_values = list(edge_betweenness.values())

min_betweenness = min(betweenness_values)
max_betweenness = max(betweenness_values)
range_betweenness = max_betweenness - min_betweenness if max_betweenness != min_betweenness else 1.0  # Prevent division by zero

# Min-max scaling to [0, 1]
scaled_betweenness = {
    edge: (value - min_betweenness) / range_betweenness
    for edge, value in edge_betweenness.items()
}

traffic_values = [float(G.edges[edge]['total_traffic']) for edge in G.edges()]
max_traffic = max(traffic_values) if traffic_values else 1.0
normalized_traffic = {edge: float(G.edges[edge]['total_traffic']) / max_traffic for edge in G.edges()}


for edge in G.edges():
    edge_data = G.edges[edge]
    criticality = edge_data.get("criticality", 0)
    traffic_component = normalized_traffic.get(edge, 0)
    betweenness_component = scaled_betweenness.get(edge, 0)
    weight_1 = edge_data.get("weight_1", np.nan)  # Ensure weight_1 is included
    criticality_data.append({
        "start_node": edge[0],
        "end_node": edge[1],
        "link_no": edge_data.get("link_no", None),
        "name": edge_data.get("name", None),
        "length": edge_data.get("length", None),
        "total_traffic": edge_data.get("total_traffic", None),
        "traffic_score": traffic_component,
        "betweenness_score": betweenness_component,  # Edge betweenness centrality
        "economic_score": weight_1,  # Alternative path ratio
        "criticality_score": edge_data.get("criticality", 0)  # Combined criticality score
    })

# Convert to a pandas DataFrame
criticality_df = pd.DataFrame(criticality_data)

# Add a ranking column based on the criticality score
criticality_df["criticality_rank"] = criticality_df["criticality_score"].rank(ascending=False)

# Sort rows by criticality rank
criticality_df = criticality_df.sort_values(by="criticality_rank").reset_index(drop=True)

# Save DataFrame to CSV
criticality_df.to_csv("../experiment/criticality_ranked_roads.csv", index=False)
