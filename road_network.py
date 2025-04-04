import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('EPA133a-G04-A4/model/gdf_roads.csv')

traffic_columns = ["Traffic Data-Heavy Truck", "Traffic Data-Small Truck", "Traffic Data-Medium Truck",
                   "Total-Total AADT"]
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
                   total_traffic=total_traffic)
    except Exception as e:
        print(f"Error processing row {row['Link no']}: {e}")


def calculate_criticality_metrics():
    # 1. Calculate betweenness centrality for edges
    edge_betweenness = nx.edge_betweenness_centrality(G, weight="weight", normalized=True)

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
            alt_path_length = sum(G[path[i]][path[i + 1]]['length'] for i in range(len(path) - 1))
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
                0.4 * edge_betweenness[edge] +  # Network centrality
                0.2 * (1 - redundancy[edge])  # Lack of alternatives (inverse of redundancy)
        )

        # Store the criticality score as an edge attribute
        G.edges[edge]['criticality'] = criticality_scores[edge]

    return criticality_scores, edge_betweenness, redundancy, max_traffic


# Calculate criticality metrics
criticality_scores, edge_betweenness, redundancy, max_traffic = calculate_criticality_metrics()

# Create a DataFrame to store edge data and criticality metrics
criticality_data = []

for edge in G.edges():
    edge_data = G.edges[edge]
    criticality_data.append({
        "start_node": edge[0],
        "end_node": edge[1],
        "link_no": edge_data.get("link_no", None),
        "name": edge_data.get("name", None),
        "length": edge_data.get("length", None),
        "total_traffic": edge_data.get("total_traffic", None),
        "traffic_score": edge_data.get("total_traffic", 0) / max_traffic,  # Normalized traffic
        "betweenness_score": edge_betweenness.get(edge, 0),  # Edge betweenness centrality
        "redundancy_score": redundancy.get(edge, 0),  # Alternative path ratio
        "criticality_score": edge_data.get("criticality", 0)  # Combined criticality score
    })

# Convert to a pandas DataFrame
criticality_df = pd.DataFrame(criticality_data)

# Add a ranking column based on the criticality score
criticality_df["criticality_rank"] = criticality_df["criticality_score"].rank(ascending=False)

# Sort rows by criticality rank
criticality_df = criticality_df.sort_values(by="criticality_rank").reset_index(drop=True)

# Save DataFrame to CSV
criticality_df.to_csv("criticality_ranked_roads.csv", index=False)

# Display top 10 rows
print("\nTop 10 Most Critical Road Segments:")
print(criticality_df.head(10))
