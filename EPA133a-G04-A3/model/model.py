from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link, Intersection
import pandas as pd
from collections import defaultdict
import networkx as nx
import sys

print(sys.getrecursionlimit())
sys.setrecursionlimit(10000)

# ---------------------------------------------------------------
def set_lat_lon_bound(lat_min, lat_max, lon_min, lon_max, edge_ratio=0.02):
    """
    Set the HTML continuous space canvas bounding box (for visualization)
    give the min and max latitudes and Longitudes in Decimal Degrees (DD)

    Add white borders at edges (default 2%) of the bounding box
    """

    lat_edge = (lat_max - lat_min) * edge_ratio
    lon_edge = (lon_max - lon_min) * edge_ratio

    x_max = lon_max + lon_edge
    y_max = lat_min - lat_edge
    x_min = lon_min - lon_edge
    y_min = lat_max + lat_edge
    return y_min, y_max, x_min, x_max


# ---------------------------------------------------------------
class BangladeshModel(Model):
    """
    The main (top-level) simulation model

    One tick represents one minute; this can be changed
    but the distance calculation need to be adapted accordingly

    Class Attributes:
    -----------------
    step_time: int
        step_time = 1 # 1 step is 1 min

    path_ids_dict: defaultdict
        Key: (origin, destination)
        Value: the shortest path (Infra component IDs) from an origin to a destination

        Only straight paths in the Demo are added into the dict;
        when there is a more complex network layout, the paths need to be managed differently

    sources: list
        all sources in the network

    sinks: list
        all sinks in the network

    """

    step_time = 1

    file_name = '../data/bridges_intersected_linked.csv'

    def __init__(self,scenario_id, seed=None, x_max=500, y_max=500, x_min=0, y_min=0):
        self.scenario_id = scenario_id

        scenarios_probs = {
            0: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.00},
            1: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.05},
            2: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.10},
            3: {"A": 0.0, "B": 0.0, "C": 0.05, "D": 0.10},
            4: {"A": 0.0, "B": 0.0, "C": 0.10, "D": 0.20},
            5: {"A": 0.0, "B": 0.05, "C": 0.10, "D": 0.20},
            6: {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.40},
            7: {"A": 0.05, "B": 0.10, "C": 0.20, "D": 0.40},
            8: {"A": 0.10, "B": 0.20, "C": 0.40, "D": 0.80}
        }

        # Store breakdown probabilities for the selected scenario
        self.breakdown_probs = scenarios_probs.get(scenario_id, {})
        self.schedule = BaseScheduler(self)
        self.running = True
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.space = None
        self.sources = []
        self.sinks = []
        self.driving_times = []
        self.network_graph = nx.DiGraph()
        self.generate_model()

    def get_average_driving_time(self):
        """
        Calculate the average driving time for all trucks in the model.
        """
        if not self.driving_times:
            return 0
        return sum(self.driving_times) / len(self.driving_times)

    def generate_model(self):
        """
        generate the simulation model according to the csv file component information

        Warning: the labels are the same as the csv column labels
        """

        df = pd.read_csv(self.file_name)
        roads = df['road'].unique().tolist()
        # Filter nodes: keep only 'intersection' and 'sourcesink'
        for _, row in df.iterrows():
            node_id = row["id"]
            self.network_graph.add_node(node_id, pos=(row["lat"], row["lon"]), road=row["road"], model_type=row["model_type"])

        # ✅ Add edges (connect all objects on the same road)
        for road in df["road"].unique():
            road_nodes = df[df["road"] == road].sort_values(by="id")[["id", "length"]].values

            for i in range(len(road_nodes) - 1):
                start_node, start_length = road_nodes[i]
                end_node, end_length = road_nodes[i + 1]

                # ✅ Use the summed length of road elements as edge weights
                segment_length = start_length + end_length  # Sum lengths
                self.network_graph.add_edge(start_node, end_node, weight=segment_length)

        print(f"Graph created with {self.network_graph.number_of_nodes()} nodes and {self.network_graph.number_of_edges()} edges.")

        # a list of names of roads to be generated
        # TODO You can also read in the road column to generate this list automatically
        roads = ['N1', 'N2']

        df_objects_all = []
        for road in roads:
            # Select all the objects on a particular road in the original order as in the cvs
            df_objects_on_road = df[df['road'] == road]

            if not df_objects_on_road.empty:
                df_objects_all.append(df_objects_on_road)

                """
                Set the path 
                1. get the serie of object IDs on a given road in the cvs in the original order
                2. add the (straight) path to the path_ids_dict
                3. put the path in reversed order and reindex
                4. add the path to the path_ids_dict so that the vehicles can drive backwards too
                """
                path_ids = df_objects_on_road['id']
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                self.path_ids_dict[path_ids[0], None] = path_ids
                path_ids = path_ids[::-1]
                path_ids.reset_index(inplace=True, drop=True)
                self.path_ids_dict[path_ids[0], path_ids.iloc[-1]] = path_ids
                self.path_ids_dict[path_ids[0], None] = path_ids

        # put back to df with selected roads so that min and max and be easily calculated
        df = pd.concat(df_objects_all)
        y_min, y_max, x_min, x_max = set_lat_lon_bound(
            df['lat'].min(),
            df['lat'].max(),
            df['lon'].min(),
            df['lon'].max(),
            0.05
        )

        # ContinuousSpace from the Mesa package;
        # not to be confused with the SimpleContinuousModule visualization
        self.space = ContinuousSpace(x_max, y_max, True, x_min, y_min)

        for df in df_objects_all:
            for _, row in df.iterrows():  # index, row in ...

                # create agents according to model_type
                model_type = row['model_type'].strip()
                agent = None

                name = row['name']
                if pd.isna(name):
                    name = ""
                else:
                    name = name.strip()

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], name, row['road'])
                    self.sinks.append(agent.unique_id)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], name, row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], name, row['road'], row['condition'])
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], name, row['road'])
                elif model_type == 'intersection':
                    if not row['id'] in self.schedule._agents:
                        agent = Intersection(row['id'], self, row['length'], name, row['road'])

                if agent:
                    self.schedule.add(agent)
                    y = row['lat']
                    x = row['lon']
                    self.space.place_agent(agent, (x, y))
                    agent.pos = (x, y)

    def get_random_route(self, source):
        """
        pick up a random route given an origin
        """
        while True:
            # different source and sink
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        return sink

    def get_route(self, source_id):
        # 1. Pick a sink or a random destination
        sink = self.get_random_route(source_id)
        sink_id = next((sink for (src, sink) in self.path_ids_dict.keys() if src == source_id), None)
        # 2. Check if (source_id, sink_id) is in path_ids_dict
        if (source_id, sink_id) in self.path_ids_dict:
            return self.path_ids_dict[(source_id, sink_id)]

        # 3. If not, compute the shortest path using NetworkX
        shortest_path = nx.shortest_path(self.network_graph, source=source_id, target=sink_id, weight="weight")
        print(f"Shortest path from {source_id} to {sink_id}: {shortest_path}")
        # 4. Save the route in path_ids_dict
        self.path_ids_dict[(source_id, sink_id)] = shortest_path

        return pd.Series(shortest_path)

    def get_straight_route(self, source):
        """
        pick up a straight route given an origin
        """
        return self.path_ids_dict[source, None]

    def step(self):
        """
        Advance the simulation by one step.
        """
        self.schedule.step()

# EOF -----------------------------------------------------------
