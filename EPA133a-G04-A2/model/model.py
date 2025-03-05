from mesa import Model
from mesa.time import BaseScheduler
from mesa.space import ContinuousSpace
from components import Source, Sink, SourceSink, Bridge, Link
import pandas as pd
from collections import defaultdict
from mesa.datacollection import DataCollector


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
    The main simulation model with DataCollector for average travel time.
    """

    step_time = 1  # 1 tick = 1 minute

    def __init__(self, seed=None, scenario=0, csv_output='scenario0.csv'):
        super().__init__()
        self.schedule = BaseScheduler(self)
        self.running = True
       # self.scenario = scenario  # Store scenario index
        self.csv_output = csv_output
        self.path_ids_dict = defaultdict(lambda: pd.Series())
        self.sources = []
        self.sinks = []
        self.travel_time = []

        self.scenario = scenario

        # Define breakdown probabilities for different bridge categories (Cat A, B, C, D)
        scenarios_probs = {
            1: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.00},
            2: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.05},
            3: {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.10},
            4: {"A": 0.0, "B": 0.0, "C": 0.05, "D": 0.10},
            5: {"A": 0.0, "B": 0.0, "C": 0.10, "D": 0.20},
            6: {"A": 0.0, "B": 0.05, "C": 0.10, "D": 0.20},
            7: {"A": 0.0, "B": 0.10, "C": 0.20, "D": 0.40},
            8: {"A": 0.05, "B": 0.10, "C": 0.20, "D": 0.40},
            9: {"A": 0.10, "B": 0.20, "C": 0.40, "D": 0.80}
        }

        # Store breakdown probabilities for the selected scenario
        self.breakdown_probs = scenarios_probs.get(scenario, {})
        # DataCollector for tracking average travel time
        self.datacollector = DataCollector(
            model_reporters={"AverageTravelTime": self.calculate_average_travel_time}
        )

        self.generate_model()

    def generate_model(self):
        """Generate the simulation model from dataset."""
        df = pd.read_csv("../data/demo-1.csv")
        roads = ['N1']
        df_objects_all = []

        for road in roads:
            df_objects_on_road = df[df['road'] == road].sort_values(by=['id'])
            if not df_objects_on_road.empty:
                df_objects_all.append(df_objects_on_road)
                path_ids = df_objects_on_road['id']
                self.path_ids_dict[path_ids.iloc[0], path_ids.iloc[-1]] = path_ids
                path_ids = path_ids[::-1].reset_index(drop=True)
                self.path_ids_dict[path_ids.iloc[0], path_ids.iloc[-1]] = path_ids

        df = pd.concat(df_objects_all)

        for df in df_objects_all:
            for _, row in df.iterrows():
                model_type = row['model_type']
                agent = None

                if model_type == 'source':
                    agent = Source(row['id'], self, row['length'], row['name'], row['road'])
                    self.sources.append(agent.unique_id)
                elif model_type == 'sink':
                    agent = Sink(row['id'], self, row['length'], row['name'], row['road'])
                    self.sinks.append(agent.unique_id)
                elif model_type == 'sourcesink':
                    agent = SourceSink(row['id'], self, row['length'], row['name'], row['road'])
                    self.sources.append(agent.unique_id)
                    self.sinks.append(agent.unique_id)
                elif model_type == 'bridge':
                    agent = Bridge(row['id'], self, row['length'], row['name'], row['road'], row['condition'])
                elif model_type == 'link':
                    agent = Link(row['id'], self, row['length'], row['name'], row['road'])

                if agent:
                    self.schedule.add(agent)

    def get_random_route(self, source):
        """Get a random route given a source."""
        while True:
            sink = self.random.choice(self.sinks)
            if sink is not source:
                break
        return self.path_ids_dict[source, sink]

    def calculate_average_travel_time(self):
        """Compute the average travel time including delays."""
        if len(self.travel_time) == 0:
            return 0
        return sum(self.travel_time) / len(self.travel_time)

    def step(self):
        """Advance the simulation and collect travel time data."""
        self.schedule.step()
        self.datacollector.collect(self)  # Collect average travel time

# EOF -----------------------------------------------------------