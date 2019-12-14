import numpy as np
import pandas as pd

write_path = "set_data/"

class MetricsRecorder:
    def __init__(self, set_number):
        self.fitness_stats = MetricStats("Fitness")
        self.collision_stats = MetricStats("Collision")
        self.infraction_stats = MetricStats("Infraction")
        self.liveliness_stats = MetricStats("Liveliness")
        self.jerk_stats = MetricStats("Jerk")
        self.traj_stats = MetricStats("Trajectory_Following")
        self.set_number = set_number

    def record(self, fitnesses, metrics):
        self.fitness_stats.record(fitnesses)
        self.collision_stats.record([m["collisions"] for m in metrics])
        self.infraction_stats.record([m["infractions"] for m in metrics])
        self.liveliness_stats.record([m["livelieness"] for m in metrics])
        self.jerk_stats.record([m["jerk"] for m in metrics])
        self.traj_stats.record([m["traj_following"] for m in metrics])

    def write_out(self):
        fitness_df = self.fitness_stats.get_df()
        collisions_df = self.collision_stats.get_df()
        infractions_df = self.infraction_stats.get_df()
        liveliness_df = self.liveliness_stats.get_df()
        jerk_df = self.jerk_stats.get_df()
        traj_df = self.traj_stats.get_df()
        df = pd.concat([
            fitness_df, collisions_df, infractions_df, liveliness_df, jerk_df, traj_df
        ], ignore_index=True)
        df["set"] = self.set_number
        file_name = write_path + "set" + str(self.set_number) + ".csv"
        df.to_csv(file_name)


class MetricStats:
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.metric_mins = []
        self.metric_maxs = []
        self.metric_avgs = []

    def record(self, values):
        self.metric_mins.append(np.min(values))
        self.metric_maxs.append(np.max(values))
        self.metric_avgs.append(np.average(values))

    def get_df(self):
        num_gens = len(self.metric_avgs)
        min_df = pd.DataFrame()
        min_df["generation"] = list(range(num_gens))
        min_df["type"] = "min"
        min_df["value"] = self.metric_mins

        max_df = pd.DataFrame()
        max_df["generation"] = list(range(num_gens))
        max_df["type"] = "max"
        max_df["value"] = self.metric_maxs

        avg_df = pd.DataFrame()
        avg_df["generation"] = list(range(num_gens))
        avg_df["type"] = "avg"
        avg_df["value"] = self.metric_avgs

        df = pd.concat([min_df, max_df, avg_df])
        df["metric"] = self.metric_name
        return df

