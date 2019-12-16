import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

output_path = "graphs_output"
data_path = "set_data"

# Baseline with global knowledge
baseline = {
    "Fitness": 32.45,
    "Infraction": 0.0,
    "Collision": 0.0,
    "Jerk": 34.13,
    "Liveliness": 689.54,
    "Trajectory_Following": 3230.32
}

# Baseline with local knowledge
baseline2 = {
    "Fitness": 26.87,
    "Infraction": 0.0,
    "Collision": 99.0,
    "Jerk": 29.21,
    "Liveliness": 690.66,
    "Trajectory_Following": 3221.74
}
num_files = 2


def main():
    print("Loading data")
    df = load_dataframe()
    metric_names = pd.unique(df["metric"])
    for metric_name in metric_names:
        print("Creating graph for {}".format(metric_name))
        filter = df["metric"] == metric_name
        graph_metric(metric_name, df[filter])


def load_dataframe():
    dataframes = []
    for file_index in range(num_files):
        file_name = "set{}.csv".format(file_index)
        with open(os.path.join(data_path, file_name), 'rb') as f:
            dataframes.append(pd.read_csv(f))
    return pd.concat(dataframes)


def graph_metric(metric_name, df):
    if metric_name == "Fitness":
        df.value *= 6.0
    df = df.rename(columns={"type": ""})
    ax = sns.lineplot(x="generation", y="value", hue="", data=df)
    ax.set_title(metric_name)

    ax.axhline(y=baseline[metric_name], ls=':', label="Baseline", color="Magenta")
    ax.axhline(y=baseline2[metric_name], ls='--', label="Baseline2")
    ax.legend(title="")

    ax.plot()
    file_name = metric_name + ".pdf"
    save_location = os.path.join(output_path, file_name)
    plt.savefig(save_location)
    plt.clf()


if __name__ == '__main__':
    main()