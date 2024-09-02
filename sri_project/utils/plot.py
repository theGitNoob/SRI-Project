import os
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt


def plot_and_save_graph(
    x: list,
    y1: list,
    y2: list,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    labels: tuple = ("BM25", "DPR"),
    path: str = "img",
):
    """
    Plots two lines on a graph and saves it as an image file.

    Args:
        x (list): The x-axis values.
        y1 (list): The y-axis values for the first line.
        y2 (list): The y-axis values for the second line.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the graph.
        filename (str): The name of the file to save the graph as.
        labels (tuple, optional): The labels for the two lines. Defaults to ("BM25", "DPR").
        path (str, optional): The path to save the image file. Defaults to "img".

    Returns:
        None
    """

    if not os.path.exists(path):
        os.makedirs(path)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label=f"{labels[0]} {ylabel}", color="blue")
    plt.plot(x, y2, label=f"{labels[1]} {ylabel}", color="orange")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{path}/{filename}.png")
    plt.show()
