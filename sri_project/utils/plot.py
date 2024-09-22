import os
from matplotlib import pyplot as plt


import matplotlib.pyplot as plt
import numpy as np


def plot_and_save_graph(
    x: list,
    y1: list,
    y2: list,
    y3: list,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    labels: tuple = ("BM25", "DPR", "Reranking"),
    path: str = "img",
):
    """
    Plots two lines on a graph and saves it as an image file.

    Args:
        x (list): The x-axis values.
        y1 (list): The y-axis values for the first line.
        y2 (list): The y-axis values for the second line.
        y3 (list): The y-axis values for the third line.
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
    plt.plot(x, y3, label=f"{labels[2]} {ylabel}", color="green")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(f"{path}/{filename}.png")
    plt.show()


def plot_comparison(x1, x2, label1: str, label2: str, title: str, filename: str):
    """
    Plot a comparison bar chart between two sets of values.

    Args:
        x1 (list): List of values for the first set.
        x2 (list): List of values for the second set.
        label1 (str): Label for the first set.
        label2 (str): Label for the second set.
        title (str): Title of the chart.
        filename (str): Filename to save the chart.

    Returns:
        str: The filename of the saved chart.

    """
    labels = ["BM25", "DPR", "Reranking"]
    x1_vals = [x1[0][0], x1[1][0], x1[2][0]]
    x2_vals = []
    if x2 is not None:
        x2_vals = [x2[0][0], x2[1][0], x2[2][0]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, x1_vals, width, label=label1)
    if x2 is not None:
        ax.bar(x + width / 2, x2_vals, width, label=label2)

    ax.set_ylabel("Valores")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return filename
