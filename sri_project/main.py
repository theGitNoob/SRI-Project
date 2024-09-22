import time
from typing import Literal

from matplotlib import pyplot as plt
import numpy as np
import gradio as gr

from sri_project.utils.metrics import evaluate_performance
from sri_project.utils.plot import plot_and_save_graph, plot_comparison
from sri_project.utils.dataset_loader import corpus, queries, grouped_data
from sri_project.utils.utils import initialize_indexes


def run_whole_evaluation(bm25, dpr_index):
    """
    Runs the whole evaluation process for the given BM25 and DPR index.

    Args:
        bm25: The BM25 model used for evaluation.
        dpr_index: The DPR index used for evaluation.

    Returns:
        None
    """
    # Performance evaluation
    (
        times,
        recall_values,
        precision_values,
        _,
    ) = evaluate_performance(grouped_data, queries, bm25, dpr_index)

    plot(times, recall_values, precision_values)


def main():
    """
    Main function that initializes indexes, performs performance evaluation,
    sets up the Gradio interface, and launches it.
    """

    # Indexes initialization
    global bm25, dpr_index
    bm25, dpr_index = initialize_indexes(corpus[:50])

    # Performance evaluation
    run_whole_evaluation(bm25, dpr_index)

    # Initialize the Gradio interface
    interface = setup_interface()

    # Launch the interface
    interface.launch()


def plot(times, recall_values, precision_values):
    """
    Plots and saves the results of the comparison between BM25, DPR, and Reranking.

    Args:
        times (list): A list of time values for each query.
        recall_values (list): A list of recall values for each query.
        precision_values (list): A list of precision values for each query.

    Returns:
        list: A list of charts representing the comparison between BM25, DPR, and Reranking.
    """

    # Plotting and saving the results
    charts = [
        plot_and_save_graph(
            x=times[0],
            y1=times[0],
            y2=times[1],
            y3=times[2],
            xlabel="Consultas",
            ylabel="Tiempo (s)",
            title="Comparación de Tiempo de Cómputo entre BM25, DPR y Reranking",
            filename="tiempo_computo_comparacion",
        ),
        plot_and_save_graph(
            x=recall_values[0],
            y1=recall_values[0],
            y2=recall_values[1],
            y3=recall_values[2],
            xlabel="Consultas",
            ylabel="Recall",
            title="Comparación de Recall entre BM25, DPR y Reranking",
            filename="recall_comparacion",
        ),
        plot_and_save_graph(
            x=precision_values[0],
            y1=precision_values[0],
            y2=precision_values[1],
            y3=precision_values[2],
            xlabel="Consultas",
            ylabel="Precision",
            title="Comparación de Precision entre BM25, DPR y Reranking",
            filename="precision_comparacion",
        ),
    ]

    return charts


def search(query: str, model: Literal["BM25", "DPR", "Reranking"]):
    """
    Perform a search query using the specified model.

    Args:
        query (str): The search query.
        model (Literal["BM25", "DPR", "Reranking"]): The model to use for the search.

    Returns:
        Tuple: A tuple containing the following information:
            - Retrieved documents for the selected model.
            - Precision value for the selected model.
            - Recall value for the selected model.
            - Time taken for the selected model.
            - Comparison chart of precision and recall for BM25 vs DPR, Reranking.
            - Comparison chart of computation time and memory usage for BM25, DPR, and Reranking.
    """
    quer_idx = queries.index(query)

    times, recall_values, precision_values, retrieved_docs = evaluate_performance(
        grouped_data, queries, bm25, dpr_index, quer_idx
    )

    charts = [
        plot_comparison(
            precision_values,
            recall_values,
            "Recall",
            "Precision",
            "Comparación: BM25 vs DPR, Reranking",
            "img/precision_recall_comparison.png",
        ),
        plot_comparison(
            times,
            None,
            "Tiempo (s)",
            "Memoria (MB)",
            "Comparación de Tiempo de Cómputo y Uso de Memoria entre BM25, DPR y Reranking",
            "img/tiempo_memoria_comparacion.png",
        ),
    ]
    idx = 0
    if model == "DPR":
        idx = 1
    elif model == "Reranking":
        idx = 2
    return (
        # respuestas del modelo seleccionado
        retrieved_docs[idx],
        precision_values[idx][0],
        recall_values[idx][0],
        times[idx][0],
        charts[0],
        charts[1],
    )


def setup_interface():
    """
    Creates and returns a gr.Interface object for the information retrieval system.

    Returns:
        gr.Interface: The interface object for the information retrieval system.
    """
    return gr.Interface(
        fn=search,
        inputs=[
            gr.Dropdown(choices=queries, label="Selecciona una Consulta"),
            gr.Dropdown(choices=["BM25", "DPR", "Reranking"], label="Modelo de Recuperación"),
        ],
        outputs=[
            gr.Textbox(label="Resultados"),
            gr.Textbox(label="Precisión"),
            gr.Textbox(label="Recall"),
            gr.Textbox(label="Tiempo de Ejecución"),
            gr.Image(type="filepath", label="Gráfica de Comparación de Precision y Recall"),
            gr.Image(type="filepath", label="Gráfica de Comparación de Tiempo de Cómputo y Uso de Memoria"),
        ],
        title="Sistema de Recuperación de Información",
        description="Realiza una búsqueda utilizando BM25,DPR y Reranking y compara las métricas de precisión, recall y tiempo de ejecución.",
    )


if __name__ == "__main__":
    main()
