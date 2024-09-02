from sri_project.models.bm25 import init_bm25
from sri_project.models.dpr import create_index
from sri_project.utils.metrics import exact_match_ratio, evaluate_performance
from sri_project.utils.plot import plot_and_save_graph
from sri_project.utils.dataset_loader import corpus, queries


def initialize_indexes(corpus):
    """
    Initializes the BM25 and DPR indexes for the given corpus.

    Parameters:
    corpus (list): A list of documents representing the corpus.

    Returns:
    tuple: A tuple containing the BM25 index and the DPR index.
    """

    bm25 = init_bm25(corpus)

    dpr_index = create_index(corpus)

    return bm25, dpr_index


def main():
    """
    This is the main function that performs the following tasks:
    1. Initializes indexes and measures memory usage.
    2. Evaluates the performance of retrieval.
    3. Prints the Exact Match Ratios for BM25 and DPR.
    4. Plots and saves the results of time and memory usage comparison between BM25 and DPR.
    """

    # Indexes initialization
    bm25, dpr_index = initialize_indexes(corpus[:100])

    # Performance evaluation
    bm25_times, dpr_times, bm25_memory_usage, dpr_memory_usage, bm25_match_list, dpr_match_list = evaluate_performance(
        queries[:100], bm25, dpr_index
    )

    # Exact Match Ratios results
    print(f"Exact Match Ratios for BM25: {exact_match_ratio(bm25_match_list)}")
    print(f"Exact Match Ratios for DPR: {exact_match_ratio(dpr_match_list)}")

    # Plotting and saving the results
    plot_and_save_graph(
        x=range(100),
        y1=bm25_times,
        y2=dpr_times,
        xlabel="Consultas",
        ylabel="Tiempo (s)",
        title="Comparación de Tiempo de Cómputo entre BM25 y DPR",
        filename="tiempo_computo_comparacion",
    )

    plot_and_save_graph(
        x=range(100),
        y1=bm25_memory_usage,
        y2=dpr_memory_usage,
        xlabel="Consultas",
        ylabel="Memoria (MB)",
        title="Comparación de Uso de Memoria entre BM25 y DPR",
        filename="uso_memoria_comparacion",
    )


if __name__ == "__main__":
    main()
