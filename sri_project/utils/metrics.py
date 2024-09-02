import os
import time
import psutil

from sri_project.models.bm25 import bm25_retrieve
from sri_project.models.dpr import retrieve_top_k_passages


def memory_usage_psutil():
    """
    Returns the memory usage of the current process in megabytes.

    Uses the `psutil` library to get the memory information of the current process.
    The memory usage is calculated by dividing the resident set size (rss) by 1024 * 1024.

    Returns:
        float: The memory usage in megabytes.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def evaluate_performance(queries: list[str], bm25, dpr_index) -> tuple:
    """
    Evaluate the performance of the BM25 and DPR retrieval models.

    Args:
        queries (list): A list of queries to evaluate.
        bm25 (BM25): The BM25 retrieval model.
        dpr_index (DPRIndex): The DPR retrieval model index.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - bm25_times (list): A list of execution times for BM25 retrieval.
            - dpr_times (list): A list of execution times for DPR retrieval.
            - bm25_memory_usage (list): A list of memory usage for BM25 retrieval.
            - dpr_memory_usage (list): A list of memory usage for DPR retrieval.
            - bm25_match_list (list): A list of matched indices for BM25 retrieval.
            - dpr_match_list (list): A list of matched indices for DPR retrieval.
    """
    bm25_times = []
    dpr_times = []
    bm25_memory_usage = []
    dpr_memory_usage = []
    bm25_match_list = []
    dpr_match_list = []

    for i in range(len(queries)):
        query = queries[i]

        # BM25 evaluation
        start_time = time.time()
        bm25_results_indices = bm25_retrieve(query, bm25, 1)
        bm25_match_list.append(bm25_results_indices[0])
        bm25_times.append(time.time() - start_time)
        bm25_memory_usage.append(memory_usage_psutil())

        # DPR evaluation
        start_time = time.time()
        _, dpr_results_indices = retrieve_top_k_passages(dpr_index, query, 1)
        dpr_match_list.append(dpr_results_indices[0])
        dpr_times.append(time.time() - start_time)
        dpr_memory_usage.append(memory_usage_psutil())

    return bm25_times, dpr_times, bm25_memory_usage, dpr_memory_usage, bm25_match_list, dpr_match_list


def exact_match_ratio(answers: list[int]) -> float:
    """
    Calculates the exact match ratio for a given list of answers.

    Parameters:
    answers (list[int]): A list of answers where each element represents the index of the correct answer.

    Returns:
    float: The exact match ratio, which is the ratio of correct answers to the total number of answers.
    """
    count = 0
    for idx, ans in enumerate(answers):
        if idx == ans:
            count += 1
    return count / len(answers)
