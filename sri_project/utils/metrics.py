import time

from torch import cosine_similarity

from sri_project.models.bm25 import bm25_retrieve
from sri_project.models.dpr import encode_query, retrieve_top_k_passages
from sri_project.utils.utils import get_retrieved_docs


def evaluate_performance(data, queries: list[str], bm25, dpr_index, query_index: int | None = None) -> tuple:
    """
    Evaluate the performance of the BM25 and DPR retrieval models.

    Args:
        data:  A dictionary containing the grouped data.
        queries (list): A list of queries to evaluate.
        bm25 (BM25): The BM25 retrieval model.
        dpr_index (DPRIndex): The DPR retrieval model index.
        query_index (int): The index of the query to evaluate. If None, all queries are evaluated.

    Returns:
        tuple: A tuple containing the time, memory usage, recall, and precision values for BM25, DPR, and Reranking.
    """

    times = [[] for _ in range(3)]
    recall_values = [[] for _ in range(3)]
    precision_values = [[] for _ in range(3)]
    retrieved_docs = [[] for _ in range(3)]

    for i in range(len(queries)):
        if query_index is not None and i != query_index:
            continue
        if i == 30:
            break

        query = queries[i]

        k = 5

        # BM25 evaluation
        start_time = time.time()

        bm25_results_indices, _ = bm25_retrieve(query, bm25, k)
        retrieved_docs[0].append(get_retrieved_docs(bm25_results_indices))

        times[0].append(time.time() - start_time)

        recall_values[0].append(recall_at_k(data[i]["answers"], bm25_results_indices))
        precision_values[0].append(precision_at_k(data[i]["answers"], bm25_results_indices))

        # DPR evaluation
        start_time = time.time()

        _, dpr_results_indices = retrieve_top_k_passages(dpr_index, query, k)
        retrieved_docs[1].append(get_retrieved_docs(dpr_results_indices))

        times[1].append(time.time() - start_time)

        recall_values[1].append(recall_at_k(data[i]["answers"], dpr_results_indices))
        precision_values[1].append(precision_at_k(data[i]["answers"], dpr_results_indices))

        # Reranking evaluation
        start_time = time.time()

        query_embedding = encode_query(query)
        bm25_results, bm25_scores = bm25_retrieve(query, bm25, 2 * k)
        docs = get_retrieved_docs(bm25_results)

        reranking_match_list = []

        for idx in range(len(docs)):
            doc_embedding = encode_query(docs[idx])
            similarity = cosine_similarity(query_embedding, doc_embedding)
            if similarity < 0.7:
                continue
            combined_score = 0.7 * similarity + (1 - 0.7) * bm25_scores[idx]
            reranking_match_list.append((bm25_results[idx], combined_score))

        reranking_match_list = sorted(reranking_match_list, key=lambda x: x[1], reverse=True)[:k]
        reranking_result_indices = [x[0] for x in reranking_match_list]
        retrieved_docs[2].append(get_retrieved_docs(reranking_result_indices))

        times[2].append(time.time() - start_time)

        recall_values[2].append(recall_at_k(data[i]["answers"], reranking_result_indices))
        precision_values[2].append(precision_at_k(data[i]["answers"], reranking_result_indices))

    if query_index is not None:
        print(f"Average Precision for BM25: {sum(precision_values[0]) / len(precision_values[0])}")
        print(f"Average Precision for DPR: {sum(precision_values[1]) / len(precision_values[1])}")
        print(f"Average Precision for Reranking: {sum(precision_values[2]) / len(precision_values[2])}")

        print("-" * 50)

        print(f"Average Recall for BM25: {sum(recall_values[0]) / len(recall_values[0])}")
        print(f"Average Recall for DPR: {sum(recall_values[1]) / len(recall_values[1])}")
        print(f"Average Recall for Reranking: {sum(recall_values[2]) / len(recall_values[2])}")

    return (times, recall_values, precision_values, retrieved_docs)


def recall_at_k(answers: list[int], match_list: list[int]) -> float:
    """
    Calculates the recall at k for a given list of answers.

    Parameters:
    answers (list[int]): A list of answers where each element represents the index of the correct answer.
    match_list (list[int]): A list with the indices of the matched documents.
    k (int): The number of documents to consider for calculating recall.

    Returns:
    float: The recall at k, which is the ratio of the number of correct answers found in the top k documents to the total number of answers.
    """
    count = 0
    for i in answers:
        if i in match_list:
            count += 1
    return count / len(answers)


def precision_at_k(answers: list[int], match_list: list[int]) -> float:
    """
    Calculates the precision at k for a given list of answers.

    Parameters:
    answers (list[int]): A list of answers where each element represents the index of the correct answer.
    match_list (list[int]): A list with the indices of the matched documents.

    Returns:
    float: The precision at k, which is the ratio of the number of correct answers found in the top k documents to the total number of documents retrieved.
    """
    count = 0
    if len(match_list) == 0:
        return 0
    for i in answers:
        if i in match_list:
            count += 1
    return count / len(match_list)
