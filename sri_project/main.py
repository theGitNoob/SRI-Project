import os

from bm25 import bm25_retrieve, init_bm25
from dataset_loader import corpus, queries
from eval import exact_match_ratio

from sri_project.dpr import create_index, retrieve_top_k_passages

dpr_index = create_index(corpus[:100])
bm25 = init_bm25(corpus[:100])

bm25_match_list = []
dpr_match_list = []

os.system("clear")
for i in range(100):
    query = queries[i]
    expected_answer = [corpus[i]]

    # Resultados de BM25
    bm25_results_indices = bm25_retrieve(query, bm25, 1)
    bm25_results = [corpus[idx] for idx in bm25_results_indices]
    bm25_match_list.append(bm25_results_indices[0])

    # Resultados de DPR
    _, dpr_results_indices = retrieve_top_k_passages(dpr_index, query, 1)
    dpr_results = [corpus[idx] for idx in dpr_results_indices]
    dpr_match_list.append(dpr_results_indices[0])

print(f"Exact Match Ratios for BM25: {exact_match_ratio(bm25_match_list)}")
print(f"Exact Match Ratios for DPR: {exact_match_ratio(dpr_match_list)}")
