import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Download NLTK data files (e.g., tokenizers)
nltk.download("punkt")
nltk.download("punkt_tab")


def init_bm25(corpus: list[str]) -> BM25Okapi:
    """
    Initialize the BM25 model with the given corpus.

    Parameters:
    - corpus (list[str]): A list of documents representing the corpus.

    Returns:
    - BM25Okapi: The initialized BM25 model.

    """
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus[:100]]  # Limiting to 100 documents for simplicity
    return BM25Okapi(tokenized_corpus)


def bm25_retrieve(query: str, bm25: BM25Okapi, top_k: int = 3) -> list[int]:
    """
    Retrieve the top-k indices of documents based on the BM25 scores for a given query.

    Args:
        query (str): The query string.
        bm25 (BM25): The BM25 object used for scoring.
        top_k (int, optional): The number of top indices to retrieve. Defaults to 3.

    Returns:
        list[int]: The top-k indices of documents based on the BM25 scores.
    """
    tokenized_query = word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return top_k_indices
