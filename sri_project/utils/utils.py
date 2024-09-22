from sri_project.models.bm25 import init_bm25
from sri_project.models.dpr import create_index
from sri_project.utils.dataset_loader import corpus


def get_retrieved_docs(retrieved_docs: list[int]) -> list[str]:
    """
    Retrieves the documents from the corpus based on the given list of document IDs.

    Args:
        retrieved_docs (list[int]): A list of document IDs.

    Returns:
        list[str]: A list of retrieved documents.

    """
    docs = []
    for doc_id in retrieved_docs:
        docs.append(corpus[doc_id])
    return docs


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
