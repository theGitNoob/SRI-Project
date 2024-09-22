import numpy
import torch
from faiss import IndexFlatIP
from .dpr_models import context_encoder, context_tokenizer, question_encoder, question_tokenizer
from numpy import ndarray


def encode_passages(passages: list[str], max_length: int = 1024, batch_size: int = 10000) -> ndarray:
    """
    Encodes the given passages into embeddings using a context encoder.

    Args:
        passages (list[str]): A list of passages to be encoded.
        max_length (int, optional): The maximum length of the encoded passages. Defaults to 512.
        batch_size (int, optional): The batch size for encoding the passages. Defaults to 10000.

    Returns:
        numpy.ndarray: An array of embeddings representing the encoded passages.
    """

    all_embeddings = []

    for i in range(0, len(passages), batch_size):
        batch_passages = passages[i : i + batch_size]
        inputs = context_tokenizer(
            batch_passages, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        )

        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output

        all_embeddings.append(embeddings.numpy())

    return numpy.vstack(all_embeddings)


def create_faiss_index(embeddings: ndarray) -> IndexFlatIP:
    """
    Create a Faiss index for the given embeddings.

    Parameters:
    embeddings (numpy.ndarray): The embeddings to be indexed.

    Returns:
    faiss.IndexFlatIP: The Faiss index object.

    """
    dimension = embeddings.shape[1]
    index = IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def encode_query(query: str, max_length: int = 1024):
    """
    Encodes the given query using the question_tokenizer and question_encoder models.

    Args:
        query (str): The query to be encoded.
        max_length (int, optional): The maximum length of the encoded query. Defaults to 512.

    Returns:
        numpy.ndarray: The encoded query as a numpy array.
    """
    inputs = question_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        query_embedding = question_encoder(**inputs).pooler_output
    return query_embedding


def create_index(corpus: list[str]) -> IndexFlatIP:
    """
    Creates an index for the given corpus.

    Parameters:
    corpus (list[str]): A list of passages in the corpus.

    Returns:
    index: The created index.

    """
    passage_embeddings = encode_passages(corpus)
    index = create_faiss_index(passage_embeddings)
    return index


def retrieve_top_k_passages(index: IndexFlatIP, query: str, k: int = 3) -> tuple[ndarray, list[int]]:
    """
    Retrieve the top k passages from the given index based on the query.

    Parameters:
        index (object): The index object used for searching.
        query (str): The query string.
        k (int, optional): The number of passages to retrieve. Defaults to 5.

    Returns:
        list: A list of passage indices representing the top k passages.
    """
    D, results = index.search(encode_query(query), k)
    return D[0], results[0].tolist()
