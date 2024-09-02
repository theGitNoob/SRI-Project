from datasets import load_dataset

# Load the TriviaQA dataset from Hugging Face

dataset = load_dataset("toughdata/quora-question-answer-dataset")

# Extract passages (answers) and queries (questions)
corpus = [item["answer"] for item in dataset["train"]]
queries = [item["question"] for item in dataset["train"]]
