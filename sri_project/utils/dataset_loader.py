import pandas as pd
from collections import defaultdict

# Load the CSV file
csv_file = "sri_project/data/data.csv"
data = pd.read_csv(csv_file)

# Initialize a dictionary to hold the grouped data
grouped_data = defaultdict(lambda: {"question": "", "answers": []})

queries = []
corpus = []

for _, row in data.iterrows():
    question_id = row["question_id"]
    question = row["question"]
    answer = row["answer"]

    corpus.append(answer)
    queries.append(question)

    # Update the dictionary
    grouped_data[question_id]["question"] = question
    grouped_data[question_id]["answers"].append(answer)

# Convert defaultdict to a regular dictionary
grouped_data = dict(grouped_data)


# Delete repeated questions
queries = list(set(queries))
