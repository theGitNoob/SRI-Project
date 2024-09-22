import os
import pandas as pd
from collections import defaultdict

# Load the CSV file
csv_file = "sri_project/data/data.csv"
data = pd.read_csv(csv_file)

# Initialize a dictionary to hold the grouped data
grouped_data = defaultdict(lambda: {"question": "", "answers": []})

queries = []
corpus = []

used_ids = set()
used_answers = set()

id = -1
for _, row in data.iterrows():
    question_id = row["question_id"]

    question = row["question"]
    answer = row["answer"]
    label = row["label"]

    if question_id not in used_ids:
        used_ids.add(question_id)
        queries.append(row["question"])

    question_id = len(used_ids) - 1

    if answer not in used_answers:
        used_answers.add(answer)
        corpus.append(answer)
        id += 1

    # Update the dictionary
    grouped_data[question_id]["question"] = question
    # if label == 1:
    grouped_data[question_id]["answers"].append(id)


# clear console
os.system("clear")
