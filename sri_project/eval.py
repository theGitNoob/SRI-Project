def exact_match_ratio(answers: list[int]) -> float:
    count = 0
    for idx, ans in enumerate(answers):
        if idx == ans:
            count += 1
    return count / len(answers)
