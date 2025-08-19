import json
import os
import re


def check_and_fix(raw_string):
    problem_pattern = re.compile(r'"Options": \[([^[\]]*?)(?<!\])\s*,\s*"Answer":')

    if problem_pattern.search(raw_string):
        fixed = problem_pattern.sub(r'"Options": [\1], "Answer":', raw_string)
        return fixed

    return raw_string

def simplify_answer(answer, options):
    normalized_options = [opt.strip().strip(".").lower() for opt in options]
    if set(normalized_options) == {"yes", "no"}:
        return answer.strip(".")

    if "." in answer:
        return answer.split(".")[0].strip()
    return answer.strip()


def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for qa in data.get("QA_pairs", []):
        answer = qa.get("Answer", "")
        options = qa.get("Options", [])
        qa["Answer"] = simplify_answer(answer, options)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_directory(directory):
    for file in os.listdir(directory):
        fipa=os.path.join(directory, file)
        for filename in os.listdir(fipa):
            if filename.endswith(".json") and "QA" in filename:
                full_path = os.path.join(fipa, filename)
                postprocee(full_path)
                process_file(full_path)
                print(f"Processed: {full_path}")


def postprocee(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_string = data[0].replace("```json\n", "").replace("```", "")
    fixed_str=check_and_fix(raw_string)
    data1 = json.loads(fixed_str)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data1, f, indent=2, ensure_ascii=False)
process_directory("./dataset")
