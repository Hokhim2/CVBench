import json
from datasets import Dataset, DatasetDict

with open("../Eval/example_QAs.json", "r", encoding="utf-8") as f:
    data = json.load(f)

test_dataset = Dataset.from_list(data)

dataset_dict = DatasetDict({
    "test": test_dataset
})

dataset_dict.save_to_disk("../Eval/mvr_dataset")
