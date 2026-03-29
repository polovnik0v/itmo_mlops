from datasets import load_dataset
from clearml import Dataset
import pandas as pd

# Загружаем SST-2 (маленький, ~67k строк)
raw = load_dataset("sst2", split="train[:2000]")  # берём 2000 строк — достаточно для лабы
df = pd.DataFrame({"text": raw["sentence"], "label": raw["label"]})
df["label"] = df["label"].map({0: "negative", 1: "positive"})
df.to_csv("sst2.csv", index=False)
print("Сохранено строк:", len(df))

# Загружаем в ClearML
dataset = Dataset.create(
    dataset_name="sst2_sentiment",
    dataset_project="sentiment-lab"
)
dataset.add_files("sst2.csv")
dataset.finalize(auto_upload=True)

print("Dataset ID:", dataset.id)  # СОХРАНИ ЭТОТ ID