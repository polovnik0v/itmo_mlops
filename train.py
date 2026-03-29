from clearml import Task, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import pickle, os

# task = Task.init(project_name="sentiment-lab", task_name="experiment-1")
task = Task.init(project_name="sentiment-lab", task_name="experiment-2")
task.execute_remotely(queue_name="students")  # уходит на агент

# Гиперпараметры для 1го эксперимента
# params = task.connect({
#     "C": 1.0,
#     "max_iter": 200,
#     "max_features": 5000
# })

# Гиперпараметры для 2го эксперимента
params = task.connect({
    "C": 0.1,
    "max_iter": 200,
    "max_features": 10000
})

# Датасет из ClearML
ds = Dataset.get(dataset_id="c5d80e5044f4473f805f5c910ffc498c")
path = ds.get_local_copy()

df = pd.read_csv(os.path.join(path, "sst2.csv"))
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Обучение
vec = TfidfVectorizer(max_features=params["max_features"])
X_tr = vec.fit_transform(X_train)
X_te = vec.transform(X_test)

model = LogisticRegression(C=params["C"], max_iter=params["max_iter"])
model.fit(X_tr, y_train)
preds = model.predict(X_te)

# Метрики
acc = accuracy_score(y_test, preds)
f1  = f1_score(y_test, preds, pos_label="positive")
task.get_logger().report_scalar("accuracy", "val", value=acc, iteration=0)
task.get_logger().report_scalar("f1",       "val", value=f1,  iteration=0)

# Confusion matrix
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
task.get_logger().report_matplotlib_figure("confusion_matrix", "val", fig)

# Сохраняем модель
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "vectorizer": vec}, f)
task.upload_artifact("model", artifact_object="model.pkl")

print(f"acc={acc:.3f}  f1={f1:.3f}")