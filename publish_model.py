from clearml import Task, OutputModel

task = Task.get_task(task_id="91ab1e3cfdf54f16af7da1ace04ae814")

output_model = OutputModel(
    task=task,
    name="sentiment-v1",
    framework="scikit-learn",
    tags=["sentiment", "tfidf", "logistic-regression"]
)

# Скачиваем артефакт и публикуем
artifact_path = task.artifacts["model"].get_local_copy()
output_model.update_weights(weights_filename=artifact_path)
output_model.publish()

print("Model ID:", output_model.id)
print("Опубликовано в Registry")