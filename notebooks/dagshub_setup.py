import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/092914rkumar/mlops-mini-project.mlflow")

dagshub.init(repo_owner='092914rkumar', repo_name='mlops-mini-project', mlflow=True)

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", 0.85)