import mlflow
mlflow.set_tracking_uri("http://localhost:5000")


experiment_id=mlflow.create_experiment("my_experiment")

