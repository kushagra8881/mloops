import mlflow
mlflow.set_tracking_uri("http://localhost:5000")


experiment_id=mlflow.create_experiment("my_experiment")

with mlflow.start_run(run_name='DecisionTreeClass') as run:
    mlflow.set_tag("model", "Decis")
mlflow.end_run()

n_estimators = 10
criteria = 'gini'
mlflow.log_param("n_estimators",n_estimators)
mlflow.log_param("criteria", criteria)
mlflow.log_metric("accuracy", 0.9)