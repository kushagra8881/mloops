import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_data():
    URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv(URL, sep=";")
        return df
    except Exception as e:
        raise e
def eval_metrics(actual, pred):
    a=[]
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    a.append(rmse)
    a.append(mae)
    a.append(r2)
    return rmse, mae, r2,a
def main(alpha,l1_ration):
    df=load_data()
    target = "quality"
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    mlflow.set_experiment("wine_quality_11")
    with mlflow.start_run(run_name="12"):
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ration)
        model=ElasticNet(alpha=alpha, l1_ratio=l1_ration, random_state=42)
        model.fit(X_train, y_train)
        predicted_qualities = model.predict(X_test)
        rmse, mae, r2,a = eval_metrics(y_test, predicted_qualities)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("a", a)
        mlflow.sklearn.log_model(model, "model")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a",type=float,default=0.2)
    parser.add_argument("--l1_ratio",'-l1', type=float,default=0.5)
    args = parser.parse_args()
    main(args.alpha, args.l1_ratio)