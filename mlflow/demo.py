import os 
import mlflow
import argparse
import time

def eval_metrics(p1,p2):
    output=p1**2+p2**2
    return output


def main(input1,input2):
    mlflow.set_experiment("appp")
    with mlflow.start_run(run_name="exampel_demo"):
        mlflow.log_param("input1", input1)
        mlflow.log_param("input2", input2)
        metric=eval_metrics(input1,input2)
        mlflow.log_metric("output", metric)
        os.makedirs("dummy", exist_ok=True)
        with open("dummy/ex.txt", "w") as f:
            
            f.write("Hello World")
            f.write(f"ARTIFACT CREATED {time.asctime()}")
        mlflow.log_artifacts("dummy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", "-p1",type=int,default=5)
    parser.add_argument("--input2",'-p2', type=int,default=5)
    args = parser.parse_args()
    main(args.input1, args.input2)