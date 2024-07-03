import mlflow
logged_model = 'runs:/27fc4150ce5f4ebca262fb51b4fb67d9/RandomForestClassifier'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


import pandas as pd
data = [[
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.98745,
                360.0,
                1.0,
                2.0,
                8.698
            ]]
print(loaded_model.predict(pd.DataFrame(data)))