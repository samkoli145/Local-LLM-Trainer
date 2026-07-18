#!/bin/bash
# Start MLflow UI

mkdir -p mlflow/mlruns
mlflow ui --backend-store-uri ./mlflow/mlruns --host 0.0.0.0 --port 5000
