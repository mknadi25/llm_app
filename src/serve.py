import argparse
import os
import time
from http import HTTPStatus
from typing import Dict

import ray
from fastapi import FastAPI
from ray import serve
from starlette.requests import Request

from src import evaluate, predict
from src.config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(
    title="LLM app",
    description="Classify machine learning projects",
    version="0.1",
)


@serve.deployment(num_replicas="1", ray_actor_options={"num_cpus": 2, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        print("✅ 1. Deployment __init__ started.")  # ADD THIS
        self.run_id = run_id
        self.threshold = threshold

        print("✅ 2. Setting MLflow tracking URI...")  # ADD THIS
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        print(f"✅ 3. Getting best checkpoint for run_id: {run_id}...")  # ADD THIS
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        print("✅ 4. Checkpoint successfully retrieved.")  # ADD THIS

        print("✅ 5. Loading predictor from checkpoint...")  # ADD THIS
        self.predictor = predict.TorchPredictor.from_checkpoint(best_checkpoint)
        print("✅ 6. Predictor successfully loaded. Deployment is ready.")

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/evaluate/")
    async def _evaluate(self, request: Request) -> Dict:
        data = await request.json()
        results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @app.post("/predict/")
    async def _predict(self, request: Request):
        data = await request.json()
        sample_ds = ray.data.from_items(
            [
                {
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "tag": "",
                }
            ]
        )
        results = predict.predict_proba(ds=sample_ds, predictor=self.predictor)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "other"

        return {"results": results}


# src/serve.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument(
        "--threshold", type=float, default=0.9, help="threshold for `other` class."
    )
    args = parser.parse_args()

    # This part is the same
    ray.init(
        runtime_env={
            "env_vars": {"GITHUB_USERNAME": os.environ.get("GITHUB_USERNAME", "")}
        }
    )
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))

    # --- THIS NEW BLOCK WILL FIX THE PROBLEM ---
    try:
        print("Server is running. Press Ctrl+C to stop.")
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Shutting down server.")
        serve.shutdown()
    # --------------------------------------------
