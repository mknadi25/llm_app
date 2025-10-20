import pytest

from src import predict
from src.predict import TorchPredictor


def pytest_addoption(parser):
    parser.addoption(
        "--run-id", action="store", default=None, help="Run ID of model to use."
    )


@pytest.fixture(scope="module")
def run_id(request):
    return request.config.getoption("--run-id")


@pytest.fixture(scope="module")
def predictor(run_id):
    # If no run_id is provided via the command line, skip these tests
    if not run_id:
        pytest.skip("Skipping behavioral tests: no --run-id specified.")

    # If run_id exists, the code continues as before
    best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
    predictor = TorchPredictor.from_checkpoint(best_checkpoint)
    return predictor
