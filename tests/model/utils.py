import ray

from src import predict


def get_label(text, predictor):
    if not isinstance(text, str):
        raise TypeError(f"get_label: expected text to be str, got {type(text)!r}")
    if predictor is None:
        raise ValueError("get_label: predictor must not be None")

    sample_ds = ray.data.from_items(
        [{"title": text, "description": "", "tag": "other"}]
    )
    results = predict.predict_proba(ds=sample_ds, predictor=predictor)
    return results[0]["prediction"]
