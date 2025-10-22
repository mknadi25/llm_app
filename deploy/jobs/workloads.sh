#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD
mkdir -p results # Use -p to avoid errors if the directory already exists

# =================================================================
# DEBUGGING FLAGS
# Set a flag to `true` to run that stage, `false` to skip it.
# =================================================================
RUN_TEST_DATA=true
RUN_TEST_CODE=true
RUN_TRAIN=true
RUN_EVALUATE=true
RUN_TEST_MODEL=true
RUN_SAVE_ARTIFACTS=true
# =================================================================


# Test data
if [ "$RUN_TEST_DATA" = true ]; then
    echo "--- 1. RUNNING DATA TESTS ---"
    export RESULTS_FILE=results/test_data_results.txt
    export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
    python -m pytest --dataset-loc=$DATASET_LOC tests/data --verbose --disable-warnings > $RESULTS_FILE
    cat $RESULTS_FILE
fi


# Test code
if [ "$RUN_TEST_CODE" = true ]; then
    echo "--- 2. RUNNING CODE TESTS ---"
    export RESULTS_FILE=results/test_code_results.txt
    python -m pytest tests/code --verbose --disable-warnings > $RESULTS_FILE
    cat $RESULTS_FILE
fi


# Train
if [ "$RUN_TRAIN" = true ]; then
    echo "--- 3. RUNNING TRAINING JOB ---"
    export EXPERIMENT_NAME="llm_job"
    export RESULTS_FILE=results/training_results.json
    export DATASET_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
    export TRAIN_LOOP_CONFIG='{"dropout_p": 0.5, "lr": 1e-4, "lr_factor": 0.8, "lr_patience": 3}'
    python -m src.train \
        --experiment-name "$EXPERIMENT_NAME" \
        --dataset-loc "$DATASET_LOC" \
        --train-loop-config "$TRAIN_LOOP_CONFIG" \
        --num-workers 2 \
        --cpu-per-worker 2 \
        --gpu-per-worker 0 \
        --num-epochs 10 \
        --batch-size 256 \
        --results-fp $RESULTS_FILE
fi


# This block depends on the training run, so we check if the results file exists
if [ "$RUN_EVALUATE" = true ] && [ -f "results/training_results.json" ]; then
    echo "--- 4. EXTRACTING RUN ID AND RUNNING EVALUATION ---"
    # Get and save run ID
    export RUN_ID=$(python -c "import os; from src import utils; d = utils.load_dict('results/training_results.json'); print(d['run_id'])")
    export RUN_ID_FILE=results/run_id.txt
    echo $RUN_ID > $RUN_ID_FILE  # used for serving later

    # Evaluate
    export RESULTS_FILE=results/evaluation_results.json
    export HOLDOUT_LOC="https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/holdout.csv"
    python -m src.evaluate \
        --run-id $RUN_ID \
        --dataset-loc $HOLDOUT_LOC \
        --results-fp $RESULTS_FILE
fi


# This block also depends on the training run
if [ "$RUN_TEST_MODEL" = true ] && [ -f "results/run_id.txt" ]; then
    echo "--- 5. RUNNING MODEL TESTS ---"
    export RUN_ID=$(cat results/run_id.txt) # Read RUN_ID from file
    RESULTS_FILE=results/test_model_results.txt
    python -m pytest --run-id=$RUN_ID tests/model --verbose --disable-warnings > $RESULTS_FILE
    cat $RESULTS_FILE
fi


# Save artifacts to Git LFS
if [ "$RUN_SAVE_ARTIFACTS" = true ]; then
    echo "--- 6. SAVING ARTIFACTS TO GIT LFS ---"
    git add .
    git commit -m "chore(jobs): save artifacts from workload run on $(date)"
    git push
    echo "Successfully pushed artifacts to remote."
fi

echo "--- WORKLOAD SCRIPT FINISHED ---"
