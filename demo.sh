#!/usr/bin/env bash
set -euo pipefail

DOCKER_IMAGE="subnet-sandbox:latest"
TRAIN_DIR="$(pwd)/fineweb_train"
OUTPUT_DIR="$(pwd)/outputs"
QUEUE_DIR="$(pwd)/job_queue"
API_URL="http://127.0.0.1:8000"
EVAL_URL="http://127.0.0.1:8001"
SUBMISSIONS_URL="$API_URL/submissions"
HOTKEY="test_miner"
export DOCKER_IMAGE TRAIN_DIR OUTPUT_DIR QUEUE_DIR

cleanup() {
    echo "--- Cleaning up ---"
    if [[ -n "${API_PID-}" ]]; then
        echo "Killing API server (PID: $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
    fi
    if [[ -n "${EVAL_PID-}" ]]; then
        echo "Killing Evaluator (PID: $EVAL_PID)..."
        kill "$EVAL_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true

    rm -rf fineweb_train test_set_temp test.tar test.tar.gpg outputs miner.py job_queue
    echo "✓ Cleanup complete."
}
trap cleanup EXIT

echo "--- Setting up ---"
if [ ! -d "tokenizer" ]; then
    python3 download_tokenizer.py
else
    echo "✓ Tokenizer already exists, skipping download."
fi
python3 fetch_fineweb.py --rows 100000
./prepare_test_set.sh
mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/processing" "$QUEUE_DIR/completed" "$QUEUE_DIR/failed"
echo "✓ Setup complete."

echo "--- Building Docker image ---"
docker build -t "$DOCKER_IMAGE" .
echo "✓ Docker image built."

echo "--- Starting API Server and Evaluator ---"
mkdir -p "$OUTPUT_DIR"

DEMO_MODE=true python3 api_server.py &
API_PID=$!
echo "✓ API server started in demo mode (PID: $API_PID)."

python3 evaluator.py &
EVAL_PID=$!
echo "✓ Evaluator started (PID: $EVAL_PID)."

echo "⏳ Waiting for services to be ready..."
sleep 10

echo "--- Submitting miner.py for evaluation (hotkey: $HOTKEY) ---"

JSON_PAYLOAD=$(python3 -c 'import json; print(json.dumps({
    "code": open("miner.py").read(),
    "hotkey": "'"$HOTKEY"'"
}))')

curl -s -X POST -H "Content-Type: application/json" -d "$JSON_PAYLOAD" "$SUBMISSIONS_URL" | (grep -q "queued" && echo "✓ Submission accepted.") || (echo "✗ Submission failed."; exit 1)

echo "⏳ Waiting for evaluation to complete for $HOTKEY..."
while [ ! -f "$OUTPUT_DIR/${HOTKEY}/score.json" ]; do
    if ! kill -0 $API_PID 2>/dev/null || ! kill -0 $EVAL_PID 2>/dev/null; then
        echo "✗ One of the services has crashed. Check logs."
        exit 1
    fi
    sleep 5
done

echo "--- Results for $HOTKEY ---"
cat "$OUTPUT_DIR/${HOTKEY}/score.json"
echo

if [ -f "$OUTPUT_DIR/scoreboard.json" ]; then
    echo "--- Final Scoreboard ---"
    cat "$OUTPUT_DIR/scoreboard.json"
    echo
fi
echo "✓ Demo finished successfully." 
