import os, json, shutil, logging, subprocess, time, sys
from pathlib import Path
import psutil
from fastapi import FastAPI, Request
import uvicorn
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

QUEUE_DIR = Path(os.getenv("QUEUE_DIR", "job_queue/pending"))
PROCESSING_DIR = Path(os.getenv("PROCESSING_DIR", "job_queue/processing"))
COMPLETED_DIR = Path(os.getenv("COMPLETED_DIR", "job_queue/completed"))
FAILED_DIR = Path(os.getenv("FAILED_DIR", "job_queue/failed"))

for d in [QUEUE_DIR, PROCESSING_DIR, COMPLETED_DIR, FAILED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT", 7200)) 
TRAIN_DIR   = os.getenv("TRAIN_DIR",   "fineweb_train")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR",  "outputs")
TEST_AES_HEX= os.getenv("TEST_AES_HEX", "deadbeefcafebabe")
DOCKER_IMG  = os.getenv("DOCKER_IMAGE", "subnet-sandbox:latest")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

MAX_LINES   = 10_000
MAX_BYTES   = 500_000
SCOREBOARD_FILE = Path(OUTPUT_DIR) / "scoreboard.json"
BASELINE_PPL = 50.0

def load_scoreboard():
    if not SCOREBOARD_FILE.exists():
        return {"baseline_ppl": BASELINE_PPL, "best_ppl": BASELINE_PPL, "scores": {}}
    try:
        return json.loads(SCOREBOARD_FILE.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
         return {"baseline_ppl": BASELINE_PPL, "best_ppl": BASELINE_PPL, "scores": {}}

def save_scoreboard(data):
    SCOREBOARD_FILE.write_text(json.dumps(data, indent=2))

def _calculate_and_update_score(uid: str, ppl: float):
    scoreboard = load_scoreboard()
    best_ppl = scoreboard.get("best_ppl", BASELINE_PPL)
    base_ppl = scoreboard.get("baseline_ppl", BASELINE_PPL)
    
    score = 0.0
    if ppl < best_ppl and ppl > 0:
        score = (best_ppl - ppl) / base_ppl
        logging.info(f"[scoring] New best PPL for {uid}: {ppl:.3f}. Score: {score:.4f}")
        scoreboard["best_ppl"] = ppl
    else:
        logging.info(f"[scoring] PPL for {uid} ({ppl:.3f}) did not beat best ({best_ppl:.3f}). Score: 0.0")
    
    if "scores" not in scoreboard:
        scoreboard["scores"] = {}
    scoreboard["scores"][uid] = {"ppl": ppl, "score": score, "timestamp": time.time()}
    save_scoreboard(scoreboard)
    logging.info("[weights] pseudo-update: UID %s -> score=%.4f", uid, score)

def _save_code(code: str, hotkey: str) -> Path:
    if len(code) > MAX_BYTES or code.count("\n") > MAX_LINES:
        logging.error(f"Code for {hotkey} exceeds size limits. Skipping.")
        raise ValueError("Code exceeds size limits")
    
    temp_code_dir = Path("/tmp/miner_code")
    temp_code_dir.mkdir(exist_ok=True)
    fpath = temp_code_dir / f"{hotkey.replace('/', '_')}_{int(time.time())}.py"
    fpath.write_text(code)
    return fpath

def run_sandbox(uid: str, code: str):
    code_path = _save_code(code, uid)
    cpus = max(1, int(os.cpu_count() * 0.9))
    mem_bytes = int(psutil.virtual_memory().total * 0.9)
    out_dir = Path(OUTPUT_DIR) / uid.replace('/', '_')
    
    cmd = [
        "docker", "run", "--rm", "--network=none",
        f"--cpus={cpus}", f"--memory={mem_bytes}", "--gpus=all",
        "-e", "RUN_MODE=container", "-e", f"TEST_KEY={TEST_AES_HEX}",
        "-v", f"{Path(TRAIN_DIR).resolve()}:/train:ro",
        "-v", f"{code_path.resolve()}:/workspace/miner.py:ro",
        "-v", f"{out_dir.resolve()}:/output",
        DOCKER_IMG
    ]
    logging.info("[evaluator] launching sandbox for UID=%s", uid)
    try:
        proc = subprocess.run(cmd, timeout=JOB_TIMEOUT, check=True, capture_output=True, text=True)
        with open(out_dir / "score.json") as fh:
            ppl = json.load(fh)["ppl"]
        logging.info("[evaluator] UID %s perplexity %.3f", uid, ppl)
        _calculate_and_update_score(uid, ppl)
    except subprocess.TimeoutExpired:
        logging.warning("[evaluator] UID %s timed out", uid)
        raise
    except subprocess.CalledProcessError as e:
        logging.error(f"[evaluator] UID {uid} failed with exit code {e.returncode}")
        logging.error(f"STDOUT: {e.stdout}")
        logging.error(f"STDERR: {e.stderr}")
        raise
    except Exception as e:
        logging.exception("[evaluator] UID %s failed: %s", uid, e)
        raise
    finally:
        if code_path.exists():
            code_path.unlink()

def worker(stop_event):
    logging.info("[worker] Starting worker loop, watching for jobs in %s", QUEUE_DIR)
    while not stop_event.is_set():
        job_files = sorted(QUEUE_DIR.glob("*.json"))
        if not job_files:
            time.sleep(5)
            continue
        
        job_path = job_files[0]
        processing_path = PROCESSING_DIR / job_path.name
        
        try:
            shutil.move(str(job_path), str(processing_path))
            logging.info(f"[worker] Picked up job: {job_path.name}")
            
            job_data = json.loads(processing_path.read_text())
            hotkey = job_data["hotkey"]
            code = job_data["code"]
            
            run_sandbox(hotkey, code)
            
            shutil.move(str(processing_path), str(COMPLETED_DIR / job_path.name))
            logging.info(f"[worker] Finished job: {job_path.name}")
        except Exception as e:
            logging.error(f"[worker] Failed to process job {job_path.name}: {e}")
            shutil.move(str(processing_path), str(FAILED_DIR / job_path.name))

app = FastAPI(title="Subnet - Evaluator")

@app.get("/scoreboard")
async def get_scoreboard():
    return load_scoreboard()

if __name__ == "__main__":
    stop_event = threading.Event()
    worker_thread = threading.Thread(target=worker, args=(stop_event,))
    worker_thread.start()

    config = uvicorn.Config(app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    
    logging.info("[evaluator] Starting API on http://0.0.0.0:8001")
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Caught interrupt, shutting down...")
    finally:
        stop_event.set()
        worker_thread.join()
        logging.info("[evaluator] Shut down gracefully.")
    sys.exit(0) 
