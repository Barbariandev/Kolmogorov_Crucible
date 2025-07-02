import sys, os, subprocess, json, logging, tarfile, tempfile, importlib.util, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s CONTAINER %(levelname)s %(message)s")

TEST_KEY = os.getenv("TEST_KEY")
WORKSPACE_DIR = Path("/workspace")
OUTPUT_DIR = Path("/output")
TEST_SET_ARCHIVE = Path("/root/test.tar.gpg")
TOKENIZER_PATH = "/root/tokenizer"

SEQUENCE_LENGTH = 2048

def decrypt_and_extract(key: str, archive_path: Path, temp_dir: Path) -> Path:
    logging.info("Decrypting and extracting test set...")
    decrypted_tar_path = temp_dir / "test.tar"
    
    gpg_cmd = [
        "gpg", "--decrypt", "--quiet", "--batch", "--yes",
        "--passphrase", key,
        "-o", str(decrypted_tar_path),
        str(archive_path)
    ]
    subprocess.run(gpg_cmd, check=True)
    
    with tarfile.open(decrypted_tar_path) as tar:
        tar.extractall(path=temp_dir)
    
    test_file = temp_dir / "test.txt"
    if not test_file.exists():
        raise FileNotFoundError("Could not find 'test.txt' in the extracted archive.")
    logging.info("Test set ready.")
    return test_file

def load_and_tokenize(file_path: Path, tokenizer) -> np.ndarray:
    text = file_path.read_text()
    tokens = tokenizer(text, return_tensors="np", max_length=len(text) // 2, truncation=True)["input_ids"][0]
    return tokens.astype(np.uint16)

def make_windows(tokens: np.ndarray, seq_len: int) -> list[np.ndarray]:
    return [tokens[i : i + seq_len] for i in range(0, len(tokens) - seq_len + 1, seq_len)]

def calculate_perplexity(miner_inference_fn, model, all_tokens, vocab_size, batch_size=4) -> float:
    logging.info("Starting perplexity calculation...")
    windows = make_windows(all_tokens, SEQUENCE_LENGTH + 1)
    if not windows:
        logging.error("Not enough tokens to create any evaluation windows.")
        return float('inf')

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i : i + batch_size]
        
        contexts = [w[:-1].tolist() for w in batch_windows]
        labels = torch.tensor([w[1:] for w in batch_windows], dtype=torch.long)
        
        logits = miner_inference_fn(model, contexts)

        if not isinstance(logits, torch.Tensor) or logits.shape != (len(contexts), SEQUENCE_LENGTH, vocab_size):
            raise ValueError(f"Invalid logits shape. Expected {(len(contexts), SEQUENCE_LENGTH, vocab_size)}, got {logits.shape if isinstance(logits, torch.Tensor) else type(logits)}")

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            reduction='sum'
        )
        total_nll += loss.item()
        total_tokens += labels.numel()

    avg_nll = total_nll / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_nll)
    logging.info(f"Calculated perplexity: {perplexity:.4f} from {total_tokens} tokens.")
    return perplexity

def main():
    try:
        logging.info("--- Container starting ---")

        miner_path = WORKSPACE_DIR / "miner.py"
        spec = importlib.util.spec_from_file_location("miner", miner_path)
        miner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(miner)
        logging.info("Successfully imported miner code.")

        logging.info("Calling miner.train()...")
        start_time = time.time()
        model = miner.train(train_path="/train")
        train_duration = time.time() - start_time
        logging.info(f"miner.train() finished in {train_duration:.2f}s.")

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        vocab_size = tokenizer.vocab_size
        logging.info(f"Using tokenizer vocab_size: {vocab_size}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file_path = decrypt_and_extract(TEST_KEY, TEST_SET_ARCHIVE, Path(temp_dir))
            all_tokens = load_and_tokenize(test_file_path, tokenizer)
            ppl = calculate_perplexity(miner.inference, model, all_tokens, vocab_size)
        
        logging.info(f"Evaluation complete. Final PPL: {ppl:.4f}")

        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "score.json").write_text(json.dumps({"ppl": ppl, "train_duration": train_duration}))
        logging.info("--- Container finished successfully ---")

    except Exception as e:
        logging.exception("An error occurred in the container")
        OUTPUT_DIR.mkdir(exist_ok=True)
        (OUTPUT_DIR / "error.log").write_text(f"{type(e).__name__}: {e}\n{sys.exc_info()}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
