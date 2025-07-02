
import argparse, tqdm, os, pathlib, itertools, time
from datasets import load_dataset

def main(out_dir: str, rows: int, skip: int):
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    for idx, row in enumerate(tqdm.tqdm(itertools.islice(ds, skip, skip + rows),
                                        total=rows, desc="writing")):
        text = row["text"]
        (out_dir / f"{idx:06d}.txt").write_text(text, encoding="utf-8")
    print(f"✓ wrote {rows} docs → {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000,
                    help="how many documents to keep")
    ap.add_argument("--skip", type=int, default=0,
                    help="how many documents to skip from the beginning")
    ap.add_argument("--out", default="fineweb_train")
    args = ap.parse_args()
    main(args.out, args.rows, args.skip)
    time.sleep(2) 
