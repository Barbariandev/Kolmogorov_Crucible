from huggingface_hub import snapshot_download
import os

if __name__ == "__main__":
    repo_id = "deepseek-ai/DeepSeek-v3"
    local_dir = "tokenizer"
    
    print(f"Downloading tokenizer for {repo_id} to {local_dir}...")
    
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=["*tokenizer*", "*.json", "*.model"],
        local_dir_use_symlinks=False,
        ignore_patterns=["*.safetensors", "*.gguf"]
    )
    
    print(f"Tokenizer downloaded successfully.") 
