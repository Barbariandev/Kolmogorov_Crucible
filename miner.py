import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import json
from pathlib import Path
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - MINER - %(levelname)s - %(message)s')

TOKENIZER_PATH = "tokenizer"
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    VOCAB_SIZE = tokenizer.vocab_size
    logging.info(f"Loaded tokenizer with vocab_size: {VOCAB_SIZE}")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    VOCAB_SIZE = 128000  
    tokenizer = None

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "n_embd": 512,      
    "n_head": 8,        
    "n_layer": 6,       
    "block_size": 2048, 
    "dropout": 0.1,
}

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head
        
        self.qkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["n_embd"], 4 * config["n_embd"])
        self.fc2 = nn.Linear(4 * config["n_embd"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class SimpleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.position_embedding = nn.Embedding(config["block_size"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])
        
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx):
        B, T = idx.size()
        
        if T > self.config["block_size"]:
            max_pos = min(T, self.config["block_size"])
            pos_ids = torch.arange(max_pos, device=idx.device)
            if T > self.config["block_size"]:
                extra_pos = torch.full((T - self.config["block_size"],), self.config["block_size"] - 1, device=idx.device)
                pos_ids = torch.cat([pos_ids, extra_pos])
        else:
            pos_ids = torch.arange(T, device=idx.device)
        
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos_ids)
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits

def load_training_data(train_path: str, max_tokens=50000):
    """Load and tokenize training data efficiently"""
    logging.info(f"Loading training data from {train_path}")
    
    data_files = list(Path(train_path).glob("*.txt"))
    if not data_files:
        logging.warning("No training files found, creating dummy data")
        dummy_text = "The quick brown fox jumps over the lazy dog. " * 1000
        tokens = tokenizer.encode(dummy_text)[:max_tokens]
        return torch.tensor(tokens, dtype=torch.long)
    
    all_tokens = []
    total_tokens = 0
    
    for file_path in data_files:
        if total_tokens >= max_tokens:
            break
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                total_tokens += len(tokens)
                
                if total_tokens >= max_tokens:
                    all_tokens = all_tokens[:max_tokens]
                    break
                    
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            continue
    
    if not all_tokens:
        dummy_text = "The quick brown fox jumps over the lazy dog. " * 1000
        all_tokens = tokenizer.encode(dummy_text)[:max_tokens]
    
    logging.info(f"Loaded {len(all_tokens)} tokens for training")
    return torch.tensor(all_tokens, dtype=torch.long)

def train(train_path: str):
    """Train a simple GPT model efficiently within 20 minutes"""
    logging.info(f"--- GPT Training Started ---")
    start_time = time.time()
    
    data = load_training_data(train_path, max_tokens=100000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGPT(MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    
    batch_size = 2
    block_size = 1024
    num_epochs = 3
    
    model.train()
    total_loss = 0
    num_batches = 0
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        for i in range(0, len(data) - block_size, batch_size * block_size):
            if time.time() - start_time > 18 * 60:  
                logging.info("time limit reached")
                break
                
            batch_data = []
            batch_targets = []
            
            for j in range(batch_size):
                start_idx = i + j * block_size
                if start_idx + block_size + 1 >= len(data):
                    break
                    
                x = data[start_idx:start_idx + block_size]
                y = data[start_idx + 1:start_idx + block_size + 1]
                batch_data.append(x)
                batch_targets.append(y)
            
            if len(batch_data) == 0:
                break
                
            batch_data = torch.stack(batch_data).to(device)
            batch_targets = torch.stack(batch_targets).to(device)
            
            logits = model(batch_data)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 50 == 0:
                avg_loss = total_loss / num_batches
                elapsed = time.time() - start_time
                logging.info(f"Batch {num_batches}, Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")
        
        if time.time() - start_time > 18 * 60:
            break
    
    model_path = "/tmp/model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path)
    
    elapsed_time = time.time() - start_time
    avg_loss = total_loss / max(num_batches, 1)
    logging.info(f"--- Training Finished in {elapsed_time:.1f}s, Avg Loss: {avg_loss:.4f} ---")
    
    return model

def inference(model, batched_tokens: list[list[int]]) -> torch.Tensor:
    """
    Run inference on a batch of token sequences.
    Returns logits for next token prediction at each position.
    """
    if not batched_tokens:
        logging.warning("Received empty batch for inference")
        return torch.empty((0, 0, VOCAB_SIZE))
    
    batch_size = len(batched_tokens)
    seq_len = len(batched_tokens[0])
    
    logging.info(f"--- GPT Inference Running ---")
    logging.info(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not hasattr(model, 'forward'):
        model_path = "/tmp/model.pt"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model = SimpleGPT(checkpoint['config']).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info("Loaded trained model from checkpoint")
        else:
            model = SimpleGPT(MODEL_CONFIG).to(device)
            logging.warning("No checkpoint found, using random model")
    
    model.eval()
    
    with torch.no_grad():
        input_ids = torch.tensor(batched_tokens, dtype=torch.long, device=device)
        
        logits = model(input_ids)
        
        logits = logits.cpu()
    
    logging.info(f"Returning logits of shape: {logits.shape}")
    return logits
