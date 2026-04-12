import os
import json
import pickle
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from sklearn.cluster import KMeans,MiniBatchKMeans
from transformers import AutoTokenizer, AutoModel
import re

# ==================== Configuration ====================
MODEL_NAME ="/path/to/your/llama3-8b/"
INPUT_JSONL = "/path/to/your/llama3_gpt4.jsonl"
EMBEDDINGS_PATH = "/path/to/your/embedding_llama3_gpt4.pkl"
CLUSTERS_PATH = "/path/to/your/cluster_llama3_gpt4.pkl"
K_CLUSTERS = 1000
BATCH_SIZE = 32


# ==================== Utility Functions ====================
def content_weights(input_ids, tok, bonus_entity=1.3):
    """Calculate token weights"""
    w = np.ones(len(input_ids), dtype=np.float32)
    
    for i, tid in enumerate(input_ids):
        try:
            tok_str = tok.decode([int(tid)]).strip()
            if re.search(r"[\d%℃°]", tok_str):
                w[i] *= bonus_entity
            if len(tok_str) == 0 or re.fullmatch(r"[\s,.;:!?()\[\]{}\"'`~\-_/\\]+", tok_str):
                w[i] = 0.0
        except:
            w[i] = 1.0
    
    return w

@torch.no_grad()
def sentence_vec(text: str, tok, model, device, layers=(-4, -3, -2)):
    """Generate sentence vector"""
    if hasattr(model, 'module'):
        hidden_size = model.module.config.hidden_size
    else:
        hidden_size = model.config.hidden_size
    if not text.strip():
        return np.zeros(hidden_size, dtype=np.float32)
    
    try:
        toks = tok(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
        input_ids = toks['input_ids'].to(device)
        attention_mask = toks.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hs = out.hidden_states
        L = len(hs)
        
        sel = [L + i if i < 0 else i for i in layers]
        sel = [s for s in sel if 0 <= s < L]
        if not sel:
            sel = [L-1]
        
        H_list = []
        for idx in sel:
            h = hs[idx]
            if isinstance(h, tuple):
                h = h[0]
            if len(h.shape) == 3:
                h = h[0]
            H_list.append(h.to(torch.float32))
        
        H = torch.stack(H_list, dim=0).mean(0)
        H = torch.nan_to_num(H, nan=0.0)
        
        input_ids_list = input_ids[0].cpu().tolist()
        w_np = content_weights(input_ids_list, tok)
        if w_np.sum() < 1e-8:
            w_np = np.ones(len(input_ids_list), dtype=np.float32)
        
        w = torch.tensor(w_np, device=device, dtype=torch.float32).unsqueeze(1)
        Hw = (H * w).sum(dim=0) / w.sum().clamp(min=1e-8)
        Hw = torch.nan_to_num(Hw, nan=0.0)
        
        norm = Hw.norm(p=2)
        if norm < 1e-12:
            return np.zeros(hidden_size, dtype=np.float32)
        
        v = Hw / norm
        result = v.detach().cpu().numpy().astype(np.float32)
        return np.nan_to_num(result, nan=0.0)
        
    except Exception as e:
        print(f"Error generating sentence vector: {e}")
        return np.zeros(hidden_size, dtype=np.float32)


class InstructionDataset(Dataset):
    """Instruction dataset"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'idx': idx,
            'instruction': self.data[idx].get("instruction", "").strip()
        }


def setup_distributed():
    """Initialize distributed environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Distributed environment not detected, using single GPU mode")
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed environment"""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_data(input_jsonl, rank):
    """Load data from JSONL file"""
    if rank == 0:
        print(f"Reading file {input_jsonl}...")
    
    data = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                if rank == 0:
                    print(f"JSON parsing error at line {line_num}: {e}")
    
    if rank == 0:
        print(f"Total {len(data)} items loaded")
    
    return data


def generate_embeddings_distributed(data, tokenizer, model, device, rank, world_size):
    """Generate embeddings in parallel across multiple GPUs"""
    dataset = InstructionDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    
    local_embeddings = []
    local_indices = []
    
    if rank == 0:
        dataloader = tqdm(dataloader, desc="Generating embeddings")
    
    for batch in dataloader:
        indices = batch['idx'].numpy()
        instructions = batch['instruction']
        
        batch_embeddings = []
        for instruction in instructions:
            if not instruction or instruction == "":
                embedding = np.zeros(model.module.config.hidden_size if hasattr(model, 'module') else model.config.hidden_size, dtype=np.float32)
            else:
                embedding = sentence_vec(instruction, tokenizer, model, device)
            batch_embeddings.append(embedding)
        
        local_embeddings.extend(batch_embeddings)
        local_indices.extend(indices)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return local_indices, local_embeddings


def gather_embeddings(local_indices, local_embeddings, total_size, rank, world_size):
    """Gather embeddings from all GPUs to main process"""
    local_indices = np.array(local_indices)
    local_embeddings = np.stack(local_embeddings)
    
    if world_size == 1:
        return local_embeddings
    
    if rank == 0:
        embedding_dim = local_embeddings.shape[1]
        all_embeddings = np.zeros((total_size, embedding_dim), dtype=np.float32)
    else:
        all_embeddings = None
    
    gathered_indices = [None] * world_size
    gathered_embeddings = [None] * world_size
    
    dist.all_gather_object(gathered_indices, local_indices)
    dist.all_gather_object(gathered_embeddings, local_embeddings)
    
    if rank == 0:
        for indices, embeddings in zip(gathered_indices, gathered_embeddings):
            all_embeddings[indices] = embeddings
        return all_embeddings
    
    return None


def perform_kmeans_clustering(embeddings, clusters_path, rank):
    """Perform K-means clustering (main process only)"""
    if rank != 0:
        return None
    
    if os.path.exists(clusters_path):
        print(f"Found existing clustering results: {clusters_path}, loading...")
        with open(clusters_path, 'rb') as f:
            cluster_labels = pickle.load(f)
        print(f"Successfully loaded clustering results with {K_CLUSTERS} clusters")
    else:
        print(f"Starting MiniBatch K-means clustering with K={K_CLUSTERS}...")
        
        print(f"Checking for NaN values in embeddings...")
        nan_mask = np.isnan(embeddings).any(axis=1)
        nan_count = nan_mask.sum()
        print(f"Found {nan_count} samples with NaN (ratio: {nan_count/len(embeddings)*100:.2f}%)")
        
        if nan_count > 0:
            nan_indices = np.where(nan_mask)[0]
            print(f"Sample indices with NaN (first 10): {nan_indices[:10].tolist()}")
            
            print("Filling NaN values with column means...")
            col_mean = np.nanmean(embeddings, axis=0)
            inds = np.where(np.isnan(embeddings))
            embeddings[inds] = np.take(col_mean, inds[1])
            embeddings = np.nan_to_num(embeddings, nan=0.0)
            
            print(f"Filling complete, remaining NaN count = {np.isnan(embeddings).sum()}")
        
        assert not np.isnan(embeddings).any(), "NaN values still exist!"
        
        kmeans = MiniBatchKMeans(
            n_clusters=K_CLUSTERS,
            random_state=42,
            batch_size=min(1000, len(embeddings) // 10),
            n_init=3,
            max_iter=100,
            verbose=0
        )

        print("Starting clustering...")
        cluster_labels = kmeans.fit_predict(embeddings)
        
        os.makedirs(os.path.dirname(clusters_path), exist_ok=True)
        with open(clusters_path, 'wb') as f:
            pickle.dump(cluster_labels, f)
        print(f"Successfully saved clustering results to {clusters_path}")
    
    return cluster_labels


def main():
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"Using {world_size} GPUs for training")
        print(f"Device: {device}")
    
    data = load_data(INPUT_JSONL, rank)
    
    if os.path.exists(EMBEDDINGS_PATH) and rank == 0:
        print(f"Found existing embedding file: {EMBEDDINGS_PATH}, loading...")
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"Successfully loaded {len(embeddings)} embeddings")
        
        if len(embeddings) != len(data):
            print(f"Warning: embedding count ({len(embeddings)}) doesn't match data count ({len(data)}), regenerating...")
            embeddings = None
    else:
        embeddings = None
    
    if world_size > 1:
        need_generate = torch.tensor([embeddings is None], dtype=torch.bool, device=device)
        dist.broadcast(need_generate, src=0)
        need_generate = need_generate.item()
    else:
        need_generate = (embeddings is None)
    
    if need_generate:
        if rank == 0:
            print("Starting to generate instruction embeddings...")
        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        model = model.to(device)
        
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        
        model.eval()
        
        local_indices, local_embeddings = generate_embeddings_distributed(
            data, tokenizer, model, device, rank, world_size
        )
        
        if world_size > 1:
            embeddings = gather_embeddings(local_indices, local_embeddings, len(data), rank, world_size)
        else:
            embeddings = np.stack(local_embeddings)
        
        if rank == 0:
            os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
            with open(EMBEDDINGS_PATH, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Successfully saved embeddings to {EMBEDDINGS_PATH}")
    
    cluster_labels = perform_kmeans_clustering(embeddings, CLUSTERS_PATH, rank)
    
    if rank == 0:
        print("All tasks completed!")
        print(f"Embeddings saved at: {EMBEDDINGS_PATH}")
        print(f"Clustering results saved at: {CLUSTERS_PATH}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()